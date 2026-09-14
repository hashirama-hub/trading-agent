import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

from src.agent.state import AgentState, TradingPlan, Decision, PortfolioState
from src.agent.prompts import build_system_prompt
from src.tools.binance_client import OrderRequest, PortfolioState as BSPortfolioState
from src.risk.guard import RiskGuard, RiskDecision
from src.memory.store import AgentMemory

logger = logging.getLogger(__name__)


@tool
def get_market_data(symbol: str, timeframe: str = "1h") -> Dict:
    """Fetch OHLCV + orderbook for symbol."""
    return {}


@tool
def get_portfolio_state() -> Dict:
    """Get current portfolio: positions, balance, PnL."""
    return {}


@tool
def place_order(symbol: str, side: str, qty: float, price: float, stop_loss: float, take_profit: float) -> Dict:
    """Place an order with risk parameters."""
    return {}


class AgentNodes:
    def __init__(
        self,
        tools: List,
        memory: AgentMemory,
        risk_guard: RiskGuard,
        deepseek_model: str = "deepseek-v4.1-flash",
        deepseek_api_key: str = "",
        max_iterations: int = 10,
    ):
        self.tools = {t.name: t for t in tools}
        self.memory = memory
        self.risk_guard = risk_guard
        self.max_iterations = max_iterations
        self._deepseek_model = deepseek_model
        self._deepseek_api_key = deepseek_api_key
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            api_key = self._deepseek_api_key or __import__("os").environ.get("DEEPSEEK_API_KEY", "")
            self._llm = ChatOpenAI(
                model=self._deepseek_model,
                api_key=api_key,
                temperature=0.1,
                max_tokens=4096,
                base_url="https://api.deepseek.com/v1",
            )
            self._llm = self._llm.bind_tools(self.tools)
        return self._llm

    def observe_node(self, state: AgentState) -> AgentState:
        """Fetch market data and update state."""
        symbols = list(state.get("market_data", {}).keys())
        if not symbols:
            symbols = ["BTCUSDT", "ETHUSDT"]

        market_data = {}
        for symbol in symbols:
            market_data[symbol] = state["market_data"].get(symbol, {})

        state["market_data"] = market_data
        state["iteration"] += 1
        logger.info(f"Observe: {len(symbols)} symbols loaded, iteration {state['iteration']}")
        return state

    def reason_node(self, state: AgentState) -> AgentState:
        """LLM reasoning with tools (ReAct loop)."""
        context = self.memory.get_context()
        portfolio = state.get("portfolio", {})
        recent_trades = self.memory.get_recent_trades(limit=5)

        system_content = build_system_prompt(equity=portfolio.get("equity", 10000))

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=self._build_user_message(state, context, recent_trades)),
        ]

        try:
            response = self.llm.invoke(messages)

            decision = self._parse_llm_response(response, state)
            state["last_decision"] = decision
            self.memory.add_turn(
                thought=decision.get("thought", ""),
                action=decision.get("action", "null"),
                result=str(decision),
            )
            logger.info(f"Reason: action={decision.get('action')}, confidence={decision.get('confidence', 0):.2f}")
        except Exception as e:
            logger.error(f"LLM reasoning error: {e}")
            state["errors"].append(str(e))

        return state

    def act_node(self, state: AgentState) -> AgentState:
        """Execute approved actions through Risk Guard."""
        decision = state.get("last_decision")
        if not decision or decision.get("type") != "action":
            return state

        action = decision.get("action")
        params = decision.get("params", {})

        if action == "place_order":
            order = OrderRequest(
                symbol=params.get("symbol", "BTCUSDT"),
                side=params.get("side", "BUY"),
                qty=params.get("qty", 0.01),
                price=params.get("price"),
                stop_loss=params.get("stop_loss"),
                take_profit=params.get("take_profit"),
            )

            portfolio = BSPortfolioState(
                equity=state["portfolio"]["equity"],
                available_margin=state["portfolio"]["available_margin"],
                positions=state["portfolio"]["positions"],
                daily_pnl=state["portfolio"]["daily_pnl"],
                max_drawdown_today=state["portfolio"]["max_drawdown_today"],
            )
            guard = RiskGuard(portfolio)
            result = guard.validate(order)

            if result.decision in (RiskDecision.APPROVE, RiskDecision.MODIFY):
                target_order = result.modified_order or order
                logger.info(f"Act: EXECUTING {target_order.side} {target_order.qty} {target_order.symbol}")
                self.memory.log_trade({
                    "symbol": target_order.symbol,
                    "decision": target_order.side,
                    "pnl": 0,
                    "reasoning": decision.get("thought", ""),
                    "regime": state.get("market_data", {}).get(target_order.symbol, {}).get("funding_rate", 0),
                    "outcome": "pending",
                    "model_version": "deepseek-v4.1-flash",
                })
            else:
                logger.warning(f"Act: REJECTED - {result.reason}")
                state["errors"].append(f"Order rejected: {result.reason}")
                self.memory.log_error(f"Order rejected: {result.reason}")

        elif action == "get_market_data":
            pass
        elif action == "get_portfolio_state":
            pass
        elif action == "cancel_order":
            pass
        elif action == "modify_order":
            pass

        return state

    def reflect_node(self, state: AgentState) -> AgentState:
        """Log decision, update memory, evaluate performance."""
        if state["iteration"] >= state["max_iterations"]:
            state["_next"] = END
        else:
            state["_next"] = "observe"

        logger.info(f"Reflect: iteration {state['iteration']}/{state['max_iterations']}, errors={len(state['errors'])}")
        return state

    def _build_user_message(self, state: AgentState, context: str, recent_trades: List[Dict]) -> str:
        market = state.get("market_data", {})
        portfolio = state.get("portfolio", {})

        return f"""Market Data: {market}
Portfolio: {portfolio}
Recent Trades: {recent_trades}
Memory Context: {context}
Iteration: {state['iteration']}

Analyze the market and decide the next action. Output JSON only.
"""

    def _parse_llm_response(self, response, state: AgentState) -> Decision:
        tool_calls = getattr(response, "tool_calls", None)

        if tool_calls:
            tc = tool_calls[0]
            return Decision(
                thought=response.content or "",
                action=tc["name"],
                params=tc["args"],
                risk_check={},
                confidence=0.8,
                type="action",
            )

        return Decision(
            thought=response.content or "",
            action="null",
            params={},
            risk_check={},
            confidence=0.5,
            type="reasoning",
        )