import json
import logging
from typing import List, Dict, Optional, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from src.agent.state import AgentState, TradingPlan, Decision, PortfolioState
from src.agent.prompts import build_system_prompt
from src.tools.binance_client import OrderRequest, PortfolioState as BSPortfolioState
from src.risk.guard import RiskGuard, RiskDecision
from src.memory.store import AgentMemory

logger = logging.getLogger(__name__)


@tool
def get_market_data(symbol: str, timeframe: str = "1h") -> Dict:
    """Fetch OHLCV + orderbook for symbol. Use timeframe: 5m, 15m, 1h, 4h."""
    from src.tools.market_data import get_market_data as _get_market_data
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    snapshot = loop.run_until_complete(_get_market_data(symbol, timeframe))
    return snapshot.dict() if hasattr(snapshot, 'dict') else snapshot


@tool
def get_portfolio_state() -> Dict:
    """Get current portfolio: positions, balance, PnL."""
    from src.tools.market_data import get_portfolio_state as _get_portfolio_state
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    state = loop.run_until_complete(_get_portfolio_state())
    return state.dict() if hasattr(state, 'dict') else state


@tool
def place_order(symbol: str, side: str, qty: float, price: float, stop_loss: float, take_profit: float) -> Dict:
    """Place a futures order with stop-loss and take-profit. Risk Guard validates before execution."""
    return {"status": "pending", "symbol": symbol, "side": side, "qty": qty, "price": price, "stop_loss": stop_loss, "take_profit": take_profit}


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
        """LLM reasoning with ReAct tool-calling loop."""
        context = self.memory.get_context()
        portfolio = state.get("portfolio", {})
        recent_trades = self.memory.get_recent_trades(limit=5)
        perf_stats = self.memory.get_performance_stats()

        system_content = build_system_prompt(
            equity=portfolio.get("equity", 10000),
            perf_stats=perf_stats,
        )

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=self._build_user_message(state, context, recent_trades)),
        ]

        try:
            for step in range(self.max_iterations):
                response = self.llm.invoke(messages)
                messages.append(AIMessage(content=response.content or "", tool_calls=getattr(response, "tool_calls", []) or []))

                tool_calls = getattr(response, "tool_calls", None)
                if not tool_calls:
                    break

                for tc in tool_calls:
                    tool_name = tc["name"]
                    tool_args = tc["args"]
                    if tool_name in self.tools:
                        tool_result = self.tools[tool_name].invoke(tool_args)
                    else:
                        tool_result = f"Unknown tool: {tool_name}"
                    messages.append(ToolMessage(content=str(tool_result), tool_call_id=tc["id"]))

            decision = self._parse_final_response(messages[-1] if messages else response, state, messages)
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
            self.risk_guard.portfolio = portfolio
            result = self.risk_guard.validate(order)

            if result.decision in (RiskDecision.APPROVE, RiskDecision.MODIFY):
                target_order = result.modified_order or order
                logger.info(f"Act: EXECUTING {target_order.side} {target_order.qty} {target_order.symbol}")
                self.memory.log_trade({
                    "symbol": target_order.symbol,
                    "decision": target_order.side,
                    "pnl": 0,
                    "reasoning": decision.get("thought", ""),
                    "regime": state.get("market_data", {}).get(target_order.symbol, {}).get("technicals", {}).get("regime", "unknown"),
                    "outcome": "pending",
                    "confidence": decision.get("confidence", 0),
                    "model_version": "deepseek-v4.1-flash",
                })
            else:
                logger.warning(f"Act: REJECTED - {result.reason}")
                state["errors"].append(f"Order rejected: {result.reason}")
                self.memory.log_error(f"Order rejected: {result.reason}")

        return state

    def reflect_node(self, state: AgentState) -> AgentState:
        """Evaluate decision quality, update memory with insights."""
        decision = state.get("last_decision")
        if decision:
            confidence = decision.get("confidence", 0)
            action = decision.get("action", "null")
            errors = len(state.get("errors", []))

            if confidence < 0.5:
                logger.warning(f"Reflect: Low confidence decision ({confidence:.2f})")
            if errors > 0:
                logger.warning(f"Reflect: {errors} errors in this cycle")

        if state["iteration"] >= state["max_iterations"]:
            state["_next"] = "END"
        else:
            state["_next"] = "observe"

        logger.info(f"Reflect: iteration {state['iteration']}/{state['max_iterations']}")
        return state

    def _build_user_message(self, state: AgentState, context: str, recent_trades: List[Dict]) -> str:
        market = state.get("market_data", {})
        portfolio = state.get("portfolio", {})

        market_summary = []
        for sym, data in market.items():
            tech = data.get("technicals", {})
            market_summary.append(
                f"{sym}: close={data.get('close', '?')} "
                f"RSI={tech.get('rsi_14', '?')} MACD={tech.get('macd_histogram', '?')} "
                f"ADX={tech.get('adx', '?')} Regime={tech.get('regime', '?')} "
                f"Trend={tech.get('trend_score', '?')} BB_width={tech.get('bb_width', '?')} "
                f"ATR={tech.get('atr_14', '?')} Volume_ratio={tech.get('volume_ratio', '?')}"
            )

        return f"""## CURRENT STATE
Portfolio: equity=${portfolio.get('equity', 0):,.2f}, daily_pnl=${portfolio.get('daily_pnl', 0):,.2f}, positions={len(portfolio.get('positions', []))}

## MARKET DATA
{chr(10).join(market_summary) if market_summary else 'No market data available'}

## RECENT TRADES
{self._format_recent_trades(recent_trades)}

## MEMORY
{context if context else 'No previous context'}

## YOUR TASK
Analyze the market using the 4-step framework. Use tools if you need more data.
Output your decision as JSON."""

    def _format_recent_trades(self, trades: List[Dict]) -> str:
        if not trades:
            return "No recent trades"
        lines = []
        for t in trades[:5]:
            lines.append(f"- {t.get('symbol', '?')}: {t.get('decision', '?')} | outcome={t.get('outcome', '?')} | pnl={t.get('pnl', '?')}")
        return "\n".join(lines)

    def _parse_final_response(self, response, state: AgentState, messages: List) -> Decision:
        content = response.content if hasattr(response, 'content') else str(response)

        try:
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                parsed = json.loads(content[json_start:json_end])
                confidence = parsed.get("confidence", 0.5)
                action = parsed.get("action", "null")

                if action and action != "null":
                    return Decision(
                        thought=parsed.get("thought", content),
                        action=action,
                        params=parsed.get("params", {}),
                        risk_check=parsed.get("risk_check", {}),
                        confidence=confidence,
                        type="action",
                    )
                return Decision(
                    thought=parsed.get("thought", content),
                    action="null",
                    params={},
                    risk_check={},
                    confidence=confidence,
                    type="reasoning",
                )
        except json.JSONDecodeError:
            pass

        return Decision(
            thought=content,
            action="null",
            params={},
            risk_check={},
            confidence=0.5,
            type="reasoning",
        )
