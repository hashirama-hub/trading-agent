# Hermes Agent Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace LangGraph agent loop with Hermes Agent as the main orchestrator, using DeepSeek as the LLM backend.

**Architecture:** Hermes Agent's `AIAgent.run_conversation()` handles the Observe→Reason→Act→Reflect loop. Trading tools registered as a Hermes toolset. Risk Guard validates every order deterministically before execution. Memory transitions to Hermes session DB + SOUL.md + AGENTS.md.

**Tech Stack:** hermes-agent 0.19.0, DeepSeek API (OpenAI-compatible), python-binance, Risk Guard (existing), Paper Trading Engine (existing)

**Spec:** docs/superpowers/specs/2026-09-15-hermes-agent-integration-design.md

## Global Constraints

- DeepSeek model: `deepseek-v4.1-flash` (env var `DEEPSEEK_MODEL`)
- Risk Guard is OUTSIDE LLM control — hard-coded validation in tool execution
- All tools return Pydantic models, never raw strings
- Every decision logged to immutable store (SQLite trade journal)
- API key in `.env` only, gitignored, never in code
- Python 3.12, hermes-agent 0.19.0, langgraph may be removed
- Binance testnet: `true` during development

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/tools/hermes_tools.py` | CREATE | Register trading tools via Hermes `register()` pattern |
| `src/agent/hermes_loop.py` | CREATE | Hermes agent wrapper with trading config |
| `src/memory/hermes_memory.py` | CREATE | Bridge existing episodic/semantic memory to Hermes |
| `src/agent/main.py` | MODIFY | Replace LangGraph entry point with Hermes loop |
| `.hermes/SOUL.md` | CREATE | Trading agent persona + hard rules (global) |
| `AGENTS.md` | CREATE | Project context for Hermes |
| `src/agent/nodes.py` | DELETE | Hermes handles the agent loop |
| `src/agent/graph.py` | DELETE | Hermes handles the agent loop |
| `tests/unit/test_hermes_tools.py` | CREATE | Unit tests for trading tools |
| `tests/unit/test_hermes_loop.py` | CREATE | Unit tests for Hermes agent wrapper |
| `tests/integration/test_hermes_e2e.py` | CREATE | E2E test with Hermes agent loop |

---

### Task 1: Install hermes-agent and verify compatibility

**Files:**
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: None
- Produces: hermes-agent package installed in venv

- [ ] **Step 1: Add hermes-agent to requirements.txt**

```bash
echo "hermes-agent>=0.19.0" >> /home/tuanlinh/trading/requirements.txt
```

- [ ] **Step 2: Install hermes-agent in venv**

```bash
/home/tuanlinh/trading/.venv/bin/pip install hermes-agent
```

- [ ] **Step 3: Verify import works**

```bash
/home/tuanlinh/trading/.venv/bin/python -c "from run_agent import AIAgent; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Verify trading tools can be imported**

```bash
/home/tuanlinh/trading/.venv/bin/python -c "from model_tools import get_tool_definitions; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Run existing tests to check compatibility**

```bash
/home/tuanlinh/trading/.venv/bin/python -m pytest tests/integration/test_e2e_paper_trading.py -v --tb=short 2>&1 | tail -20
```

Expected: All 10 tests pass (hermes-agent should not break existing code)

- [ ] **Step 6: Commit**

```bash
cd /home/tuanlinh/trading && git add requirements.txt && git commit -m "chore: add hermes-agent dependency"
```

---

### Task 2: Create SOUL.md and AGENTS.md

**Files:**
- Create: `/home/tuanlinh/.hermes/SOUL.md` (global Hermes persona)
- Create: `AGENTS.md` (project context)

**Interfaces:**
- Consumes: None
- Produces: Hermes persona + project context files

- [ ] **Step 1: Create SOUL.md**

```bash
mkdir -p /home/tuanlinh/.hermes
```

```markdown
# TRADING AGENT CONSTITUTION v1.0

## IDENTITY
You are an aggressive quantitative futures trader on Binance.
Capital varies per session — check portfolio state before trading.
Max Drawdown: 25% | Risk/Trade: 5-10%

## HARD RULES (VIOLATION = IMMEDIATE STOP)
1. NEVER trade without stop-loss
2. NEVER exceed 10% equity risk per trade
3. NEVER hold > 3 positions simultaneously
4. NEVER revenge trade (wait 1 hour after stop-loss)
5. ALWAYS verify risk/reward >= 2:1 before entry
6. ALWAYS log reasoning for audit

## DECISION FRAMEWORK
### Step 1: Regime Detection (5m/15m/1h/4h)
- Trend: EMA alignment, ADX > 25
- Range: BB squeeze, ADX < 20
- Volatile: ATR spike, funding extreme

### Step 2: Setup Identification
- Trend: Pullback to EMA, RSI reset, volume confirmation
- Range: Support/Resistance + divergence + volume
- Breakout: Volume surge, OI increase, funding flip

### Step 3: Risk Calculation
- Entry: Limit at value area
- Stop: Structure invalidation (swing high/low)
- Target: 2R minimum, trail at 1.5R

### Step 4: Portfolio Check
- Correlation with existing positions < 0.7
- Daily PnL > -5%
- Margin usage < 50%

## OUTPUT
Always think step-by-step. Use tools to gather data before deciding.
Output JSON only when calling place_order tool.
```

- [ ] **Step 2: Create AGENTS.md**

```markdown
# Trading Agent Project

## Overview
Autonomous futures trading agent using DeepSeek + Hermes Agent.
Runs on Binance testnet with paper trading engine.

## Key Commands
- Run agent: `python -m src.agent.main`
- Run tests: `pytest tests/ -v`
- Run e2e tests: `pytest tests/integration/test_e2e_paper_trading.py -v`

## Risk Rules
- Max risk per trade: 10% equity
- Max daily loss: 5% equity
- Max concurrent positions: 3
- Min risk/reward: 2.0
- Mandatory stop-loss

## Tech Stack
- LLM: DeepSeek-V4.1-Flash (via Hermes Agent)
- Agent: Hermes Agent 0.19.0
- Exchange: Binance Futures (testnet)
- Memory: Hermes Session DB + SQLite + ChromaDB
- Risk: Deterministic RiskGuard (outside LLM)
```

- [ ] **Step 3: Verify SOUL.md is loadable**

```bash
/home/tuanlinh/trading/.venv/bin/python -c "
from hermes_cli.config import cfg_get
from pathlib import Path
soul_path = Path.home() / '.hermes' / 'SOUL.md'
print(f'SOUL.md exists: {soul_path.exists()}')
print(f'AGENTS.md exists: {Path(\"AGENTS.md\").exists()}')
"
```

Expected: Both True

- [ ] **Step 4: Commit**

```bash
cd /home/tuanlinh/trading && git add AGENTS.md && git commit -m "docs: add Hermes SOUL.md and AGENTS.md for trading persona"
```

---

### Task 3: Register trading tools

**Files:**
- Create: `src/tools/hermes_tools.py`
- Create: `tests/unit/test_hermes_tools.py`

**Interfaces:**
- Consumes: `src.tools.binance_client.BinanceClient`, `src.risk.guard.RiskGuard`, `src.executor.paper_trading.PaperTradingEngine`
- Produces: Registered tools available to Hermes agent via `get_tool_definitions(enabled_toolsets=["trading"])`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_hermes_tools.py
import pytest
from unittest.mock import MagicMock, patch

def test_trading_toolset_registered():
    """Verify trading tools are registered in Hermes registry."""
    from tools.registry import registry
    import src.tools.hermes_tools  # Import to trigger registration
    
    tool_names = list(registry._tools.keys())
    assert "get_market_data" in tool_names
    assert "place_order" in tool_names
    assert "get_portfolio_state" in tool_names
    assert "calculate_technicals" in tool_names

def test_get_market_data_returns_dict():
    """Verify get_market_data tool returns valid dict."""
    from src.tools.hermes_tools import _get_market_data
    with patch("src.tools.hermes_tools._binance_client") as mock_client:
        mock_client.get_klines = MagicMock(return_value=[
            [1700000000000, "100", "101", "99", "100.5", "1000",
             1700000060000, "100000", 500, "500", "50000", "0"]
        ])
        mock_client.get_ticker = MagicMock(return_value={"bidPrice": "100.3", "askPrice": "100.7"})
        mock_client.get_orderbook = MagicMock(return_value={"bids": [["100.3", "10"]], "asks": [["100.7", "10"]]})
        mock_client.get_funding_rate = MagicMock(return_value=0.0001)
        
        result = _get_market_data("BTCUSDT")
        assert isinstance(result, dict)
        assert result["symbol"] == "BTCUSDT"

def test_place_order_validates_through_risk_guard():
    """Verify place_order validates through RiskGuard before execution."""
    from src.tools.hermes_tools import _place_order
    from src.risk.guard import RiskGuard, RiskDecision
    from src.tools.binance_client import PortfolioState
    
    guard = RiskGuard(PortfolioState(equity=10000, positions=[], daily_pnl=0))
    
    # Order without stop-loss should be rejected
    result = _place_order(
        symbol="BTCUSDT", side="BUY", qty=0.01,
        price=100000, stop_loss=None, take_profit=102000,
        _risk_guard=guard, _paper_engine=MagicMock()
    )
    assert result["status"] == "rejected"
    assert "Stop-loss" in result["reason"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/home/tuanlinh/trading/.venv/bin/python -m pytest tests/unit/test_hermes_tools.py -v 2>&1 | tail -15
```

Expected: FAIL (module not found)

- [ ] **Step 3: Write minimal implementation**

```python
# src/tools/hermes_tools.py
"""Hermes Agent trading tools — registered via tools.registry.register()."""

import logging
from typing import Optional

from tools.registry import register

logger = logging.getLogger(__name__)

# Module-level singletons (set by init_trading_tools)
_binance_client = None
_risk_guard = None
_paper_engine = None


def init_trading_tools(binance_client, risk_guard, paper_engine):
    """Initialize trading tool dependencies. Call once at startup."""
    global _binance_client, _risk_guard, _paper_engine
    _binance_client = binance_client
    _risk_guard = risk_guard
    _paper_engine = paper_engine


@register(
    name="get_market_data",
    description="Fetch OHLCV + orderbook for a symbol from Binance",
    toolset="trading",
)
def _get_market_data(symbol: str, timeframe: str = "1h") -> dict:
    """Fetch market data: klines, ticker, orderbook, funding rate."""
    if _binance_client is None:
        return {"error": "Binance client not initialized"}
    
    try:
        klines = _binance_client.get_klines(symbol, timeframe, limit=100)
        ticker = _binance_client.get_ticker(symbol)
        orderbook = _binance_client.get_orderbook(symbol, limit=5)
        funding = _binance_client.get_funding_rate(symbol)
        
        latest = klines[-1]
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "open": float(latest[1]),
            "high": float(latest[2]),
            "low": float(latest[3]),
            "close": float(latest[4]),
            "volume": float(latest[5]),
            "bid": float(ticker["bidPrice"]),
            "ask": float(ticker["askPrice"]),
            "bid_vol": float(orderbook["bids"][0][1]),
            "ask_vol": float(orderbook["asks"][0][1]),
            "funding_rate": funding,
        }
    except Exception as e:
        logger.error(f"get_market_data error: {e}")
        return {"error": str(e)}


@register(
    name="calculate_technicals",
    description="Calculate technical indicators (RSI, MACD, BB, ADX, regime)",
    toolset="trading",
)
def _calculate_technicals(symbol: str, timeframe: str = "1h") -> dict:
    """Compute technical indicators for symbol."""
    if _binance_client is None:
        return {"error": "Binance client not initialized"}
    
    try:
        from src.tools.technical_analysis import calculate_technicals
        import asyncio
        indicators = asyncio.get_event_loop().run_until_complete(
            calculate_technicals(symbol, timeframe)
        )
        return indicators.dict()
    except Exception as e:
        logger.error(f"calculate_technicals error: {e}")
        return {"error": str(e)}


@register(
    name="get_portfolio_state",
    description="Get current portfolio: positions, balance, PnL",
    toolset="trading",
)
def _get_portfolio_state() -> dict:
    """Get current portfolio state from paper engine."""
    if _paper_engine is None:
        return {"error": "Paper engine not initialized"}
    
    try:
        state = _paper_engine.get_portfolio_state()
        return state.dict() if hasattr(state, 'dict') else state
    except Exception as e:
        logger.error(f"get_portfolio_state error: {e}")
        return {"error": str(e)}


@register(
    name="place_order",
    description="Place a futures order with stop-loss and take-profit. Validated by Risk Guard.",
    toolset="trading",
)
def _place_order(
    symbol: str,
    side: str,
    qty: float,
    price: float,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    _risk_guard=None,
    _paper_engine=None,
) -> dict:
    """Place order through Risk Guard validation, then execute on paper engine."""
    guard = _risk_guard or _risk_guard_module
    engine = _paper_engine or _paper_engine_module
    
    if guard is None or engine is None:
        return {"error": "Risk guard or paper engine not initialized"}
    
    from src.tools.binance_client import OrderRequest
    
    order = OrderRequest(
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )
    
    # Risk Guard validation
    result = guard.validate(order)
    
    if result.decision.value == "reject":
        return {
            "status": "rejected",
            "reason": result.reason,
            "order": {"symbol": symbol, "side": side, "qty": qty},
        }
    
    # Execute (modified or original)
    target_order = result.modified_order or order
    
    try:
        import asyncio
        fill = asyncio.get_event_loop().run_until_complete(
            engine.execute_order(target_order)
        )
        return {
            "status": "filled",
            "order_id": fill.order_id,
            "symbol": fill.symbol,
            "side": fill.side,
            "qty": fill.qty,
            "price": fill.price,
            "fee": fill.fee,
        }
    except Exception as e:
        logger.error(f"place_order execution error: {e}")
        return {"error": str(e)}


@register(
    name="get_funding_rate",
    description="Get current and predicted funding rate for a symbol",
    toolset="trading",
)
def _get_funding_rate(symbol: str) -> dict:
    """Fetch funding rate from Binance."""
    if _binance_client is None:
        return {"error": "Binance client not initialized"}
    
    try:
        rate = _binance_client.get_funding_rate(symbol)
        return {"symbol": symbol, "funding_rate": rate}
    except Exception as e:
        return {"error": str(e)}


@register(
    name="get_recent_trades",
    description="Get recent trades from episodic memory",
    toolset="trading",
)
def _get_recent_trades(limit: int = 10) -> dict:
    """Retrieve recent trades from episodic memory."""
    try:
        from src.memory.store import AgentMemory
        memory = AgentMemory()
        trades = memory.get_recent_trades(limit=limit)
        return {"trades": trades, "count": len(trades)}
    except Exception as e:
        return {"error": str(e)}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
/home/tuanlinh/trading/.venv/bin/python -m pytest tests/unit/test_hermes_tools.py -v 2>&1 | tail -15
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/tuanlinh/trading && git add src/tools/hermes_tools.py tests/unit/test_hermes_tools.py && git commit -m "feat: register trading tools for Hermes Agent"
```

---

### Task 4: Create Hermes agent wrapper

**Files:**
- Create: `src/agent/hermes_loop.py`
- Create: `tests/unit/test_hermes_loop.py`

**Interfaces:**
- Consumes: `run_agent.AIAgent`, `src.tools.hermes_tools.init_trading_tools`
- Produces: `create_trading_agent()`, `run_trading_cycle(agent, portfolio, market_data)`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_hermes_loop.py
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

def test_create_trading_agent_returns_ai_agent():
    """Verify create_trading_agent returns an AIAgent instance."""
    from src.agent.hermes_loop import create_trading_agent
    
    with patch("src.agent.hermes_loop.AIAgent") as MockAgent:
        MockAgent.return_value = MagicMock()
        agent = create_trading_agent(
            deepseek_api_key="test-key",
            deepseek_model="deepseek-v4.1-flash",
            equity=10000,
        )
        assert agent is not None
        MockAgent.assert_called_once()

def test_create_trading_agent_configures_deepseek():
    """Verify DeepSeek config is passed correctly."""
    from src.agent.hermes_loop import create_trading_agent
    
    with patch("src.agent.hermes_loop.AIAgent") as MockAgent:
        create_trading_agent(
            deepseek_api_key="sk-test",
            deepseek_model="deepseek-v4.1-flash",
            equity=10000,
        )
        call_kwargs = MockAgent.call_args[1]
        assert call_kwargs["base_url"] == "https://api.deepseek.com/v1"
        assert call_kwargs["api_key"] == "sk-test"
        assert call_kwargs["model"] == "deepseek-v4.1-flash"
        assert call_kwargs["provider"] == "openai"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/home/tuanlinh/trading/.venv/bin/python -m pytest tests/unit/test_hermes_loop.py -v 2>&1 | tail -15
```

Expected: FAIL (module not found)

- [ ] **Step 3: Write minimal implementation**

```python
# src/agent/hermes_loop.py
"""Hermes Agent wrapper for trading — replaces LangGraph agent loop."""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def build_trading_prompt(equity: float = 10000) -> str:
    """Build the ephemeral system prompt with current equity."""
    return f"""# TRADING AGENT — Current Session

Capital: ${equity:,.2f} USDT

## YOUR TASK
Analyze the market, check portfolio, and decide on trades.
Use tools to gather data before making decisions.

## WORKFLOW
1. Call get_portfolio_state() to see current positions
2. Call get_market_data() for each symbol you want to analyze
3. Call calculate_technicals() for technical indicators
4. Decide: place_order() or wait

## RULES
- NEVER trade without stop-loss
- NEVER exceed 10% equity risk per trade
- NEVER hold > 3 positions simultaneously
- ALWAYS verify risk/reward >= 2:1 before entry
- Output your reasoning step-by-step
"""


def create_trading_agent(
    deepseek_api_key: str,
    deepseek_model: str = "deepseek-v4.1-flash",
    equity: float = 10000,
    max_iterations: int = 10,
):
    """Create a Hermes Agent configured for trading with DeepSeek backend.
    
    Args:
        deepseek_api_key: DeepSeek API key
        deepseek_model: Model name (default: deepseek-v4.1-flash)
        equity: Current portfolio equity for prompt
        max_iterations: Max tool-calling iterations per cycle
    
    Returns:
        Configured AIAgent instance
    """
    from run_agent import AIAgent
    
    agent = AIAgent(
        base_url="https://api.deepseek.com/v1",
        api_key=deepseek_api_key,
        model=deepseek_model,
        provider="openai",
        max_iterations=max_iterations,
        enabled_toolsets=["trading"],
        ephemeral_system_prompt=build_trading_prompt(equity),
        skip_memory=False,
        tool_progress_mode="all",
        save_trajectories=True,
        verbose_logging=False,
    )
    
    logger.info(f"Hermes trading agent created: model={deepseek_model}, max_iterations={max_iterations}")
    return agent


def run_trading_cycle(agent, market_data: dict = None) -> dict:
    """Run one trading cycle through the Hermes agent.
    
    Args:
        agent: AIAgent instance from create_trading_agent()
        market_data: Optional market data dict to include in prompt
    
    Returns:
        Dict with 'response' (final text) and 'tool_calls' (list)
    """
    prompt = "Analyze the market and decide on trades."
    
    if market_data:
        prompt += f"\n\nCurrent Market Data:\n{market_data}"
    
    try:
        result = agent.run_conversation(prompt)
        return {
            "response": result.get("final_response", ""),
            "tool_calls": result.get("tool_calls", []),
            "messages": result.get("messages", []),
        }
    except Exception as e:
        logger.error(f"Trading cycle error: {e}")
        return {"error": str(e)}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
/home/tuanlinh/trading/.venv/bin/python -m pytest tests/unit/test_hermes_loop.py -v 2>&1 | tail -15
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/tuanlinh/trading && git add src/agent/hermes_loop.py tests/unit/test_hermes_loop.py && git commit -m "feat: add Hermes Agent trading loop wrapper"
```

---

### Task 5: Create Hermes memory bridge

**Files:**
- Create: `src/memory/hermes_memory.py`
- Create: `tests/unit/test_hermes_memory.py`

**Interfaces:**
- Consumes: `src.memory.store.AgentMemory` (existing)
- Produces: `HermesTradingMemory` class bridging existing memory to Hermes-compatible format

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_hermes_memory.py
import pytest

def test_hermes_memory_wraps_episodic():
    """Verify HermesTradingMemory wraps existing episodic memory."""
    from src.memory.hermes_memory import HermesTradingMemory
    
    mem = HermesTradingMemory(db_path="/tmp/test_hermes_mem.db")
    assert mem.episodic is not None

def test_hermes_memory_add_trade():
    """Verify add_trade logs to episodic memory."""
    from src.memory.hermes_memory import HermesTradingMemory
    
    mem = HermesTradingMemory(db_path="/tmp/test_hermes_mem2.db")
    mem.add_trade({
        "symbol": "BTCUSDT",
        "decision": "BUY",
        "pnl": 100,
        "reasoning": "Trend break",
    })
    trades = mem.get_recent_trades(limit=5)
    assert len(trades) >= 1
    assert trades[0]["symbol"] == "BTCUSDT"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/home/tuanlinh/trading/.venv/bin/python -m pytest tests/unit/test_hermes_memory.py -v 2>&1 | tail -15
```

Expected: FAIL (module not found)

- [ ] **Step 3: Write minimal implementation**

```python
# src/memory/hermes_memory.py
"""Bridge between Hermes Agent and existing memory stores."""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class HermesTradingMemory:
    """Wraps existing episodic/semantic memory for Hermes Agent integration."""
    
    def __init__(self, db_path: str = "data/trade_journal.db"):
        from src.memory.store import AgentMemory
        self.episodic = AgentMemory(db_path=db_path)
    
    def add_trade(self, trade: Dict) -> None:
        """Log a trade decision to episodic memory."""
        self.episodic.log_trade(trade)
    
    def get_recent_trades(self, limit: int = 20) -> List[Dict]:
        """Get recent trades from episodic memory."""
        return self.episodic.get_recent_trades(limit=limit)
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics from trade journal."""
        return self.episodic.get_performance_stats()
    
    def get_context(self) -> str:
        """Get memory context string for Hermes prompt injection."""
        trades = self.get_recent_trades(limit=5)
        if not trades:
            return "No recent trades."
        
        lines = ["Recent trading history:"]
        for t in trades:
            lines.append(f"  {t['symbol']}: {t['decision']} | PnL: {t.get('pnl', 'N/A')} | {t.get('reasoning', '')}")
        return "\n".join(lines)
    
    def log_error(self, error: str) -> None:
        """Log an error to the memory store."""
        self.episodic.log_error(error)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
/home/tuanlinh/trading/.venv/bin/python -m pytest tests/unit/test_hermes_memory.py -v 2>&1 | tail -15
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/tuanlinh/trading && git add src/memory/hermes_memory.py tests/unit/test_hermes_memory.py && git commit -m "feat: add Hermes memory bridge for trading"
```

---

### Task 6: Update main.py to use Hermes agent

**Files:**
- Modify: `src/agent/main.py`

**Interfaces:**
- Consumes: `src.agent.hermes_loop.create_trading_agent`, `src.tools.hermes_tools.init_trading_tools`
- Produces: Updated main loop using Hermes agent

- [ ] **Step 1: Read current main.py**

Read `/home/tuanlinh/trading/src/agent/main.py` to understand current structure.

- [ ] **Step 2: Modify main.py**

Replace LangGraph imports and agent loop with Hermes:

```python
# src/agent/main.py
import asyncio
import logging
import os
import sys

from src.tools.binance_client import BinanceClient, PortfolioState as BSPortfolioState
from src.tools.market_data import get_market_data, get_portfolio_state as get_market_portfolio
from src.tools.technical_analysis import calculate_technicals, binance_client as ta_binance_client
from src.tools.hermes_tools import init_trading_tools
from src.agent.hermes_loop import create_trading_agent, run_trading_cycle
from src.risk.guard import RiskGuard
from src.memory.hermes_memory import HermesTradingMemory
from src.executor.paper_trading import PaperTradingEngine
from src.utils.logging_config import setup_logging
from src.utils.health_check import HealthCheck

setup_logging()
logger = logging.getLogger(__name__)


async def main():
    # Configuration
    deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4.1-flash")
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
    binance_testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
    loop_interval = int(os.getenv("AGENT_LOOP_INTERVAL", "60"))
    max_iterations = int(os.getenv("MAX_ITERATIONS_PER_LOOP", "10"))

    # Initialize Binance client
    binance_client = BinanceClient(
        api_key=os.getenv("BINANCE_API_KEY", ""),
        api_secret=os.getenv("BINANCE_API_SECRET", ""),
        testnet=binance_testnet,
    )
    await binance_client.connect()

    # Initialize memory
    memory = HermesTradingMemory(
        db_path=os.getenv("DATABASE_URL", "data/trade_journal.db"),
    )

    # Initialize portfolio and risk guard
    portfolio_state = BSPortfolioState(equity=10000, positions=[], daily_pnl=0, available_margin=5000, max_drawdown_today=0)
    risk_guard = RiskGuard(portfolio_state)

    # Initialize paper trading engine
    paper_engine = PaperTradingEngine(initial_equity=portfolio_state.equity)

    # Initialize trading tools (registers with Hermes registry)
    init_trading_tools(binance_client, risk_guard, paper_engine)

    # Create Hermes agent
    agent = create_trading_agent(
        deepseek_api_key=deepseek_api_key,
        deepseek_model=deepseek_model,
        equity=portfolio_state.equity,
        max_iterations=max_iterations,
    )

    # Health check
    health = HealthCheck()

    logger.info(f"Hermes Trading Agent started: {deepseek_model}")
    logger.info(f"Binance testnet: {binance_testnet}")
    logger.info(f"Loop interval: {loop_interval}s, Max iterations: {max_iterations}")

    # Main loop
    while True:
        try:
            # Get current portfolio
            portfolio = paper_engine.get_portfolio_state()
            risk_guard.portfolio = portfolio

            # Get market data for prompt
            market_data = {}
            symbols = ["BTCUSDT", "ETHUSDT"]
            for symbol in symbols:
                try:
                    snapshot = await get_market_data(symbol)
                    market_data[symbol] = snapshot.dict()
                    indicators = await calculate_technicals(symbol)
                    market_data[symbol]["technicals"] = indicators.dict()
                except Exception as e:
                    logger.error(f"Error fetching {symbol}: {e}")

            # Run one Hermes trading cycle
            result = run_trading_cycle(agent, market_data)
            
            if "error" in result:
                logger.error(f"Trading cycle error: {result['error']}")
                memory.log_error(result["error"])
            
            # Update portfolio
            portfolio = paper_engine.get_portfolio_state()
            health.check()

            await asyncio.sleep(loop_interval)

        except KeyboardInterrupt:
            logger.info("Shutting down trading agent...")
            break
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            memory.log_error(str(e))
            await asyncio.sleep(30)

    await binance_client.close()
    logger.info("Trading agent stopped.")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 3: Run existing tests to verify nothing breaks**

```bash
/home/tuanlinh/trading/.venv/bin/python -m pytest tests/integration/test_e2e_paper_trading.py -v --tb=short 2>&1 | tail -20
```

Expected: All 10 tests pass

- [ ] **Step 4: Commit**

```bash
cd /home/tuanlinh/trading && git add src/agent/main.py && git commit -m "feat: replace LangGraph with Hermes Agent in main loop"
```

---

### Task 7: Create E2E test with Hermes agent

**Files:**
- Create: `tests/integration/test_hermes_e2e.py`

**Interfaces:**
- Consumes: `src.agent.hermes_loop.create_trading_agent`, `src.tools.hermes_tools.init_trading_tools`
- Produces: E2E test verifying full Hermes agent trading cycle

- [ ] **Step 1: Write the test**

```python
# tests/integration/test_hermes_e2e.py
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

@pytest.mark.asyncio
async def test_hermes_agent_full_trading_cycle():
    """Run a full Hermes agent trading cycle with mocked LLM and Binance."""
    from src.agent.hermes_loop import create_trading_agent
    from src.tools.hermes_tools import init_trading_tools
    from src.risk.guard import RiskGuard
    from src.executor.paper_trading import PaperTradingEngine
    from src.tools.binance_client import PortfolioState as BSPortfolioState
    
    # Setup
    engine = PaperTradingEngine(initial_equity=10000)
    guard = RiskGuard(BSPortfolioState(equity=10000, positions=[], daily_pnl=0))
    
    # Mock Binance client
    mock_client = MagicMock()
    mock_client.get_klines.return_value = [
        [1700000000000, "100", "101", "99", "100.5", "1000",
         1700000060000, "100000", 500, "500", "50000", "0"]
    ]
    mock_client.get_ticker.return_value = {"bidPrice": "100.3", "askPrice": "100.7"}
    mock_client.get_orderbook.return_value = {"bids": [["100.3", "10"]], "asks": [["100.7", "10"]]}
    mock_client.get_funding_rate.return_value = 0.0001
    
    # Initialize tools
    init_trading_tools(mock_client, guard, engine)
    
    # Create agent
    agent = create_trading_agent(
        deepseek_api_key="test-key",
        deepseek_model="deepseek-v4.1-flash",
        equity=10000,
        max_iterations=3,
    )
    
    # Mock the agent's run_conversation to avoid real API call
    with patch.object(agent, "run_conversation") as mock_run:
        mock_run.return_value = {
            "final_response": "Market stable, no action needed.",
            "tool_calls": [],
            "messages": [],
        }
        
        from src.agent.hermes_loop import run_trading_cycle
        result = run_trading_cycle(agent, {"BTCUSDT": {"close": 100000}})
        
        assert "response" in result
        assert result["response"] == "Market stable, no action needed."


@pytest.mark.asyncio
async def test_hermes_agent_with_order_placement():
    """Test Hermes agent placing an order through tools."""
    from src.tools.hermes_tools import _place_order
    from src.risk.guard import RiskGuard
    from src.executor.paper_trading import PaperTradingEngine
    from src.tools.binance_client import PortfolioState as BSPortfolioState
    
    engine = PaperTradingEngine(initial_equity=10000)
    guard = RiskGuard(BSPortfolioState(equity=10000, positions=[], daily_pnl=0))
    
    # Place valid order
    result = _place_order(
        symbol="BTCUSDT", side="BUY", qty=0.01,
        price=100000, stop_loss=99000, take_profit=102000,
        _risk_guard=guard, _paper_engine=engine,
    )
    
    assert result["status"] == "filled"
    assert result["symbol"] == "BTCUSDT"
    assert result["side"] == "BUY"
```

- [ ] **Step 2: Run test**

```bash
/home/tuanlinh/trading/.venv/bin/python -m pytest tests/integration/test_hermes_e2e.py -v 2>&1 | tail -15
```

Expected: PASS

- [ ] **Step 3: Run all e2e tests**

```bash
/home/tuanlinh/trading/.venv/bin/python -m pytest tests/integration/test_e2e_paper_trading.py tests/integration/test_hermes_e2e.py -v 2>&1 | tail -20
```

Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
cd /home/tuanlinh/trading && git add tests/integration/test_hermes_e2e.py && git commit -m "test: add Hermes Agent E2E integration tests"
```

---

### Task 8: Remove LangGraph files

**Files:**
- Delete: `src/agent/nodes.py`
- Delete: `src/agent/graph.py`
- Delete: `tests/unit/test_agent_graph.py` (if exists)

**Interfaces:**
- Consumes: None
- Produces: Clean codebase with LangGraph removed

- [ ] **Step 1: Verify no other files import from nodes.py or graph.py**

```bash
grep -r "from src.agent.nodes\|from src.agent.graph\|import nodes\|import graph" /home/tuanlinh/trading/src/ 2>/dev/null
```

Expected: Only `main.py` should reference them (already replaced)

- [ ] **Step 2: Remove LangGraph files**

```bash
rm /home/tuanlinh/trading/src/agent/nodes.py
rm /home/tuanlinh/trading/src/agent/graph.py
rm /home/tuanlinh/trading/tests/unit/test_agent_graph.py 2>/dev/null
```

- [ ] **Step 3: Remove langgraph from requirements.txt (optional)**

```bash
sed -i '/langgraph/d' /home/tuanlinh/trading/requirements.txt
sed -i '/langchain-openai/d' /home/tuanlinh/trading/requirements.txt
sed -i '/langchain-core/d' /home/tuanlinh/trading/requirements.txt
```

- [ ] **Step 4: Run all tests**

```bash
/home/tuanlinh/trading/.venv/bin/python -m pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
cd /home/tuanlinh/trading && git add -A && git commit -m "refactor: remove LangGraph agent files, fully replaced by Hermes Agent"
```

---

### Task 9: Update Docker configuration (if needed)

**Files:**
- Modify: `Dockerfile`
- Modify: `docker-compose.yml` (if needed)

**Interfaces:**
- Consumes: Updated requirements.txt
- Produces: Docker image with Hermes Agent

- [ ] **Step 1: Check if Dockerfile needs updating**

```bash
cat /home/tuanlinh/trading/Dockerfile
```

- [ ] **Step 2: Update Dockerfile if needed**

Ensure the Dockerfile installs hermes-agent and uses the correct Python version.

- [ ] **Step 3: Build and test Docker image**

```bash
cd /home/tuanlinh/trading && docker compose build
```

- [ ] **Step 4: Commit**

```bash
cd /home/tuanlinh/trading && git add Dockerfile docker-compose.yml && git commit -m "chore: update Docker config for Hermes Agent"
```

---

### Task 10: Final verification

**Files:**
- None (verification only)

**Interfaces:**
- Consumes: All previous tasks
- Produces: Confirmed working Hermes Agent trading system

- [ ] **Step 1: Run all unit tests**

```bash
/home/tuanlinh/trading/.venv/bin/python -m pytest tests/unit/ -v 2>&1 | tail -20
```

Expected: All unit tests pass

- [ ] **Step 2: Run all integration tests**

```bash
/home/tuanlinh/trading/.venv/bin/python -m pytest tests/integration/test_e2e_paper_trading.py tests/integration/test_hermes_e2e.py -v 2>&1 | tail -20
```

Expected: All integration tests pass

- [ ] **Step 3: Verify import chain works**

```bash
/home/tuanlinh/trading/.venv/bin/python -c "
from src.agent.hermes_loop import create_trading_agent, run_trading_cycle
from src.tools.hermes_tools import init_trading_tools, _get_market_data, _place_order
from src.memory.hermes_memory import HermesTradingMemory
print('All imports OK')
"
```

Expected: `All imports OK`

- [ ] **Step 4: Verify no LangGraph references remain**

```bash
grep -r "langgraph\|LangGraph" /home/tuanlinh/trading/src/ 2>/dev/null
```

Expected: Empty (no references)

- [ ] **Step 5: Final commit**

```bash
cd /home/tuanlinh/trading && git add -A && git commit -m "feat: complete Hermes Agent integration — LangGraph fully replaced"
```

---

**End of Implementation Plan**
