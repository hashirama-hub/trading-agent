# Hermes Agent Integration Design Spec

**Date:** 2026-09-15
**Status:** Approved
**Author:** AI Assistant
**Reviewed by:** User

---

## 1. Executive Summary

Replace the LangGraph agent loop with **Hermes Agent** (Nous Research) as the main agent orchestrator. DeepSeek remains the LLM backend via Hermes's OpenAI-compatible provider. Trading tools are registered as a Hermes toolset. Risk Guard stays deterministic (outside LLM control). Memory transitions from LangGraph state to Hermes session DB + SOUL.md + AGENTS.md.

**Goal:** Leverage Hermes Agent's built-in learning loop, skills system, memory, and tool calling to create a self-improving trading agent.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HERMES AGENT (AIAgent)                    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  run_conversation() → tool calling loop → response   │   │
│  │                                                     │   │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────────┐  │   │
│  │  │ Observe  │───▶│ Reason   │───▶│  Act         │  │   │
│  │  │ (tools)  │    │ (LLM)   │    │  (tools)     │  │   │
│  │  └──────────┘    └──────────┘    └──────────────┘  │   │
│  │       │                              │             │   │
│  │       ▼                              ▼             │   │
│  │  ┌──────────────────────────────────────────────┐  │   │
│  │  │            Hermes Tool Registry               │  │   │
│  │  │  - get_market_data (trading toolset)          │  │   │
│  │  │  - place_order (trading toolset)              │  │   │
│  │  │  - get_portfolio_state (trading toolset)      │  │   │
│  │  │  - calculate_technicals (trading toolset)     │  │   │
│  │  │  - terminal, web_search (built-in)            │  │   │
│  │  └──────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Risk Guard (deterministic, in place_order tool)    │   │
│  │  - Validates every order before execution           │   │
│  │  - Max risk, stop-loss, DD, correlation             │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Paper Trading Engine (unchanged backend service)   │   │
│  │  - Executes validated orders                        │   │
│  │  - Tracks positions, PnL, fills                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Memory                                             │   │
│  │  - Hermes Session DB (conversation + tool traces)   │   │
│  │  - SOUL.md (persona, trading rules)                 │   │
│  │  - AGENTS.md (project context)                      │   │
│  │  - Learning Graph (self-improving patterns)         │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Main Loop:
  1. Build trading prompt (equity, portfolio, market state)
  2. agent.run_conversation(prompt)
     → Hermes LLM call → tool calls → Risk Guard → Paper Engine
     → Hermes learning loop → session saved
  3. Parse result for portfolio update
  4. Sleep (loop interval)
  5. Repeat
```

---

## 3. Component Specifications

### 3.1 Hermes Agent Configuration

**File:** `src/agent/hermes_loop.py`

```python
from run_agent import AIAgent

def create_trading_agent(
    deepseek_api_key: str,
    deepseek_model: str = "deepseek-v4.1-flash",
    equity: float = 10000,
    max_iterations: int = 10,
) -> AIAgent:
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
    )
    return agent
```

### 3.2 Trading Toolset

**File:** `src/tools/hermes_tools.py`

Uses Hermes `tools.registry.register()` pattern. Each tool self-registers with a schema and handler.

| Tool | Parameters | Returns | Risk Guard |
|------|-----------|---------|------------|
| `get_market_data` | symbol, timeframe | MarketSnapshot dict | No |
| `calculate_technicals` | symbol, timeframe | TechnicalIndicators dict | No |
| `get_portfolio_state` | - | PortfolioState dict | No |
| `place_order` | symbol, side, qty, price, stop_loss, take_profit | FillResult dict | **Yes** |
| `get_funding_rate` | symbol | FundingData dict | No |
| `get_recent_trades` | limit | List[Dict] | No |

**Risk Guard Integration:** `place_order` tool validates through `RiskGuard` before execution. If rejected, returns error message to LLM (not an exception). If modified, executes the modified order.

### 3.3 Memory Architecture

| Layer | Backend | Content |
|-------|---------|---------|
| Session | Hermes Session DB | Conversation history, tool call traces, working context |
| Persona | `.hermes/SOUL.md` | Trading rules, risk params, personality |
| Project | `AGENTS.md` | Project-specific instructions, trading context |
| Learning | Hermes Learning Graph | Self-improving patterns from trading experience |
| Episodic | SQLite (keep existing) | Trade journal: every decision + outcome |
| Semantic | ChromaDB (keep existing) | Strategy patterns, regime lessons |

**Bridge:** `src/memory/hermes_memory.py` wraps existing memory stores (episodic, semantic) as Hermes-compatible contexts.

### 3.4 Risk Guard (Unchanged)

- Deterministic validation
- Called inside `place_order` tool before execution
- Rejects/modifies orders that violate rules
- LLM sees rejection reason and adjusts

### 3.5 Paper Trading Engine (Unchanged)

- Executes validated orders
- Tracks positions, PnL, fills
- Returns `OrderResult` to Hermes tool
- No changes to engine logic

### 3.6 Entry Point

**File:** `src/agent/main.py` (modified)

```python
async def main():
    agent = create_trading_agent(...)
    
    while True:
        portfolio = paper_engine.get_portfolio_state()
        prompt = build_trading_prompt(portfolio)
        
        result = agent.run_conversation(prompt)
        # Parse result, update portfolio, sleep
        
        await asyncio.sleep(loop_interval)
```

---

## 4. Configuration

### 4.1 Environment Variables (unchanged)
```bash
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_MODEL=deepseek-v4.1-flash
BINANCE_API_KEY=xxx
BINANCE_API_SECRET=xxx
BINANCE_TESTNET=true
MAX_RISK_PER_TRADE=0.10
MAX_DAILY_LOSS=0.05
MAX_POSITIONS=3
MIN_RR=2.0
AGENT_LOOP_INTERVAL=60
MAX_ITERATIONS_PER_LOOP=10
DATABASE_URL=postgresql://user:pass@timescaledb:5432/trading
CHROMA_HOST=chromadb
CHROMA_PORT=8000
```

### 4.2 Hermes Config
```bash
# .hermes/config.yaml
default_provider: openai
providers:
  openai:
    base_url: https://api.deepseek.com/v1
    api_key: ${DEEPSEEK_API_KEY}
    model: deepseek-v4.1-flash
```

### 4.3 SOUL.md (Trading Persona)
```markdown
# TRADING AGENT

## IDENTITY
You are an aggressive quantitative futures trader on Binance.
Capital: $10,000 USDT | Max Drawdown: 25% | Risk/Trade: 5-10%

## HARD RULES
1. NEVER trade without stop-loss
2. NEVER exceed 10% equity risk per trade
3. NEVER hold > 3 positions simultaneously
4. ALWAYS verify risk/reward >= 2:1 before entry
5. ALWAYS log reasoning for audit

## DECISION FRAMEWORK
1. Regime Detection (trend/range/volatile)
2. Setup Identification
3. Risk Calculation
4. Portfolio Check
```

---

## 5. Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `src/tools/hermes_tools.py` | CREATE | Register trading tools via Hermes `register()` |
| `src/agent/hermes_loop.py` | CREATE | Hermes agent wrapper with trading config |
| `src/memory/hermes_memory.py` | CREATE | Bridge: Hermes session DB ↔ existing memory |
| `src/agent/main.py` | MODIFY | Replace LangGraph entry point with Hermes loop |
| `src/agent/prompts.py` | MODIFY | System prompt moves to SOUL.md / ephemeral |
| `.hermes/SOUL.md` | CREATE | Trading agent persona + hard rules |
| `AGENTS.md` | CREATE | Project context (trading rules, risk params) |
| `src/agent/nodes.py` | DELETE | Hermes handles the agent loop |
| `src/agent/graph.py` | DELETE | Hermes handles the agent loop |
| `src/agent/state.py` | KEEP | Still needed for state typing |

---

## 6. Testing Strategy

| Level | Tool | Coverage |
|-------|------|----------|
| Unit | pytest | Tool registration, Risk Guard, memory bridge |
| Integration | pytest | Hermes agent loop with mocked LLM |
| End-to-end | pytest | Full loop: observe → reason → act → reflect |

---

## 7. Migration Steps

1. Install `hermes-agent` in venv
2. Create `.hermes/SOUL.md` + `AGENTS.md`
3. Register trading tools in `hermes_tools.py`
4. Create `hermes_loop.py` wrapper
5. Create `hermes_memory.py` bridge
6. Update `main.py` to use Hermes agent
7. Remove LangGraph files (`nodes.py`, `graph.py`)
8. Run tests, verify all 10 e2e tests pass
9. Update Docker Compose if needed

---

**End of Design Spec**
