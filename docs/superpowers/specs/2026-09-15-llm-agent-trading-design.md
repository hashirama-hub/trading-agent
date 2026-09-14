# LLM Agent Trading System Design Document

**Date:** 2026-09-15
**Status:** Approved for Implementation
**Author:** AI Assistant
**Reviewed by:** User

---

## 1. Executive Summary

Build a fully autonomous futures trading agent using **DeepSeek-V4.1-Flash** API as the reasoning engine, **LangGraph** for the agent orchestration, running on **local machine** with **Docker Compose**. The agent operates in a **ReAct (Reasoning + Acting)** loop with structured tool calling, strict risk guards, and persistent memory.

**Core Philosophy:** LLM proposes → Risk Guard disposes. The LLM never executes trades directly; all orders pass through deterministic validation.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
                        LLM AGENT TRADING SYSTEM
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    AGENT CORE (LangGraph StateGraph)                │   │
│  │                                                                     │   │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌────────────┐   │   │
│  │  │ Observe  │───▶│ Reason   │───▶│   Act    │───▶│  Reflect   │   │   │
│  │  │  Node    │    │  Node    │    │  Node    │    │  Node      │   │   │
│  │  └──────────┘    └──────────┘    └──────────┘    └────────────┘   │   │
│  │       │              │              │              │                │   │
│  │       ▼              ▼              ▼              ▼                │   │
│  │  ┌──────────────────────────────────────────────────────────────┐  │   │
│  │  │                    STATE (TypedDict)                         │  │   │
│  │  │  - market_data: Dict[str, MarketSnapshot]                   │  │   │
│  │  │  - portfolio: PortfolioState                                 │  │   │
│  │  │  - memory: AgentMemory                                       │  │   │
│  │  │  - current_plan: TradingPlan                                 │  │   │
│  │  │  - iteration: int                                            │  │   │
│  │  │  - last_decision: Decision                                   │  │   │
│  │  └──────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│           │                    │                    │                      │
│           ▼                    ▼                    ▼                      │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────────────┐  │
│  │   DATA TOOLS     │ │  EXECUTION TOOLS │ │      MEMORY STORE        │  │
│  │  (Binance WS/REST)│ │  (Binance API)   │ │  (ChromaDB + SQLite)     │  │
│  │  - get_klines    │ │  - place_order   │ │  - Episodic: trade log   │  │
│  │  - get_orderbook │ │  - cancel_order  │ │  - Semantic: strategies  │  │
│  │  - get_funding   │ │  - modify_order  │ │  - Working: context      │  │
│  │  - get_news      │ │  - get_position  │ └──────────────────────────┘  │
│  │  - calc_ta       │ │  - get_balance   │                                │
│  └──────────────────┘ └──────────────────┘                                │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    RISK GUARD (Deterministic, Outside LLM)          │   │
│  │  - Max position size: equity × 10%                                  │   │
│  │  - Mandatory stop-loss                                              │   │
│  │  - Max 3 concurrent positions                                       │   │
│  │  - Daily loss limit: -5% equity                                     │   │
│  │  - Correlation check                                                │   │
│  │  - Kill switch                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Config → Agent Init → [Loop: Observe → Reason → Act → Reflect] → Persist → Dashboard
                                    │
                                    ▼
                            Risk Guard (every order)
                                    │
                                    ▼
                              Binance API
```

---

## 3. Component Specifications

### 3.1 Agent Core (LangGraph)

**State Schema** (`src/agent/state.py`):
```python
class AgentState(TypedDict):
    # Market data
    market_data: Dict[str, MarketSnapshot]  # symbol → snapshot
    
    # Portfolio state
    portfolio: PortfolioState
    
    # Memory
    memory: AgentMemory
    
    # Current trading plan
    current_plan: Optional[TradingPlan]
    
    # Loop control
    iteration: int
    max_iterations: int
    
    # Last decision for reflection
    last_decision: Optional[Decision]
    
    # Error tracking
    errors: List[str]
```

**Nodes:**
1. **ObserveNode** - Fetch market data, update state
2. **ReasonNode** - LLM reasoning with tools
3. **ActNode** - Execute approved actions
4. **ReflectNode** - Log, update memory, evaluate performance

**Edges:** Observe → Reason → Act → Reflect → (loop or END)

### 3.2 Tools (Function Calling)

All tools return structured Pydantic models, never raw strings.

| Tool | Description | Parameters | Returns |
|------|-------------|------------|---------|
| `get_market_data` | OHLCV + orderbook | symbol, timeframe, limit | MarketSnapshot |
| `calculate_technicals` | RSI, MACD, BB, VPIN, Volume Profile | symbol, indicators[] | TechnicalIndicators |
| `get_funding_rate` | Current + predicted funding | symbol | FundingData |
| `get_market_sentiment` | Fear/Greed, liquidations, OI | - | SentimentData |
| `get_news` | Recent news for symbol | symbol, hours | List[NewsItem] |
| `get_portfolio_state` | Positions, balance, PnL | - | PortfolioState |
| `place_order` | Submit order with risk params | OrderRequest | OrderResult |
| `cancel_order` | Cancel by client_order_id | client_order_id | CancelResult |
| `modify_order` | Modify stop/TP | client_order_id, new_sl, new_tp | ModifyResult |

### 3.3 Risk Guard (`src/risk/guard.py`)

**Hard Constraints (Non-negotiable):**
```python
MAX_RISK_PER_TRADE = 0.10      # 10% equity
MAX_DAILY_LOSS = 0.05          # 5% equity
MAX_CONCURRENT_POSITIONS = 3
MAX_LEVERAGE = 20
MIN_RISK_REWARD = 2.0
MANDATORY_STOP_LOSS = True
CORRELATION_THRESHOLD = 0.7
```

**Validation Pipeline:**
1. Position size check
2. Stop-loss presence
3. Concurrent positions
4. Daily loss limit
5. Correlation with existing positions
6. Leverage limit

### 3.4 Memory System

| Memory Type | Backend | TTL | Content |
|-------------|---------|-----|---------|
| Working | In-memory (LangGraph state) | Session | Current context, last 20 turns |
| Episodic | SQLite + JSONL | Forever | Every trade: decision, outcome, PnL |
| Semantic | ChromaDB (embeddings) | Forever | Strategy docs, regime patterns, lessons |
| Procedural | Few-shot in prompt | Static | Example decisions, edge cases |

### 3.5 Binance Integration

- **WebSocket:** Real-time klines, ticker, depth, trades, liquidations
- **REST:** Historical data, account info, order management
- **Testnet:** All development on testnet first
- **API Key Management:** Environment variables + Docker secrets

### 3.6 Dashboard (Real-time)

**Backend:** FastAPI + WebSocket
**Frontend:** Next.js + React + Tailwind + Recharts

**Features:**
- Live PnL, positions, equity curve
- Agent reasoning trace (thought → action → result)
- Trade history with filters
- Risk metrics (Sharpe, max DD, win rate)
- Manual override (pause, close all, kill switch)

---

## 4. Configuration

### 4.1 Environment Variables (`.env`)
```bash
# DeepSeek
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_MODEL=deepseek-v4.1-flash
DEEPSEEK_TEMPERATURE=0.1
DEEPSEEK_MAX_TOKENS=4096

# Binance
BINANCE_API_KEY=xxx
BINANCE_API_SECRET=xxx
BINANCE_TESTNET=true

# Risk
MAX_RISK_PER_TRADE=0.10
MAX_DAILY_LOSS=0.05
MAX_POSITIONS=3
MIN_RR=2.0

# Agent
AGENT_LOOP_INTERVAL=60  # seconds
MAX_ITERATIONS_PER_LOOP=10

# Database
DATABASE_URL=postgresql://user:pass@timescaledb:5432/trading
CHROMA_HOST=chromadb
CHROMA_PORT=8000
REDIS_URL=redis://redis:6379
```

### 4.2 Docker Compose Services

```yaml
services:
  redis:
    image: redis:7-alpine
  
  chromadb:
    image: chromadb/chroma:latest
    volumes: [chroma_data:/data]
  
  timescaledb:
    image: timescale/timescaledb:latest-pg16
    environment:
      POSTGRES_DB: trading
      POSTGRES_USER: trader
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes: [timeseries_data:/var/lib/postgresql/data]
  
  agent-core:
    build: ./services/agent-core
    depends_on: [redis, chromadb, timescaledb]
    env_file: .env
    volumes: [./logs:/app/logs]
  
  executor:
    build: ./services/executor
    depends_on: [redis, timescaledb]
    env_file: .env
  
  dashboard-api:
    build: ./services/dashboard-api
    depends_on: [timescaledb, redis]
    ports: ["8000:8000"]
    env_file: .env
  
  dashboard-ui:
    build: ./services/dashboard-ui
    ports: ["3000:3000"]
    depends_on: [dashboard-api]
```

---

## 5. Security

- **API Keys:** Never in code, only in `.env` (gitignored) + Docker secrets
- **Network:** Services communicate via Docker network, no external exposure except dashboard
- **Binance:** IP whitelist, withdrawal disabled, testnet first
- **Audit Log:** Every decision + action logged to immutable store

---

## 6. Testing Strategy

| Level | Tool | Coverage |
|-------|------|----------|
| Unit | pytest | Tools, Risk Guard, State transitions |
| Integration | pytest + testcontainers | Binance WS/REST, ChromaDB, Redis |
| Agent Simulation | Custom harness | Replay historical data through agent |
| Paper Trading | Live testnet | 60-90 days before live |

---

## 7. Deployment Phases

| Phase | Duration | Goal |
|-------|----------|------|
| **Phase 1: Foundation** | Week 1-2 | Infrastructure, Binance connector, Risk Guard, Paper trading harness |
| **Phase 2: Agent Core** | Week 2-3 | LangGraph setup, System prompt, Basic tools, Memory |
| **Phase 3: Intelligence** | Week 3-4 | Multi-timeframe analysis, Regime detection, News integration |
| **Phase 4: Optimization** | Week 4-5 | Prompt tuning, Few-shot examples, Ensemble voting |
| **Phase 5: Production** | Week 5-6 | Live deployment, Monitoring, Alerting, Auto-recovery |

---

## 8. Success Criteria

- **Paper Trading (60 days):** Sharpe > 1.5, Max DD < 15%, Win rate > 45%
- **Latency:** Agent loop < 10s, Order execution < 500ms
- **Uptime:** 99.9% during market hours
- **Recovery:** Auto-restart < 30s on crash

---

## 9. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| LLM hallucination → bad trade | Medium | High | Risk Guard, mandatory SL, paper trading |
| API rate limit / downtime | Medium | High | Local fallback model, caching, circuit breaker |
| Memory leak / context overflow | Low | Medium | Token counting, periodic summarization |
| Market regime change | High | High | Regime detection, adaptive prompts |
| Bug in execution logic | Low | Critical | Extensive unit tests, simulation harness |

---

## 10. Appendix: System Prompt Template

```markdown
# TRADING AGENT CONSTITUTION v1.0

## IDENTITY
You are an aggressive quantitative futures trader on Binance.
Capital: ${EQUITY} USDT | Max Drawdown: 25% | Risk/Trade: 5-10%

## HARD RULES (VIOLATION = IMMEDIATE STOP)
1. NEVER trade without stop-loss
2. NEVER exceed 10% equity risk per trade
3. NEVER hold > 3 positions simultaneously
4. NEVER revenge trade (wait 1 hour after stop-loss)
5. ALWAYS verify risk/reward ≥ 2:1 before entry
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

## TOOLS
{tools_schema}

## OUTPUT FORMAT (JSON ONLY)
{
  "thought": "Step-by-step reasoning...",
  "regime": "trend|range|volatile",
  "setup": "description",
  "action": "tool_name|null",
  "params": {...},
  "risk_check": {
    "position_size_pct": 0.08,
    "stop_loss": 123.45,
    "take_profit": 135.67,
    "risk_reward": 2.3
  },
  "confidence": 0.85
}
```

---

**End of Design Document**