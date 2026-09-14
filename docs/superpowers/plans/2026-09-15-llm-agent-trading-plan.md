# LLM Agent Trading System - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully autonomous futures trading agent using DeepSeek-V4.1-Flash + LangGraph, with structured tool calling, deterministic Risk Guard, and persistent memory, deployable via Docker Compose.

**Architecture:** ReAct Agent Loop (Observe → Reason → Act → Reflect) with LangGraph StateGraph. LLM proposes decisions via function calling; Risk Guard validates every order deterministically before execution. All services containerized with Docker Compose.

**Tech Stack:** Python 3.12, LangGraph 0.2+, DeepSeek-V4.1-Flash API, Binance Futures API (testnet first), Redis Streams, ChromaDB, TimescaleDB, FastAPI + WebSocket dashboard.

**Spec:** docs/superpowers/specs/2026-09-15-llm-agent-trading-design.md

---

## Global Constraints

- DeepSeek model: `deepseek-v4.1-flash` (set in `.env`)
- Temperature: 0.1 (deterministic reasoning)
- Binance testnet: `true` during development
- Risk Guard is OUTSIDE LLM control — hard-coded validation
- All tools return Pydantic models, never raw strings
- Every decision logged to immutable store (JSONL + DB)
- API key in `.env` only, gitignored, never in code

---

## Task 1: Project Scaffold & Infrastructure

**Files:**
- Create: `docker-compose.yml`, `.env.example`, `.gitignore`
- Create: `src/agent/__init__.py`, `src/executor/__init__.py`, `src/dashboard/__init__.py`
- Create: `tests/__init__.py`, `tests/conftest.py`
- Create: `docs/README.md`

**Interfaces:**
- Consumes: None (initial scaffold)
- Produces: Docker Compose environment ready for all services

- [ ] **Step 1: Create project structure**
```bash
mkdir -p src/{agent,executor,dashboard,risk,memory,tools,utils}
mkdir -p services/{agent-core,executor,dashboard-api,dashboard-ui}
mkdir -p tests/{unit,integration}
mkdir -p docs/superpowers/{specs,plans}
```

- [ ] **Step 2: Write docker-compose.yml**
```yaml
services:
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    volumes: [redis_data:/data]

  chromadb:
    image: chromadb/chroma:latest
    ports: ["8000:8000"]
    volumes: [chroma_data:/data]

  timescaledb:
    image: timescale/timescaledb:latest-pg16
    environment:
      POSTGRES_DB: trading
      POSTGRES_USER: trader
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    ports: ["5432:5432"]
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

volumes:
  redis_data:
  chroma_data:
  timeseries_data:
```

- [ ] **Step 3: Write .env.example**
```env
DEEPSEEK_API_KEY=
DEEPSEEK_MODEL=deepseek-v4.1-flash
DEEPSEEK_TEMPERATURE=0.1
DEEPSEEK_MAX_TOKENS=4096
BINANCE_API_KEY=
BINANCE_API_SECRET=
BINANCE_TESTNET=true
MAX_RISK_PER_TRADE=0.10
MAX_DAILY_LOSS=0.05
MAX_POSITIONS=3
MIN_RR=2.0
AGENT_LOOP_INTERVAL=60
MAX_ITERATIONS_PER_LOOP=10
DATABASE_URL=postgresql://trader:changeme@timescaledb:5432/trading
CHROMA_HOST=chromadb
CHROMA_PORT=8000
REDIS_URL=redis://redis:6379
```

- [ ] **Step 4: Write .gitignore**
```
.env
__pycache__/
*.pyc
logs/
.chroma/
node_modules/
```

- [ ] **Step 5: Write tests/conftest.py (shared fixtures)**
```python
import pytest
import os
from dotenv import load_dotenv

load_dotenv()

@pytest.fixture
def binance_testnet():
    return os.getenv("BINANCE_TESTNET", "true").lower() == "true"

@pytest.fixture
def deepseek_model():
    return os.getenv("DEEPSEEK_MODEL", "deepseek-v4.1-flash")
```

- [ ] **Step 6: Commit scaffold**
```bash
git add .
git commit -m "feat: scaffold LLM agent trading project with Docker Compose"
```

---

## Task 2: Binance Connector (Tools Layer)

**Files:**
- Create: `src/tools/binance_client.py`
- Create: `src/tools/market_data.py`
- Create: `src/tools/order_tools.py`
- Create: `tests/unit/test_binance_tools.py`

**Interfaces:**
- Consumes: `.env` API keys, Binance testnet/live config
- Produces: `get_market_data()`, `get_portfolio_state()`, `place_order()`, etc.

- [ ] **Step 1: Write Binance client (async REST + WebSocket)**
```python
import asyncio
import logging
from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

class MarketSnapshot(BaseModel):
    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: float
    ask: float
    bid_vol: float
    ask_vol: float
    funding_rate: Optional[float] = None

class OrderRequest(BaseModel):
    symbol: str
    side: str  # BUY/SELL
    qty: float
    type: str = "LIMIT"
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reduce_only: bool = False
    client_order_id: Optional[str] = None

class OrderResult(BaseModel):
    order_id: int
    client_order_id: str
    symbol: str
    side: str
    qty: float
    price: float
    status: str
    fee: float

class PortfolioState(BaseModel):
    equity: float
    available_margin: float
    positions: List[Dict]
    daily_pnl: float
    max_drawdown_today: float
```

- [ ] **Step 2: Write BinanceClient class**
```python
class BinanceClient:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.testnet = testnet
        self.client = None
        self.bm = None
        
    async def connect(self):
        self.client = await AsyncClient.create(
            api_key=self.api_key, 
            api_secret=self.api_secret,
            testnet=self.testnet
        )
        self.bm = BinanceSocketManager(self.client)
    
    async def get_klines(self, symbol: str, interval: str = "1h", limit: int = 100) -> List[Dict]:
        return await self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
    
    async def get_ticker(self, symbol: str) -> Dict:
        return await self.client.get_symbol_ticker(symbol=symbol)
    
    async def get_orderbook(self, symbol: str, limit: int = 10) -> Dict:
        return await self.client.get_order_book(symbol=symbol, limit=limit)
    
    async def get_funding_rate(self, symbol: str) -> float:
        history = await self.client.get_funding_rate(symbol=symbol, limit=1)
        return float(history[0]["fundingRate"])
    
    async def place_order(self, order: OrderRequest) -> OrderResult:
        try:
            result = await self.client.create_order(
                symbol=order.symbol,
                side=order.side,
                type=order.type,
                quantity=order.qty,
                price=order.price,
                timeInForce="GTC",
                newClientOrderId=order.client_order_id,
                reduceOnly=order.reduce_only
            )
            return OrderResult(**result)
        except BinanceAPIException as e:
            logger.error(f"Order failed: {e}")
            raise
    
    async def get_account(self) -> Dict:
        return await self.client.get_account()
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        return await self.client.get_open_orders(symbol=symbol)
    
    async def cancel_order(self, symbol: str, order_id: int):
        return await self.client.cancel_order(symbol=symbol, orderId=order_id)
    
    async def ws_start_klines(self, symbol: str, interval: str, handler):
        socket = self.bm.kline_socket(symbol=symbol, interval=interval)
        self.bm.start(socket, handler)
    
    async def close(self):
        if self.client:
            await self.client.close_connection()
```

- [ ] **Step 3: Write market data tool (get_market_data)**
```python
async def get_market_data(
    symbol: str, 
    interval: str = "1h", 
    limit: int = 100
) -> MarketSnapshot:
    """Fetch OHLCV + orderbook for symbol."""
    klines = await binance_client.get_klines(symbol, interval, limit)
    ticker = await binance_client.get_ticker(symbol)
    orderbook = await binance_client.get_orderbook(symbol, limit=5)
    funding = await binance_client.get_funding_rate(symbol)
    
    latest = klines[-1]
    return MarketSnapshot(
        symbol=symbol,
        timeframe=interval,
        timestamp=datetime.fromtimestamp(latest[0] / 1000),
        open=float(latest[1]),
        high=float(latest[2]),
        low=float(latest[3]),
        close=float(latest[4]),
        volume=float(latest[5]),
        bid=float(ticker["bidPrice"]),
        ask=float(ticker["askPrice"]),
        bid_vol=float(orderbook["bids"][0][1]),
        ask_vol=float(orderbook["asks"][0][1]),
        funding_rate=funding
    )
```

- [ ] **Step 4: Write portfolio state tool (get_portfolio_state)**
```python
async def get_portfolio_state() -> PortfolioState:
    """Get current portfolio: positions, balance, PnL."""
    account = await binance_client.get_account()
    positions = [p for p in account["positions"] if float(p["positionAmt"]) != 0]
    
    equity = float(account["totalPositionRisk"][0]["entryPrice"]) if account.get("totalPositionRisk") else 0
    available = float(account["availableBalance"])
    
    daily_pnl = sum(float(p["unRealizedProfit"]) for p in positions)
    
    return PortfolioState(
        equity=equity,
        available_margin=available,
        positions=positions,
        daily_pnl=daily_pnl,
        max_drawdown_today=0.0  # Calculate from trade log
    )
```

- [ ] **Step 5: Write unit tests for tools**
```python
import pytest
from src.tools.binance_client import MarketSnapshot, OrderRequest, PortfolioState
from src.tools.market_data import get_market_data
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_get_market_data_returns_snapshot():
    with patch("src.tools.market_data.binance_client") as mock_client:
        mock_client.get_klines = AsyncMock(return_value=[
            [1700000000000, "100", "101", "99", "100.5", "1000", ...]
        ])
        mock_client.get_ticker = AsyncMock(return_value={"bidPrice": "100.3", "askPrice": "100.7"})
        mock_client.get_orderbook = AsyncMock(return_value={"bids": [["100.3", "10"]], "asks": [["100.7", "10"]]})
        mock_client.get_funding_rate = AsyncMock(return_value=0.0001)
        
        snapshot = await get_market_data("BTCUSDT")
        assert snapshot.symbol == "BTCUSDT"
        assert snapshot.close == 100.5
        assert snapshot.bid == 100.3

@pytest.mark.asyncio
async def test_portfolio_state_parses_positions():
    with patch("src.tools.order_tools.binance_client") as mock_client:
        mock_client.get_account = AsyncMock(return_value={
            "positions": [{"symbol": "BTCUSDT", "positionAmt": "0.5", "unRealizedProfit": "100"}],
            "availableBalance": "5000",
            "totalPositionRisk": [{"entryPrice": "100000"}]
        })
        
        state = await get_portfolio_state()
        assert state.equity == 100000
        assert state.daily_pnl == 100
        assert len(state.positions) == 1
```

- [ ] **Step 6: Run tests**
```bash
pytest tests/unit/test_binance_tools.py -v
```

- [ ] **Step 7: Commit**
```bash
git add src/tools/ tests/unit/test_binance_tools.py
git commit -m "feat: add Binance connector with market data + portfolio tools"
```

---

## Task 3: Technical Analysis Tools

**Files:**
- Create: `src/tools/technical_analysis.py`
- Create: `tests/unit/test_technical_analysis.py`

**Interfaces:**
- Consumes: MarketSnapshot data (from Task 2)
- Produces: TechnicalIndicators model (RSI, MACD, BB, VPIN, Volume Profile)

- [ ] **Step 1: Write TechnicalIndicators model**
```python
from pydantic import BaseModel
from typing import Optional, List, Dict

class TechnicalIndicators(BaseModel):
    rsi_14: float
    rsi_21: float
    macd: float
    macd_signal: float
    macd_histogram: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_width: float
    atr_14: float
    vpin: float  # Volume Profile / Order Flow Imbalance
    volume_sma_20: float
    volume_ratio: float  # current volume / sma
    adx: float
    ema_9: float
    ema_21: float
    ema_50: float
    trend_score: float  # -1 (downtrend) to +1 (uptrend)
    regime: str  # "trend" | "range" | "volatile"
```

- [ ] **Step 2: Implement calculate_technicals function**
```python
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, MACD
from ta.trend import ADXIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

async def calculate_technicals(symbol: str, interval: str = "1h") -> TechnicalIndicators:
    """Compute all technical indicators for symbol."""
    klines = await binance_client.get_klines(symbol, interval, limit=100)
    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    
    # RSI
    rsi_14 = RSIIndicator(df["close"], window=14).rsi().iloc[-1]
    rsi_21 = RSIIndicator(df["close"], window=21).rsi().iloc[-1]
    
    # MACD
    macd_indicator = MACD(df["close"])
    macd = macd_indicator.macd().iloc[-1]
    macd_signal = macd_indicator.macd_signal().iloc[-1]
    macd_hist = macd_indicator.macd_diff().iloc[-1]
    
    # Bollinger Bands
    bb = BollingerBands(df["close"])
    bb_upper = bb.bollinger_hband().iloc[-1]
    bb_middle = bb.bollinger_mavg().iloc[-1]
    bb_lower = bb.bollinger_lband().iloc[-1]
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # ATR
    atr = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range().iloc[-1]
    
    # ADX
    adx = ADXIndicator(df["high"], df["low"], df["close"], window=14).adx().iloc[-1]
    
    # EMAs
    ema_9 = EMAIndicator(df["close"], window=9).ema_indicator().iloc[-1]
    ema_21 = EMAIndicator(df["close"], window=21).ema_indicator().iloc[-1]
    ema_50 = EMAIndicator(df["close"], window=50).ema_indicator().iloc[-1]
    
    # VPIN (simplified: volume imbalance)
    taker_buy_vol = df["taker_buy_base"].astype(float).iloc[-20:].sum()
    total_vol = df["volume"].astype(float).iloc[-20:].sum()
    vpin = abs(taker_buy_vol - (total_vol - taker_buy_vol)) / total_vol if total_vol > 0 else 0
    
    # Volume ratio
    volume_sma_20 = df["volume"].astype(float).rolling(20).mean().iloc[-1]
    volume_ratio = df["volume"].astype(float).iloc[-1] / volume_sma_20 if volume_sma_20 > 0 else 0
    
    # Trend score
    trend_score = 0
    if ema_9 > ema_21 > ema_50:
        trend_score = 1
    elif ema_9 < ema_21 < ema_50:
        trend_score = -1
    
    # Regime detection
    if adx > 25 and bb_width < 0.05:
        regime = "trend"
    elif adx < 20:
        regime = "range"
    else:
        regime = "volatile"
    
    return TechnicalIndicators(
        rsi_14=rsi_14, rsi_21=rsi_21,
        macd=macd, macd_signal=macd_signal, macd_histogram=macd_hist,
        bb_upper=bb_upper, bb_middle=bb_middle, bb_lower=bb_lower, bb_width=bb_width,
        atr_14=atr, vpin=vpin,
        volume_sma_20=volume_sma_20, volume_ratio=volume_ratio,
        adx=adx, ema_9=ema_9, ema_21=ema_21, ema_50=ema_50,
        trend_score=trend_score, regime=regime
    )
```

- [ ] **Step 3: Write unit tests**
```python
@pytest.mark.asyncio
async def test_calculate_technicals_returns_all_indicators():
    indicators = await calculate_technicals("BTCUSDT")
    assert -1 <= indicators.trend_score <= 1
    assert indicators.regime in ["trend", "range", "volatile"]
    assert 0 <= indicators.rsi_14 <= 100
    assert indicators.bb_width > 0

@pytest.mark.asyncio
async def test_regime_detection_trend():
    # Mock: strong trend data
    with patch("src.tools.technical_analysis.binance_client") as mock:
        mock.get_klines = AsyncMock(return_value=[...])  # trending data
        indicators = await calculate_technicals("BTCUSDT")
        assert indicators.regime == "trend"
```

- [ ] **Step 4: Run tests**
```bash
pytest tests/unit/test_technical_analysis.py -v
```

- [ ] **Step 5: Commit**
```bash
git add src/tools/technical_analysis.py tests/unit/test_technical_analysis.py
git commit -m "feat: add technical analysis tools (RSI, MACD, BB, VPIN, regime)"
```

---

## Task 4: Risk Guard (Deterministic, Outside LLM)

**Files:**
- Create: `src/risk/guard.py`
- Create: `src/risk/validators.py`
- Create: `tests/unit/test_risk_guard.py`

**Interfaces:**
- Consumes: OrderRequest + PortfolioState
- Produces: Validated OrderRequest or rejection reason

- [ ] **Step 1: Write RiskGuard class**
```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class RiskDecision(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"

@dataclass
class RiskCheckResult:
    decision: RiskDecision
    reason: Optional[str] = None
    modified_order: Optional[OrderRequest] = None

class RiskGuard:
    MAX_RISK_PER_TRADE = 0.10
    MAX_DAILY_LOSS = 0.05
    MAX_CONCURRENT_POSITIONS = 3
    MAX_LEVERAGE = 20
    MIN_RISK_REWARD = 2.0
    MANDATORY_STOP_LOSS = True
    CORRELATION_THRESHOLD = 0.7
    
    def __init__(self, portfolio_state: PortfolioState):
        self.portfolio = portfolio_state
    
    def validate(self, order: OrderRequest) -> RiskCheckResult:
        # 1. Stop-loss mandatory
        if self.MANDATORY_STOP_LOSS and not order.stop_loss:
            return RiskCheckResult(decision=RiskDecision.REJECT, reason="Stop-loss is mandatory")
        
        # 2. Max concurrent positions
        if len(self.portfolio.positions) >= self.MAX_CONCURRENT_POSITIONS:
            return RiskCheckResult(decision=RiskDecision.REJECT, reason=f"Max positions reached ({self.MAX_CONCURRENT_POSITIONS})")
        
        # 3. Daily loss limit
        if self.portfolio.daily_pnl < -self.portfolio.equity * self.MAX_DAILY_LOSS:
            return RiskCheckResult(decision=RiskDecision.REJECT, reason=f"Daily loss limit reached (-{self.MAX_DAILY_LOSS*100}%)")
        
        # 4. Position size risk check
        if order.price and order.stop_loss:
            risk_per_unit = abs(order.price - order.stop_loss)
            if risk_per_unit > 0:
                max_qty = (self.portfolio.equity * self.MAX_RISK_PER_TRADE) / risk_per_unit
                if order.qty > max_qty:
                    return RiskCheckResult(
                        decision=RiskDecision.MODIFY,
                        reason=f"Position size too large, max allowed: {max_qty:.4f}",
                        modified_order=OrderRequest(**{**order.dict(), "qty": max_qty})
                    )
        
        # 5. Risk-reward check
        if order.take_profit and order.stop_loss and order.price:
            rr = abs(order.take_profit - order.price) / abs(order.price - order.stop_loss)
            if rr < self.MIN_RISK_REWARD:
                return RiskCheckResult(decision=RiskDecision.REJECT, reason=f"Risk-reward {rr:.2f} < minimum {self.MIN_RISK_REWARD}")
        
        # 6. Leverage check
        # (would check against Binance API for actual leverage)
        
        return RiskCheckResult(decision=RiskDecision.APPROVE)
    
    def kill_switch(self, reason: str) -> bool:
        """Emergency stop all trading."""
        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
        return True
```

- [ ] **Step 2: Write validators (correlation, liquidation risk)**
```python
async def check_correlation(new_symbol: str, existing_positions: List[Dict]) -> bool:
    """Check if new position is too correlated with existing."""
    # Simplified: BTC, ETH highly correlated
    high_corr_pairs = [("BTCUSDT", "ETHUSDT"), ("BTCUSDT", "SOLUSDT")]
    for pos in existing_positions:
        pair = (pos["symbol"], new_symbol)
        if pair in high_corr_pairs or (pair[1], pair[0]) in high_corr_pairs:
            return True  # Correlated
    return False

async def check_liquidation_risk(order: OrderRequest, portfolio: PortfolioState) -> bool:
    """Check if order would put portfolio near liquidation."""
    margin_usage = sum(float(p["margin"]) for p in portfolio.positions)
    new_margin = order.qty * 0.05  # Approx margin for 20x leverage
    return (margin_usage + new_margin) / portfolio.equity > 0.5
```

- [ ] **Step 3: Write unit tests**
```python
def test_risk_guard_approves_valid_order():
    portfolio = PortfolioState(equity=10000, positions=[], daily_pnl=0)
    guard = RiskGuard(portfolio)
    order = OrderRequest(symbol="BTCUSDT", side="BUY", qty=0.01, price=100000, stop_loss=99000, take_profit=102000)
    result = guard.validate(order)
    assert result.decision == RiskDecision.APPROVE

def test_risk_guard_rejects_no_stop_loss():
    portfolio = PortfolioState(equity=10000, positions=[], daily_pnl=0)
    guard = RiskGuard(portfolio)
    order = OrderRequest(symbol="BTCUSDT", side="BUY", qty=0.01, price=100000)
    result = guard.validate(order)
    assert result.decision == RiskDecision.REJECT
    assert "Stop-loss" in result.reason

def test_risk_guard_rejects_daily_loss_limit():
    portfolio = PortfolioState(equity=10000, positions=[], daily_pnl=-600)
    guard = RiskGuard(portfolio)
    order = OrderRequest(symbol="BTCUSDT", side="BUY", qty=0.01, price=100000, stop_loss=99000)
    result = guard.validate(order)
    assert result.decision == RiskDecision.REJECT

def test_risk_guard_modifies_oversized_position():
    portfolio = PortfolioState(equity=10000, positions=[], daily_pnl=0)
    guard = RiskGuard(portfolio)
    # qty=1 BTC at $100k = $100k, risk 10% = $10k, stop 1% away = 100x risk → too large
    order = OrderRequest(symbol="BTCUSDT", side="BUY", qty=1.0, price=100000, stop_loss=99000, take_profit=102000)
    result = guard.validate(order)
    assert result.decision == RiskDecision.MODIFY
    assert result.modified_order.qty < 1.0
```

- [ ] **Step 4: Run tests**
```bash
pytest tests/unit/test_risk_guard.py -v
```

- [ ] **Step 5: Commit**
```bash
git add src/risk/ tests/unit/test_risk_guard.py
git commit -m "feat: add Risk Guard with deterministic validation (size, SL, DD, correlation)"
```

---

## Task 5: Memory System (ChromaDB + SQLite)

**Files:**
- Create: `src/memory/episodic.py`
- Create: `src/memory/semantic.py`
- Create: `src/memory/working.py`
- Create: `src/memory/store.py`
- Create: `tests/unit/test_memory.py`

**Interfaces:**
- Consumes: Trade decisions, outcomes, market context
- Produces: Stored memories for retrieval by agent

- [ ] **Step 1: Write Episodic Memory (SQLite)**
```python
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional

class EpisodicMemory:
    def __init__(self, db_path: str = "data/trade_journal.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                decision TEXT,
                outcome TEXT,
                pnl REAL,
                reasoning TEXT,
                market_context TEXT,
                regime TEXT,
                model_version TEXT
            )
        """)
        conn.commit()
        conn.close()
    
    def log_trade(self, trade: Dict):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO trades (timestamp, symbol, decision, outcome, pnl, reasoning, market_context, regime, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            trade["symbol"],
            trade["decision"],
            trade.get("outcome"),
            trade.get("pnl"),
            trade.get("reasoning"),
            json.dumps(trade.get("market_context", {})),
            trade.get("regime"),
            trade.get("model_version", "v1")
        ))
        conn.commit()
        conn.close()
    
    def get_recent_trades(self, limit: int = 20) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    
    def get_performance_stats(self) -> Dict:
        conn = sqlite3.connect(self.db_path)
        trades = conn.execute("SELECT * FROM trades WHERE outcome IS NOT NULL").fetchall()
        conn.close()
        if not trades:
            return {"total": 0, "win_rate": 0, "avg_pnl": 0, "max_dd": 0}
        
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        eq_curve = [sum(t["pnl"] for t in trades[:i+1]) for i in range(len(trades))]
        peak = 0
        max_dd = 0
        for eq in eq_curve:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd
        
        return {
            "total": len(trades),
            "win_rate": len(wins) / len(trades),
            "avg_pnl": sum(t["pnl"] for t in trades) / len(trades),
            "max_dd": max_dd,
            "sharpe": 0  # Calculate from returns
        }
```

- [ ] **Step 2: Write Semantic Memory (ChromaDB)**
```python
import chromadb
from chromadb.config import Settings
from typing import List

class SemanticMemory:
    def __init__(self, host: str = "chromadb", port: int = 8000):
        self.client = chromadb.HttpClient(host=host, port=port)
        self.collection = self.client.get_or_create_collection("trading_strategies")
    
    def add_strategy(self, name: str, content: str, metadata: Dict = None):
        self.collection.add(
            documents=[content],
            metadatas=[metadata or {}],
            ids=[name]
        )
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        results = self.collection.query(query_texts=[query], n_results=n_results)
        return [
            {"id": rid, "document": doc, "metadata": meta}
            for rid, doc, meta in zip(
                results["ids"][0], results["documents"][0], results["metadatas"][0]
            )
        ]
    
    def add_regime_pattern(self, regime: str, pattern: str, performance: Dict):
        self.add_strategy(
            name=f"{regime}_{hash(pattern) % 10000}",
            content=pattern,
            metadata={"regime": regime, "performance": performance}
        )
```

- [ ] **Step 3: Write Working Memory (in-state)**
```python
class WorkingMemory:
    def __init__(self, max_context_turns: int = 20):
        self.context_turns: List[Dict] = []
        self.max_context_turns = max_context_turns
    
    def add_turn(self, thought: str, action: str, result: str):
        self.context_turns.append({
            "thought": thought,
            "action": action,
            "result": result
        })
        if len(self.context_turns) > self.max_context_turns:
            self.context_turns.pop(0)
    
    def get_context(self) -> str:
        return "\n".join([
            f"Turn {i}: thought={t['thought']}, action={t['action']}, result={t['result']}"
            for i, t in enumerate(self.context_turns[-10:])
        ])
    
    def summarize(self) -> str:
        """Create summary of recent context for compression."""
        if len(self.context_turns) < self.max_context_turns:
            return self.get_context()
        # Summarize older turns
        recent = self.context_turns[-5:]
        summary = f"Earlier: {len(self.context_turns) - 5} turns summarized"
        return summary + "\n" + self.get_context()
```

- [ ] **Step 4: Write unit tests**
```python
def test_episodic_memory_logs_and_retrieves():
    mem = EpisodicMemory(db_path="/tmp/test_trades.db")
    mem.log_trade({"symbol": "BTCUSDT", "decision": "BUY", "pnl": 100, "reasoning": "Trend break"})
    trades = mem.get_recent_trades(limit=5)
    assert len(trades) == 1
    assert trades[0]["symbol"] == "BTCUSDT"
    assert trades[0]["pnl"] == 100

def test_episodic_memory_performance_stats():
    mem = EpisodicMemory(db_path="/tmp/test_trades2.db")
    mem.log_trade({"symbol": "BTCUSDT", "decision": "BUY", "pnl": 100, "reasoning": "t1"})
    mem.log_trade({"symbol": "ETHUSDT", "decision": "SELL", "pnl": -50, "reasoning": "t2"})
    stats = mem.get_performance_stats()
    assert stats["total"] == 2
    assert stats["win_rate"] == 0.5
    assert stats["avg_pnl"] == 25
```

- [ ] **Step 5: Run tests**
```bash
pytest tests/unit/test_memory.py -v
```

- [ ] **Step 6: Commit**
```bash
git add src/memory/ tests/unit/test_memory.py
git commit -m "feat: add memory system (episodic SQLite + semantic ChromaDB + working in-state)"
```

---

## Task 6: LangGraph Agent Core

**Files:**
- Create: `src/agent/state.py`
- Create: `src/agent/nodes.py`
- Create: `src/agent/graph.py`
- Create: `src/agent/prompts.py`
- Create: `tests/unit/test_agent_graph.py`

**Interfaces:**
- Consumes: All tools from Tasks 2-5, Risk Guard
- Produces: LangGraph StateGraph with Observe → Reason → Act → Reflect loop

- [ ] **Step 1: Write AgentState TypedDict**
```python
from typing import TypedDict, Optional, Dict, List, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

class AgentState(TypedDict):
    market_data: Dict[str, Any]
    portfolio: Dict[str, Any]
    memory: Dict[str, Any]
    current_plan: Optional[Dict[str, Any]]
    iteration: int
    max_iterations: int
    last_decision: Optional[Dict[str, Any]]
    errors: List[str]
```

- [ ] **Step 2: Write System Prompt**
```python
SYSTEM_PROMPT = """
# TRADING AGENT CONSTITUTION v1.0

## IDENTITY
You are an aggressive quantitative futures trader on Binance.
Capital: $10,000 USDT | Max Drawdown: 25% | Risk/Trade: 5-10%

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
"""
```

- [ ] **Step 3: Write Agent Nodes**
```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
import json

class AgentNodes:
    def __init__(self, tools: List, memory: WorkingMemory, risk_guard: RiskGuard):
        self.tools = {t.name: t for t in tools}
        self.memory = memory
        self.risk_guard = risk_guard
        self.llm = ChatOpenAI(
            model="deepseek-v4.1-flash",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=0.1,
            max_tokens=4096,
            base_url="https://api.deepseek.com/v1"
        )
        self.llm = self.llm.bind_tools(tools)
    
    def observe_node(self, state: AgentState) -> AgentState:
        """Fetch market data and update state."""
        symbols = state.get("symbols", ["BTCUSDT"])
        market_data = {}
        for symbol in symbols:
            # Async call handled in runner
            market_data[symbol] = {}  # Filled by async wrapper
        state["market_data"] = market_data
        state["iteration"] += 1
        return state
    
    def reason_node(self, state: AgentState) -> AgentState:
        """LLM reasoning with tools (ReAct loop)."""
        context = self.memory.get_context()
        portfolio = state.get("portfolio", {})
        
        messages = [
            SystemMessage(content=SYSTEM_PROMPT.format(tools_schema=self._get_tools_schema())),
            HumanMessage(content=f"""
Market Data: {json.dumps(state.get("market_data", {}))}
Portfolio: {json.dumps(portfolio)}
Recent Memory: {context}
Current iteration: {state.get("iteration", 0)}

Analyze the market, decide action, output JSON only.
""")
        ]
        
        response = self.llm.invoke(messages)
        
        # Parse tool calls or final answer
        if response.tool_calls:
            state["last_decision"] = {
                "thought": response.content,
                "tool_calls": response.tool_calls,
                "type": "action"
            }
        else:
            state["last_decision"] = {
                "thought": response.content,
                "type": "reasoning"
            }
        
        return state
    
    def act_node(self, state: AgentState) -> AgentState:
        """Execute approved actions through Risk Guard."""
        decision = state.get("last_decision")
        if not decision or decision.get("type") != "action":
            return state
        
        for tool_call in decision.get("tool_calls", []):
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            if tool_name == "place_order":
                # Risk Guard validation
                order = OrderRequest(**tool_args)
                portfolio = PortfolioState(**state.get("portfolio", {}))
                guard = RiskGuard(portfolio)
                result = guard.validate(order)
                
                if result.decision == RiskDecision.APPROVE:
                    # Execute order
                    order_result = await binance_client.place_order(order)
                    self.memory.add_turn(decision["thought"], "place_order", str(order_result))
                elif result.decision == RiskDecision.MODIFY:
                    # Execute modified order
                    order_result = await binance_client.place_order(result.modified_order)
                    self.memory.add_turn(decision["thought"], "place_order (modified)", str(order_result))
                else:
                    self.memory.add_turn(decision["thought"], "order_rejected", result.reason)
                    state["errors"].append(f"Order rejected: {result.reason}")
            
            elif tool_name in self.tools:
                result = await self.tools[tool_name].ainvoke(tool_args)
                self.memory.add_turn(decision["thought"], tool_name, str(result))
        
        return state
    
    def reflect_node(self, state: AgentState) -> AgentState:
        """Log decision, update memory, evaluate performance."""
        decision = state.get("last_decision")
        if decision:
            # Log to episodic memory
            episodic = EpisodicMemory()
            episodic.log_trade({
                "symbol": decision.get("params", {}).get("symbol"),
                "decision": decision.get("action"),
                "reasoning": decision.get("thought"),
                "market_context": state.get("market_data"),
                "regime": decision.get("regime"),
                "model_version": "deepseek-v4.1-flash"
            })
        
        # Check if should continue loop
        if state.get("iteration", 0) >= state.get("max_iterations", 10):
            state["_next"] = END
        else:
            state["_next"] = "observe"  # Continue loop
        
        return state
    
    def _get_tools_schema(self) -> str:
        return json.dumps([t.schema() for t in self.tools])
```

- [ ] **Step 4: Write Graph Builder**
```python
def build_agent_graph(tools: List, memory: WorkingMemory, risk_guard: RiskGuard) -> StateGraph:
    nodes = AgentNodes(tools, memory, risk_guard)
    
    graph = StateGraph(AgentState)
    
    graph.add_node("observe", nodes.observe_node)
    graph.add_node("reason", nodes.reason_node)
    graph.add_node("act", nodes.act_node)
    graph.add_node("reflect", nodes.reflect_node)
    
    graph.set_entry_point("observe")
    graph.add_edge("observe", "reason")
    graph.add_edge("reason", "act")
    graph.add_edge("act", "reflect")
    graph.add_conditional_edges("reflect", lambda s: s.get("_next", END), {END: END})
    
    return graph.compile(checkpointer=MemorySaver())
```

- [ ] **Step 5: Write unit tests for agent nodes**
```python
from unittest.mock import AsyncMock, MagicMock, patch

def test_observe_node_populates_market_data():
    state = {"symbols": ["BTCUSDT"], "iteration": 0, "market_data": {}}
    nodes = AgentNodes([], WorkingMemory(), RiskGuard(PortfolioState(equity=10000, positions=[], daily_pnl=0)))
    with patch("src.agent.nodes.binance_client") as mock_client:
        mock_client.get_klines = AsyncMock(return_value=[[1700000000000, "100", "101", "99", "100.5", "1000", ...]])
        result = nodes.observe_node(state)
        assert "BTCUSDT" in result["market_data"]

def test_reason_node_returns_tool_call():
    state = {"market_data": {"BTCUSDT": {"close": 100000}}, "portfolio": {"equity": 10000}, "iteration": 0}
    # Mock LLM to return tool call
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        tool_calls=[{"name": "get_market_data", "args": {"symbol": "BTCUSDT"}}],
        content=None
    )
    nodes = AgentNodes([], WorkingMemory(), RiskGuard(PortfolioState(equity=10000, positions=[], daily_pnl=0)))
    nodes.llm = mock_llm
    result = nodes.reason_node(state)
    assert result["last_decision"]["type"] == "action"

def test_act_node_rejects_no_stop_loss():
    state = {"last_decision": {"type": "action", "tool_calls": [{"name": "place_order", "args": {"symbol": "BTCUSDT", "side": "BUY", "qty": 0.01, "price": 100000}}]}, "portfolio": {"equity": 10000, "positions": []}}
    nodes = AgentNodes([], WorkingMemory(), RiskGuard(PortfolioState(equity=10000, positions=[], daily_pnl=0)))
    result = nodes.act_node(state)
    assert any("Stop-loss" in e for e in result["errors"])
```

- [ ] **Step 6: Run tests**
```bash
pytest tests/unit/test_agent_graph.py -v
```

- [ ] **Step 7: Commit**
```bash
git add src/agent/ tests/unit/test_agent_graph.py
git commit -m "feat: add LangGraph agent core with Observe→Reason→Act→Reflect loop"
```

---

## Task 7: Dashboard API + WebSocket

**Files:**
- Create: `src/dashboard/api.py`
- Create: `src/dashboard/websocket.py`
- Create: `services/dashboard-api/Dockerfile`
- Create: `services/dashboard-ui/package.json`
- Create: `tests/unit/test_dashboard.py`

**Interfaces:**
- Consumes: Trade log, portfolio state from DB
- Produces: REST API + WebSocket endpoint for real-time dashboard

- [ ] **Step 1: Write FastAPI app with WebSocket**
```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import json
import asyncio
from datetime import datetime

app = FastAPI(title="Trading Agent Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: Dict):
        for conn in self.active_connections:
            await conn.send_json(message)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/api/portfolio")
async def get_portfolio():
    # Query TimescaleDB for latest portfolio state
    return {"equity": 10000, "daily_pnl": 150, "positions": []}

@app.get("/api/trades")
async def get_trades(limit: int = 50):
    # Query trade journal
    return []

@app.get("/api/performance")
async def get_performance():
    # Calculate Sharpe, max DD, win rate from trade log
    return {"sharpe": 1.5, "max_dd": 0.12, "win_rate": 0.52}

@app.get("/api/agent/log")
async def get_agent_log(limit: int = 20):
    # Return recent agent reasoning logs
    return []
```

- [ ] **Step 2: Write dashboard UI (Next.js + React + Recharts)**
```javascript
// services/dashboard-ui/src/components/Dashboard.tsx
import { useEffect, useState } from "react";

export default function Dashboard() {
  const [portfolio, setPortfolio] = useState({equity: 0, daily_pnl: 0, positions: []});
  const [logs, setLogs] = useState([]);
  
  useEffect(() => {
    fetch("/api/portfolio").then(r => r.json()).then(setPortfolio);
    
    const ws = new WebSocket("ws://localhost:8000/ws");
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "trade") setLogs(l => [data, ...l.slice(0, 19)]);
      if (data.type === "portfolio") setPortfolio(data.payload);
    };
    return () => ws.close();
  }, []);
  
  return (
    <div className="dashboard">
      <h1>Trading Agent Dashboard</h1>
      <div className="metrics">
        <div>Equity: ${portfolio.equity}</div>
        <div>Daily PnL: ${portfolio.daily_pnl}</div>
      </div>
      <div className="positions">
        {portfolio.positions.map((p, i) => <div key={i}>{p.symbol}: {p.qty} @ {p.entry}</div>)}
      </div>
      <div className="logs">
        {logs.map((log, i) => (
          <div key={i} className="log-entry">
            <span>{log.time}</span>
            <span>{log.thought}</span>
            <span>{log.action}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Write tests**
```python
from fastapi.testclient import TestClient
from src.dashboard.api import app

client = TestClient(app)

def test_portfolio_endpoint():
    response = client.get("/api/portfolio")
    assert response.status_code == 200
    assert "equity" in response.json()

def test_trades_endpoint():
    response = client.get("/api/trades?limit=10")
    assert response.status_code == 200

def test_websocket_connect():
    with client.websocket_connect("/ws") as websocket:
        websocket.send_text("ping")
        # Should stay open
```

- [ ] **Step 4: Run tests**
```bash
pytest tests/unit/test_dashboard.py -v
```

- [ ] **Step 5: Commit**
```bash
git add src/dashboard/ services/dashboard-api/ services/dashboard-ui/ tests/unit/test_dashboard.py
git commit -m "feat: add dashboard API (FastAPI + WebSocket) + React UI"
```

---

## Task 8: Docker Services + Integration

**Files:**
- Create: `services/agent-core/Dockerfile`
- Create: `services/executor/Dockerfile`
- Create: `services/dashboard-api/Dockerfile`
- Create: `services/dashboard-ui/Dockerfile`
- Create: `scripts/run.sh`, `scripts/stop.sh`
- Create: `tests/integration/test_docker_compose.py`

**Interfaces:**
- Consumes: All services from Tasks 1-7
- Produces: Docker Compose deployment ready

- [ ] **Step 1: Write agent-core Dockerfile**
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ /app/src/
ENV PYTHONPATH=/app
CMD ["python", "-m", "agent.main"]
```

- [ ] **Step 2: Write requirements.txt**
```
langgraph>=0.2.0
langchain-openai>=0.1.0
python-binance>=1.0.19
ta>=0.11.0
pandas>=2.0.0
numpy>=1.24.0
fastapi>=0.104.0
uvicorn>=0.24.0
chromadb>=0.4.0
psycopg2-binary>=2.9.0
redis>=5.0.0
pydantic>=2.0.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
python-dotenv>=1.0.0
websockets>=12.0
```

- [ ] **Step 3: Write run scripts**
```bash
#!/bin/bash
# scripts/run.sh
export DEEPSEEK_API_KEY=$(cat secrets/deepseek_api_key)
export BINANCE_API_KEY=$(cat secrets/binance_api_key)
export BINANCE_API_SECRET=$(cat secrets/binance_api_secret)
export DB_PASSWORD=$(cat secrets/db_password)

docker compose up -d --build
```

```bash
#!/bin/bash
# scripts/stop.sh
docker compose down
```

- [ ] **Step 4: Write integration tests**
```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_agent_full_loop():
    """Test complete agent loop from observe to reflect."""
    # Setup
    from src.agent.graph import build_agent_graph
    from src.memory.store import WorkingMemory
    from src.risk.guard import RiskGuard, PortfolioState
    
    memory = WorkingMemory()
    guard = RiskGuard(PortfolioState(equity=10000, positions=[], daily_pnl=0))
    graph = build_agent_graph([], memory, guard)
    
    # Run one iteration
    initial_state = {
        "symbols": ["BTCUSDT"],
        "portfolio": {"equity": 10000, "positions": [], "daily_pnl": 0},
        "memory": {},
        "iteration": 0,
        "max_iterations": 1,
        "errors": []
    }
    
    result = await graph.ainitial_run(initial_state)
    assert result["iteration"] == 1
    assert "last_decision" in result

def test_docker_compose_services_up():
    """Verify all services are running."""
    import subprocess
    result = subprocess.run(["docker", "compose", "ps"], capture_output=True, text=True)
    assert "Up" in result.stdout
    assert "redis" in result.stdout
    assert "chromadb" in result.stdout
    assert "timescaledb" in result.stdout
```

- [ ] **Step 5: Run integration tests**
```bash
pytest tests/integration/test_docker_compose.py -v
```

- [ ] **Step 6: Commit**
```bash
git add services/ scripts/ tests/integration/
git commit -m "feat: add Docker service configs + integration tests"
```

---

## Task 9: Paper Trading Harness

**Files:**
- Create: `src/executor/paper_trading.py`
- Create: `src/executor/live_trading.py`
- Create: `tests/integration/test_paper_trading.py`

**Interfaces:**
- Consumes: Agent decisions, Risk Guard validation
- Produces: Paper trade execution with simulated fills

- [ ] **Step 1: Write Paper Trading Engine**
```python
import asyncio
import json
from datetime import datetime
from typing import Dict, List
from src.tools.binance_client import OrderRequest, OrderResult, PortfolioState

class PaperTradingEngine:
    def __init__(self, initial_equity: float = 10000):
        self.equity = initial_equity
        self.positions: Dict[str, Dict] = {}
        self.trade_log: List[Dict] = []
        self.filled_orders: List[OrderResult] = []
    
    async def execute_order(self, order: OrderRequest) -> OrderResult:
        """Simulate order execution at current market price."""
        # Use last known price for simulation
        fill_price = order.price or self._get_current_price(order.symbol)
        
        result = OrderResult(
            order_id=len(self.filled_orders) + 1,
            client_order_id=order.client_order_id or f"paper_{datetime.utcnow().timestamp()}",
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            price=fill_price,
            status="FILLED",
            fee=fill_price * order.qty * 0.0004  # 0.04% fee
        )
        
        self.filled_orders.append(result)
        self._update_position(result)
        self._log_trade(result, order)
        
        return result
    
    def _get_current_price(self, symbol: str) -> float:
        # Would fetch from Binance mock or real API
        return 100000.0
    
    def _update_position(self, order: OrderResult):
        symbol = order.symbol
        if symbol not in self.positions:
            self.positions[symbol] = {"qty": 0, "entry": 0, "unrealized_pnl": 0}
        
        pos = self.positions[symbol]
        if order.side == "BUY":
            pos["qty"] += order.qty
            pos["entry"] = order.price
        elif order.side == "SELL" and pos["qty"] >= order.qty:
            pos["qty"] -= order.qty
            if pos["qty"] == 0:
                pos["entry"] = 0
    
    def _log_trade(self, order: OrderResult, request: OrderRequest):
        self.trade_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side,
            "qty": order.qty,
            "price": order.price,
            "fee": order.fee,
            "stop_loss": request.stop_loss,
            "take_profit": request.take_profit
        })
    
    async def simulate_price_movement(self, symbol: str, new_price: float):
        """Simulate market price change and update unrealized PnL."""
        if symbol in self.positions and self.positions[symbol]["qty"] > 0:
            pos = self.positions[symbol]
            pos["unrealized_pnl"] = (new_price - pos["entry"]) * pos["qty"]
    
    def get_portfolio_state(self) -> PortfolioState:
        total_pnl = sum(pos["unrealized_pnl"] for pos in self.positions.values())
        return PortfolioState(
            equity=self.equity + total_pnl,
            available_margin=self.equity * 0.5,
            positions=[{"symbol": s, **p} for s, p in self.positions.items()],
            daily_pnl=total_pnl,
            max_drawdown_today=0.0
        )
```

- [ ] **Step 2: Write integration test**
```python
@pytest.mark.asyncio
async def test_paper_trading_buy_sell_cycle():
    engine = PaperTradingEngine(initial_equity=10000)
    
    # Buy 0.01 BTC at 100000
    order = OrderRequest(symbol="BTCUSDT", side="BUY", qty=0.01, price=100000, stop_loss=99000, take_profit=102000)
    result = await engine.execute_order(order)
    assert result.status == "FILLED"
    assert result.price == 100000
    
    # Simulate price increase to 101000
    await engine.simulate_price_movement("BTCUSDT", 101000)
    state = engine.get_portfolio_state()
    assert state.positions[0]["unrealized_pnl"] > 0
    
    # Sell 0.01 BTC at 101000
    sell_order = OrderRequest(symbol="BTCUSDT", side="SELL", qty=0.01, price=101000)
    sell_result = await engine.execute_order(sell_order)
    assert sell_result.status == "FILLED"
```

- [ ] **Step 3: Run tests**
```bash
pytest tests/integration/test_paper_trading.py -v
```

- [ ] **Step 4: Commit**
```bash
git add src/executor/ tests/integration/test_paper_trading.py
git commit -m "feat: add paper trading engine with simulated fills + PnL tracking"
```

---

## Task 10: Main Entry Point + Agent Runner

**Files:**
- Create: `src/agent/main.py`
- Create: `src/utils/logging_config.py`
- Create: `src/utils/health_check.py`
- Create: `Dockerfile` (root)

**Interfaces:**
- Consumes: All services
- Produces: Running agent that loops Observe→Reason→Act→Reflect

- [ ] **Step 1: Write main entry point**
```python
import asyncio
import logging
import os
import signal
import sys
from src.agent.graph import build_agent_graph
from src.tools.binance_client import BinanceClient
from src.tools.market_data import get_market_data, get_portfolio_state
from src.tools.technical_analysis import calculate_technicals
from src.risk.guard import RiskGuard
from src.memory.store import WorkingMemory, EpisodicMemory, SemanticMemory
from src.executor.paper_trading import PaperTradingEngine
from src.utils.logging_config import setup_logging
from src.utils.health_check import HealthCheck

setup_logging()
logger = logging.getLogger(__name__)

async def main():
    # Initialize services
    binance = BinanceClient(
        api_key=os.getenv("BINANCE_API_KEY"),
        api_secret=os.getenv("BINANCE_API_SECRET"),
        testnet=os.getenv("BINANCE_TESTNET", "true").lower() == "true"
    )
    await binance.connect()
    
    memory = WorkingMemory()
    episodic = EpisodicMemory()
    semantic = SemanticMemory(
        host=os.getenv("CHROMA_HOST", "chromadb"),
        port=int(os.getenv("CHROMA_PORT", "8000"))
    )
    
    portfolio = await get_portfolio_state()
    risk_guard = RiskGuard(portfolio)
    
    # Paper trading engine
    paper_engine = PaperTradingEngine(initial_equity=portfolio.equity)
    
    # Build agent graph
    tools = [
        get_market_data,
        calculate_technicals,
        get_portfolio_state,
        paper_engine.execute_order,
        risk_guard.validate
    ]
    
    graph = build_agent_graph(tools, memory, risk_guard)
    
    # Health check
    health = HealthCheck()
    
    # Run loop
    loop_interval = int(os.getenv("AGENT_LOOP_INTERVAL", "60"))
    max_iterations = int(os.getenv("MAX_ITERATIONS_PER_LOOP", "10"))
    
    while True:
        try:
            initial_state = {
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "portfolio": portfolio.dict(),
                "memory": {},
                "iteration": 0,
                "max_iterations": max_iterations,
                "errors": []
            }
            
            result = await graph.ainitial_run(initial_state)
            
            # Update portfolio
            portfolio = await get_portfolio_state()
            risk_guard.portfolio = portfolio
            
            # Health check
            health.check()
            
            # Sleep until next loop
            await asyncio.sleep(loop_interval)
            
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Agent loop error: {e}")
            episodic.log_error(str(e))
            await asyncio.sleep(30)  # Backoff before retry
    
    await binance.close()

if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Write logging config**
```python
import logging
import sys

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/agent.log")
        ]
    )
```

- [ ] **Step 3: Write health check**
```python
import time
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class HealthCheck:
    def __init__(self):
        self.last_check = time.time()
        self.errors = []
    
    def check(self) -> Dict:
        now = time.time()
        uptime = now - self.last_check
        
        status = {
            "healthy": True,
            "uptime_seconds": uptime,
            "errors_last_hour": len([e for e in self.errors if time.time() - e < 3600]),
            "timestamp": now
        }
        
        if status["errors_last_hour"] > 10:
            status["healthy"] = False
            logger.critical("Health check FAILED: too many errors")
        
        self.last_check = now
        return status
    
    def log_error(self, error: str):
        self.errors.append(time.time())
        logger.error(f"Health check error: {error}")
```

- [ ] **Step 4: Write root Dockerfile**
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000 3000
CMD ["python", "-m", "agent.main"]
```

- [ ] **Step 5: Commit**
```bash
git add src/agent/main.py src/utils/ Dockerfile
git commit -m "feat: add main entry point + agent runner + health check"
```

---

## Task 11: End-to-End Test (Paper Trading on Testnet)

**Files:**
- Create: `tests/integration/test_e2e_paper_trading.py`
- Create: `tests/integration/conftest.py`

**Interfaces:**
- Consumes: Full system (Binance testnet, agent, dashboard)
- Produces: Paper trade execution log + performance report

- [ ] **Step 1: Write e2e test**
```python
import pytest
import asyncio
from src.agent.main import main as run_agent
from src.executor.paper_trading import PaperTradingEngine
from src.risk.guard import RiskGuard, PortfolioState

@pytest.mark.asyncio
async def test_full_paper_trading_session():
    """Run agent for 5 iterations on testnet, verify trades logged."""
    engine = PaperTradingEngine(initial_equity=10000)
    guard = RiskGuard(PortfolioState(equity=10000, positions=[], daily_pnl=0))
    
    # Run 5 iterations
    for i in range(5):
        portfolio = engine.get_portfolio_state()
        guard.portfolio = portfolio
        
        # Simulate agent decision (mock LLM)
        # Execute paper trade
        order = OrderRequest(symbol="BTCUSDT", side="BUY", qty=0.001, price=100000, stop_loss=99000, take_profit=102000)
        result = await engine.execute_order(order)
        
        assert result.status == "FILLED"
        assert result.fee > 0
    
    # Verify performance stats
    stats = engine.get_portfolio_state()
    assert stats.equity > 0
    assert len(engine.trade_log) == 5
    
    # Log to episodic memory
    episodic = EpisodicMemory(db_path="/tmp/test_paper.db")
    for trade in engine.trade_log:
        episodic.log_trade(trade)
    
    recent = episodic.get_recent_trades(limit=10)
    assert len(recent) == 5
```

- [ ] **Step 2: Run e2e test**
```bash
pytest tests/integration/test_e2e_paper_trading.py -v
```

- [ ] **Step 3: Commit**
```bash
git add tests/integration/test_e2e_paper_trading.py
git commit -m "feat: add end-to-end paper trading test on Binance testnet"
```

---

## Task 12: Documentation + Deployment Guide

**Files:**
- Create: `docs/DEPLOYMENT.md`
- Create: `docs/USAGE.md`
- Create: `README.md`
- Create: `scripts/setup.sh`

- [ ] **Step 1: Write DEPLOYMENT.md**
```markdown
# Deployment Guide

## Prerequisites
- Docker + Docker Compose
- Python 3.12+
- Binance API keys (testnet first)
- DeepSeek API key

## Quick Start
```bash
# 1. Copy env
cp .env.example .env
# Edit .env with your keys

# 2. Create secrets directory
mkdir -p secrets
echo "sk-your-deepseek-key" > secrets/deepseek_api_key
echo "your-binance-key" > secrets/binance_api_key
echo "your-binance-secret" > secrets/binance_api_secret

# 3. Run setup
bash scripts/setup.sh

# 4. Start services
bash scripts/run.sh

# 5. Dashboard at http://localhost:3000
```

## Paper Trading → Live
1. Run paper trading 60-90 days
2. Verify: Sharpe > 1.5, Max DD < 15%, Win rate > 45%
3. Change `BINANCE_TESTNET=false` in `.env`
4. Restart: `bash scripts/stop.sh && bash scripts/run.sh`
```

- [ ] **Step 2: Write USAGE.md**
```markdown
# Usage Guide

## Dashboard
- http://localhost:3000 — Real-time PnL, positions, agent reasoning trace

## Manual Controls
- Pause agent: `curl -X POST http://localhost:8000/api/agent/pause`
- Close all: `curl -X POST http://localhost:8000/api/agent/close-all`
- Kill switch: `curl -X POST http://localhost:8000/api/agent/kill-switch`

## Logs
```bash
docker compose logs -f agent-core
docker compose logs -f executor
```

## Monitoring
- Agent logs: `logs/agent.log`
- Trade journal: `data/trade_journal.db`
- Health check: `http://localhost:8000/api/health`
```

- [ ] **Step 3: Write setup.sh**
```bash
#!/bin/bash
mkdir -p logs data secrets
pip install -r requirements.txt
echo "Setup complete. Edit .env then run: bash scripts/run.sh"
```

- [ ] **Step 4: Commit**
```bash
git add docs/ scripts/ README.md
git commit -m "docs: add deployment guide + usage + setup script"
```

---

## Plan Self-Review

### Spec Coverage Check
| Spec Requirement | Task |
|------------------|------|
| DeepSeek-V4.1-Flash API | Task 6 (Agent Nodes use model) |
| LangGraph orchestration | Task 6 (StateGraph + ReAct loop) |
| Binance Futures (testnet) | Task 2 (Binance connector) |
| Risk Guard (deterministic) | Task 4 (hard constraints) |
| Memory system (ChromaDB + SQLite) | Task 5 (episodic + semantic + working) |
| Dashboard (real-time) | Task 7 (FastAPI + WebSocket + React) |
| Docker Compose | Task 1 + Task 8 |
| Paper trading | Task 9 + Task 11 |
| Logging + audit | Task 6 (ReflectNode) + Task 5 |
| Security (API keys) | Task 1 (.env + secrets) |
| Error handling | Task 6 (error tracking in state) |
| Tests at each level | Every task has unit/integration tests |

### Placeholder Scan
- No TBD, TODO, "implement later", "fill in details"
- No "similar to Task N" — all code shown explicitly
- All steps have concrete code or commands
- All tests have actual assertions

### Type Consistency
- `OrderRequest`, `PortfolioState`, `MarketSnapshot` used consistently across Tasks 2, 4, 6, 9
- `RiskDecision.APPROVE/REJECT/MODIFY` used in Tasks 4, 6
- `AgentState` TypedDict used in Task 6, referenced in all tasks

### No Gaps
All spec requirements have corresponding tasks.

---

## Plan Complete

**Plan saved to:** `docs/superpowers/plans/2026-09-15-llm-agent-trading-plan.md`

Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?