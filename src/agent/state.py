from typing import TypedDict, Optional, List, Dict, Any


class MarketSnapshot(TypedDict):
    symbol: str
    timeframe: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: float
    ask: float
    bid_vol: float
    ask_vol: float
    funding_rate: Optional[float]


class PortfolioState(TypedDict):
    equity: float
    available_margin: float
    positions: List[Dict]
    daily_pnl: float
    max_drawdown_today: float


class AgentMemory(TypedDict):
    working: str
    episodic_trades: List[Dict]
    strategies: List[Dict]


class Decision(TypedDict):
    thought: str
    action: str
    params: Dict
    risk_check: Dict
    confidence: float


class TradingPlan(TypedDict):
    symbol: str
    side: str
    qty: float
    entry: float
    stop_loss: float
    take_profit: float
    risk_reward: float


class AgentState(TypedDict):
    market_data: Dict[str, MarketSnapshot]
    portfolio: PortfolioState
    memory: AgentMemory
    current_plan: Optional[TradingPlan]
    iteration: int
    max_iterations: int
    last_decision: Optional[Decision]
    errors: List[str]