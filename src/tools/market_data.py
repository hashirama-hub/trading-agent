import asyncio
from datetime import datetime
from typing import Optional, Dict, List

from binance import AsyncClient

from src.tools.binance_client import MarketSnapshot, PortfolioState


# Module-level client reference (set during initialization)
binance_client: Optional[AsyncClient] = None

TIMEFRAMES = ["5m", "15m", "1h", "4h"]


async def get_market_data(
    symbol: str,
    interval: str = "1h",
    limit: int = 100,
) -> MarketSnapshot:
    """Fetch OHLCV + orderbook for symbol."""
    klines = await binance_client.get_klines(symbol=symbol, interval=interval, limit=limit)
    ticker = await binance_client.get_symbol_ticker(symbol=symbol)
    orderbook = await binance_client.get_orderbook(symbol=symbol, limit=5)
    funding = await binance_client.get_funding_rate(symbol=symbol)

    latest = klines[-1]
    funding_rate = float(funding[0]["fundingRate"]) if funding else None

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
        funding_rate=funding_rate,
    )


async def get_multi_timeframe_data(symbol: str, timeframes: List[str] = None) -> Dict[str, MarketSnapshot]:
    """Fetch market data for multiple timeframes in parallel.

    Args:
        symbol: Trading pair (e.g. "BTCUSDT")
        timeframes: List of intervals (default: ["5m", "15m", "1h", "4h"])

    Returns:
        Dict mapping timeframe to MarketSnapshot
    """
    if timeframes is None:
        timeframes = TIMEFRAMES

    tasks = [get_market_data(symbol, interval=tf) for tf in timeframes]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    data = {}
    for tf, result in zip(timeframes, results):
        if isinstance(result, Exception):
            continue
        data[tf] = result
    return data


async def get_portfolio_state() -> PortfolioState:
    """Get current portfolio: positions, balance, PnL."""
    account = await binance_client.get_account()
    positions = [p for p in account["positions"] if float(p["positionAmt"]) != 0]

    equity = 0.0
    if account.get("totalPositionRisk"):
        equity = float(account["totalPositionRisk"][0].get("entryPrice", 0))
    available = float(account.get("availableBalance", 0))

    daily_pnl = sum(float(p.get("unRealizedProfit", 0)) for p in positions)

    return PortfolioState(
        equity=equity,
        available_margin=available,
        positions=positions,
        daily_pnl=daily_pnl,
        max_drawdown_today=0.0,
    )
