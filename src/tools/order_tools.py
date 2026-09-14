from typing import Optional

from .binance_client import BinanceClient, PortfolioState

binance_client: Optional[BinanceClient] = None


async def get_portfolio_state() -> PortfolioState:
    account = await binance_client.get_account()
    positions = [p for p in account["positions"] if float(p["positionAmt"]) != 0]

    equity = (
        float(account["totalPositionRisk"][0]["entryPrice"])
        if account.get("totalPositionRisk")
        else 0.0
    )
    available = float(account["availableBalance"])

    daily_pnl = sum(float(p["unRealizedProfit"]) for p in positions)

    return PortfolioState(
        equity=equity,
        available_margin=available,
        positions=positions,
        daily_pnl=daily_pnl,
        max_drawdown_today=0.0,
    )
