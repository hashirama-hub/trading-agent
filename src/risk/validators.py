from typing import List, Dict

from src.tools.binance_client import OrderRequest, PortfolioState


async def check_correlation(new_symbol: str, existing_positions: List[Dict]) -> bool:
    """Check if new position is too correlated with existing."""
    high_corr_pairs = [
        ("BTCUSDT", "ETHUSDT"),
        ("BTCUSDT", "SOLUSDT"),
        ("ETHUSDT", "SOLUSDT"),
    ]
    for pos in existing_positions:
        pair = (pos["symbol"], new_symbol)
        if pair in high_corr_pairs or (pair[1], pair[0]) in high_corr_pairs:
            return True  # Correlated
    return False


async def check_liquidation_risk(order: OrderRequest, portfolio: PortfolioState) -> bool:
    """Check if order would put portfolio near liquidation."""
    if not portfolio.positions:
        return False  # No existing positions, no liquidation risk

    margin_usage = sum(float(p.get("margin", 0)) for p in portfolio.positions)
    # Approximate margin for 20x leverage: notional / leverage
    notional = order.qty * order.price if order.price else 0
    new_margin = notional / 20 if notional > 0 else 0

    total_margin_pct = (margin_usage + new_margin) / portfolio.equity if portfolio.equity > 0 else 0
    return total_margin_pct > 0.5