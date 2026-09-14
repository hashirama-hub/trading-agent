from datetime import datetime
from typing import Dict, List, Optional

from src.tools.binance_client import OrderRequest, OrderResult, PortfolioState


class PaperTradingEngine:
    def __init__(self, initial_equity: float = 10000):
        self.initial_equity = initial_equity
        self.equity = initial_equity
        self.positions: Dict[str, Dict] = {}
        self.trade_log: List[Dict] = []
        self.filled_orders: List[OrderResult] = []

    async def execute_order(self, order: OrderRequest) -> OrderResult:
        """Simulate order execution at current market price."""
        fill_price = order.price or self._get_current_price(order.symbol)

        result = OrderResult(
            order_id=len(self.filled_orders) + 1,
            client_order_id=order.client_order_id or f"paper_{datetime.utcnow().timestamp()}",
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            price=fill_price,
            status="FILLED",
            fee=fill_price * order.qty * 0.0004,
        )

        self.filled_orders.append(result)
        self._update_position(result)
        self._log_trade(result, order)
        return result

    def _get_current_price(self, symbol: str) -> float:
        return 100000.0

    def _update_position(self, order: OrderResult):
        symbol = order.symbol
        if symbol not in self.positions:
            self.positions[symbol] = {
                "qty": 0.0,
                "entry_price": 0.0,
                "side": "",
                "unrealized_pnl": 0.0,
            }

        pos = self.positions[symbol]
        if order.side == "BUY":
            pos["qty"] += order.qty
            pos["entry_price"] = order.price
            pos["side"] = "LONG"
        elif order.side == "SELL" and pos["qty"] >= order.qty:
            pos["qty"] -= order.qty
            if pos["qty"] == 0:
                pos["entry_price"] = 0
                pos["side"] = ""

    def _log_trade(self, result: OrderResult, request: OrderRequest):
        self.trade_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "order_id": result.order_id,
            "symbol": result.symbol,
            "side": result.side,
            "qty": result.qty,
            "price": result.price,
            "fee": result.fee,
            "stop_loss": request.stop_loss,
            "take_profit": request.take_profit,
            "outcome": "filled",
        })

    async def simulate_price_movement(self, symbol: str, new_price: float):
        """Simulate market price change and update unrealized PnL."""
        if symbol in self.positions and self.positions[symbol]["qty"] > 0:
            pos = self.positions[symbol]
            pos["unrealized_pnl"] = (new_price - pos["entry_price"]) * pos["qty"]

    def get_portfolio_state(self) -> PortfolioState:
        total_pnl = sum(
            pos["unrealized_pnl"] for pos in self.positions.values()
        )
        return PortfolioState(
            equity=self.initial_equity + total_pnl,
            available_margin=self.equity * 0.5,
            positions=[
                {"symbol": s, **p} for s, p in self.positions.items()
            ],
            daily_pnl=total_pnl,
            max_drawdown_today=0.0,
        )

    def get_performance(self) -> Dict:
        if not self.trade_log:
            return {"total_trades": 0, "win_rate": 0, "total_pnl": 0}

        wins = [t for t in self.trade_log if t.get("pnl", 0) > 0]
        total = len(self.trade_log)
        return {
            "total_trades": total,
            "win_rate": len(wins) / total if total > 0 else 0,
            "total_pnl": sum(t.get("pnl", 0) for t in self.trade_log),
            "filled_orders": len(self.filled_orders),
        }

    def close_all_positions(self):
        """Close all open positions."""
        for symbol in list(self.positions.keys()):
            self.positions[symbol]["qty"] = 0
            self.positions[symbol]["side"] = ""
        self.positions.clear()