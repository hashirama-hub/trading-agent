import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.tools.binance_client import OrderRequest, PortfolioState

logger = logging.getLogger(__name__)


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
            return RiskCheckResult(
                decision=RiskDecision.REJECT,
                reason="Stop-loss is mandatory",
            )

        # 2. Max concurrent positions
        if len(self.portfolio.positions) >= self.MAX_CONCURRENT_POSITIONS:
            return RiskCheckResult(
                decision=RiskDecision.REJECT,
                reason=f"Max positions reached ({self.MAX_CONCURRENT_POSITIONS})",
            )

        # 3. Daily loss limit
        if self.portfolio.daily_pnl < -self.portfolio.equity * self.MAX_DAILY_LOSS:
            return RiskCheckResult(
                decision=RiskDecision.REJECT,
                reason=f"Daily loss limit reached (-{self.MAX_DAILY_LOSS * 100}%)",
            )

        # 4. Position size risk check
        if order.price and order.stop_loss:
            risk_per_unit = abs(order.price - order.stop_loss)
            if risk_per_unit > 0:
                max_qty = (self.portfolio.equity * self.MAX_RISK_PER_TRADE) / risk_per_unit
                if order.qty > max_qty:
                    modified_dict = {**order.model_dump(), "qty": max_qty}
                    return RiskCheckResult(
                        decision=RiskDecision.MODIFY,
                        reason=f"Position size too large, max allowed: {max_qty:.4f}",
                        modified_order=OrderRequest(**modified_dict),
                    )

        # 5. Risk-reward check
        if order.take_profit and order.stop_loss and order.price:
            rr = abs(order.take_profit - order.price) / abs(order.price - order.stop_loss)
            if rr < self.MIN_RISK_REWARD:
                return RiskCheckResult(
                    decision=RiskDecision.REJECT,
                    reason=f"Risk-reward {rr:.2f} < minimum {self.MIN_RISK_REWARD}",
                )

        # 6. Leverage check (placeholder for Binance API integration)

        return RiskCheckResult(decision=RiskDecision.APPROVE)

    def kill_switch(self, reason: str) -> bool:
        """Emergency stop all trading."""
        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
        return True