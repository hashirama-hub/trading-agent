import pytest
from unittest.mock import AsyncMock, patch

from src.executor.paper_trading import PaperTradingEngine
from src.tools.binance_client import OrderRequest


@pytest.fixture
def paper_engine():
    return PaperTradingEngine(initial_equity=10000)


class TestPaperTradingEngine:
    @pytest.mark.asyncio
    async def test_execute_buy_order(self, paper_engine):
        order = OrderRequest(
            symbol="BTCUSDT", side="BUY", qty=0.01, price=100000,
            stop_loss=99000, take_profit=102000,
        )
        result = await paper_engine.execute_order(order)
        assert result.status == "FILLED"
        assert result.price == 100000
        assert result.fee > 0

    @pytest.mark.asyncio
    async def test_execute_sell_order(self, paper_engine):
        # First buy
        buy_order = OrderRequest(
            symbol="BTCUSDT", side="BUY", qty=0.01, price=100000,
            stop_loss=99000, take_profit=102000,
        )
        await paper_engine.execute_order(buy_order)

        # Then sell
        sell_order = OrderRequest(
            symbol="BTCUSDT", side="SELL", qty=0.01, price=101000,
        )
        result = await paper_engine.execute_order(sell_order)
        assert result.status == "FILLED"
        assert result.side == "SELL"

    @pytest.mark.asyncio
    async def test_portfolio_state_after_trade(self, paper_engine):
        order = OrderRequest(
            symbol="BTCUSDT", side="BUY", qty=0.01, price=100000,
            stop_loss=99000, take_profit=102000,
        )
        await paper_engine.execute_order(order)
        state = paper_engine.get_portfolio_state()
        assert len(state.positions) == 1
        assert state.positions[0]["qty"] == 0.01
        assert state.positions[0]["side"] == "LONG"

    @pytest.mark.asyncio
    async def test_simulate_price_movement(self, paper_engine):
        order = OrderRequest(
            symbol="BTCUSDT", side="BUY", qty=0.01, price=100000,
            stop_loss=99000, take_profit=102000,
        )
        await paper_engine.execute_order(order)
        await paper_engine.simulate_price_movement("BTCUSDT", 101000)
        state = paper_engine.get_portfolio_state()
        assert state.positions[0]["unrealized_pnl"] > 0

    @pytest.mark.asyncio
    async def test_get_performance(self, paper_engine):
        for i in range(3):
            order = OrderRequest(
                symbol="BTCUSDT", side="BUY", qty=0.01, price=100000,
                stop_loss=99000, take_profit=102000,
            )
            await paper_engine.execute_order(order)
        perf = paper_engine.get_performance()
        assert perf["total_trades"] == 3
        assert perf["filled_orders"] == 3

    @pytest.mark.asyncio
    async def test_close_all_positions(self, paper_engine):
        order = OrderRequest(
            symbol="BTCUSDT", side="BUY", qty=0.01, price=100000,
            stop_loss=99000, take_profit=102000,
        )
        await paper_engine.execute_order(order)
        assert len(paper_engine.positions) == 1

        paper_engine.close_all_positions()
        assert len(paper_engine.positions) == 0

    def test_initial_equity(self, paper_engine):
        assert paper_engine.equity == 10000
        assert paper_engine.initial_equity == 10000

    @pytest.mark.asyncio
    async def test_multiple_symbols(self, paper_engine):
        btc_order = OrderRequest(
            symbol="BTCUSDT", side="BUY", qty=0.01, price=100000,
            stop_loss=99000, take_profit=102000,
        )
        eth_order = OrderRequest(
            symbol="ETHUSDT", side="BUY", qty=1.0, price=3000,
            stop_loss=2900, take_profit=3200,
        )
        await paper_engine.execute_order(btc_order)
        await paper_engine.execute_order(eth_order)
        assert len(paper_engine.positions) == 2
        assert "BTCUSDT" in paper_engine.positions
        assert "ETHUSDT" in paper_engine.positions

    @pytest.mark.asyncio
    async def test_trade_log(self, paper_engine):
        order = OrderRequest(
            symbol="BTCUSDT", side="BUY", qty=0.01, price=100000,
            stop_loss=99000, take_profit=102000,
        )
        await paper_engine.execute_order(order)
        assert len(paper_engine.trade_log) == 1
        assert paper_engine.trade_log[0]["symbol"] == "BTCUSDT"
        assert paper_engine.trade_log[0]["side"] == "BUY"

    def test_order_request_model(self):
        order = OrderRequest(
            symbol="BTCUSDT", side="BUY", qty=0.01, price=100000,
            stop_loss=99000, take_profit=102000,
        )
        assert order.symbol == "BTCUSDT"
        assert order.qty == 0.01
        assert order.stop_loss == 99000

    @pytest.mark.asyncio
    async def test_pnl_calculation(self, paper_engine):
        order = OrderRequest(
            symbol="BTCUSDT", side="BUY", qty=0.01, price=100000,
            stop_loss=99000, take_profit=102000,
        )
        await paper_engine.execute_order(order)
        await paper_engine.simulate_price_movement("BTCUSDT", 102000)
        state = paper_engine.get_portfolio_state()
        # (102000 - 100000) * 0.01 = 20
        assert state.positions[0]["unrealized_pnl"] == 20.0