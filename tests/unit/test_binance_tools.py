import pytest
from unittest.mock import AsyncMock, patch

from src.tools.binance_client import BinanceClient, MarketSnapshot, OrderRequest, PortfolioState


@pytest.fixture
def binance_client():
    return BinanceClient(api_key="test", api_secret="test", testnet=True)


@pytest.mark.asyncio
async def test_binance_client_connect(binance_client):
    with patch("src.tools.binance_client.AsyncClient.create") as mock_create:
        mock_create.return_value = AsyncMock()
        await binance_client.connect()
        assert binance_client.client is not None


@pytest.mark.asyncio
async def test_binance_client_get_klines(binance_client):
    with patch("src.tools.binance_client.AsyncClient.create") as mock_create:
        mock_client = AsyncMock()
        mock_create.return_value = mock_client
        mock_client.get_klines = AsyncMock(return_value=[
            [1700000000000, "100", "101", "99", "100.5", "1000", 1700000060000, "100500", 100, "500", "50000", "0"]
        ])
        binance_client.client = mock_client
        result = await binance_client.get_klines("BTCUSDT")
        assert len(result) == 1
        assert result[0][4] == "100.5"


@pytest.mark.asyncio
async def test_binance_client_get_funding_rate(binance_client):
    with patch("src.tools.binance_client.AsyncClient.create") as mock_create:
        mock_client = AsyncMock()
        mock_create.return_value = mock_client
        mock_client.get_funding_rate = AsyncMock(return_value=[{"fundingRate": "0.0001"}])
        binance_client.client = mock_client
        result = await binance_client.get_funding_rate("BTCUSDT")
        assert result == 0.0001


@pytest.mark.asyncio
async def test_market_snapshot_model():
    snapshot = MarketSnapshot(
        symbol="BTCUSDT",
        timeframe="1h",
        timestamp=__import__("datetime").datetime.utcnow(),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1000.0,
        bid=100.3,
        ask=100.7,
        bid_vol=10.0,
        ask_vol=10.0,
        funding_rate=0.0001,
    )
    assert snapshot.symbol == "BTCUSDT"
    assert snapshot.close == 100.5
    assert snapshot.funding_rate == 0.0001


@pytest.mark.asyncio
async def test_order_request_model():
    order = OrderRequest(
        symbol="BTCUSDT",
        side="BUY",
        qty=0.01,
        price=100000.0,
        stop_loss=99000.0,
        take_profit=102000.0,
    )
    assert order.symbol == "BTCUSDT"
    assert order.side == "BUY"
    assert order.qty == 0.01


@pytest.mark.asyncio
async def test_portfolio_state_model():
    state = PortfolioState(
        equity=10000.0,
        available_margin=5000.0,
        positions=[{"symbol": "BTCUSDT", "positionAmt": "0.5"}],
        daily_pnl=100.0,
        max_drawdown_today=0.0,
    )
    assert state.equity == 10000.0
    assert len(state.positions) == 1


@pytest.mark.asyncio
async def test_market_data_returns_snapshot(binance_client):
    with patch("src.tools.binance_client.AsyncClient.create") as mock_create:
        mock_client = AsyncMock()
        mock_create.return_value = mock_client
        mock_client.get_klines = AsyncMock(return_value=[
            [1700000000000, "100", "101", "99", "100.5", "1000", 1700000060000, "100500", 100, "500", "50000", "0"]
        ])
        mock_client.get_symbol_ticker = AsyncMock(return_value={"bidPrice": "100.3", "askPrice": "100.7"})
        mock_client.get_order_book = AsyncMock(return_value={"bids": [["100.3", "10"]], "asks": [["100.7", "10"]]})
        mock_client.get_funding_rate = AsyncMock(return_value=[{"fundingRate": "0.0001"}])
        binance_client.client = mock_client

        import src.tools.market_data as md
        md.binance_client = mock_client

        snapshot = await md.get_market_data("BTCUSDT")
        assert snapshot.symbol == "BTCUSDT"
        assert snapshot.close == 100.5
        assert snapshot.bid == 100.3


@pytest.mark.asyncio
async def test_portfolio_state_parses_positions(binance_client):
    with patch("src.tools.binance_client.AsyncClient.create") as mock_create:
        mock_client = AsyncMock()
        mock_create.return_value = mock_client
        mock_client.get_account = AsyncMock(return_value={
            "positions": [{"symbol": "BTCUSDT", "positionAmt": "0.5", "unRealizedProfit": "100"}],
            "availableBalance": "5000",
            "totalPositionRisk": [{"entryPrice": "100000"}],
        })
        binance_client.client = mock_client

        import src.tools.market_data as md
        md.binance_client = mock_client

        state = await md.get_portfolio_state()
        assert state.equity == 100000.0
        assert state.daily_pnl == 100.0
        assert len(state.positions) == 1


@pytest.mark.asyncio
async def test_binance_client_place_order(binance_client):
    with patch("src.tools.binance_client.AsyncClient.create") as mock_create:
        mock_client = AsyncMock()
        mock_create.return_value = mock_client
        mock_client.create_order = AsyncMock(return_value={
            "orderId": 12345,
            "clientOrderId": "test_order",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "qty": "0.01",
            "price": "100000",
            "status": "FILLED",
            "commission": "0.4",
        })
        binance_client.client = mock_client

        order = OrderRequest(symbol="BTCUSDT", side="BUY", qty=0.01, price=100000.0)
        result = await binance_client.place_order(order)
        assert result.order_id == 12345
        assert result.client_order_id == "test_order"
        assert result.status == "FILLED"
        assert result.fee == 0.4


@pytest.mark.asyncio
async def test_binance_client_cancel_order(binance_client):
    with patch("src.tools.binance_client.AsyncClient.create") as mock_create:
        mock_client = AsyncMock()
        mock_create.return_value = mock_client
        mock_client.cancel_order = AsyncMock(return_value={"status": "CANCELED"})
        binance_client.client = mock_client

        result = await binance_client.cancel_order("BTCUSDT", 12345)
        assert result["status"] == "CANCELED"


@pytest.mark.asyncio
async def test_binance_client_get_open_orders(binance_client):
    with patch("src.tools.binance_client.AsyncClient.create") as mock_create:
        mock_client = AsyncMock()
        mock_create.return_value = mock_client
        mock_client.get_open_orders = AsyncMock(return_value=[{"symbol": "BTCUSDT", "orderId": 1}])
        binance_client.client = mock_client

        orders = await binance_client.get_open_orders("BTCUSDT")
        assert len(orders) == 1


@pytest.mark.asyncio
async def test_binance_client_close(binance_client):
    with patch("src.tools.binance_client.AsyncClient.create") as mock_create:
        mock_client = AsyncMock()
        mock_create.return_value = mock_client
        mock_client.close_connection = AsyncMock()
        binance_client.client = mock_client
        await binance_client.close()
        mock_client.close_connection.assert_awaited_once()