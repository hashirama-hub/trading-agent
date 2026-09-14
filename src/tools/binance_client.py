import asyncio
import logging
from datetime import datetime
from typing import Optional, List, Dict

from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException
from pydantic import BaseModel, Field

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
    side: str
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


class BinanceClient:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.client: Optional[AsyncClient] = None
        self.bm: Optional[BinanceSocketManager] = None

    async def connect(self):
        self.client = await AsyncClient.create(
            api_key=self.api_key,
            api_secret=self.api_secret,
            testnet=self.testnet,
        )
        self.bm = BinanceSocketManager(self.client)

    async def get_klines(self, symbol: str, interval: str = "1h", limit: int = 100) -> List[Dict]:
        return await self.client.get_klines(symbol=symbol, interval=interval, limit=limit)

    async def get_ticker(self, symbol: str) -> Dict:
        return await self.client.get_symbol_ticker(symbol=symbol)

    async def get_order_book(self, symbol: str, limit: int = 10) -> Dict:
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
                reduceOnly=order.reduce_only,
            )
            return OrderResult(
                order_id=result["orderId"],
                client_order_id=result.get("clientOrderId", ""),
                symbol=result["symbol"],
                side=result["side"],
                qty=float(result["qty"]),
                price=float(result["price"]),
                status=result["status"],
                fee=float(result.get("commission", 0)),
            )
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
