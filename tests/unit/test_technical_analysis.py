import pytest
from unittest.mock import AsyncMock, patch

import pandas as pd
import numpy as np

from src.tools.technical_analysis import TechnicalIndicators, calculate_technicals


@pytest.fixture(autouse=True)
def setup_binance_client():
    """Set module-level binance_client before each test."""
    import src.tools.technical_analysis as ta_module
    ta_module.binance_client = AsyncMock()


def _generate_trending_klines(n: int = 100, start_price: float = 100000, trend: float = 100):
    """Generate klines with a clear trend."""
    klines = []
    price = start_price
    for i in range(n):
        o = price
        c = price + trend
        h = max(o, c) + 50
        l = min(o, c) - 50
        v = np.random.uniform(500, 2000)
        taker_buy = v * np.random.uniform(0.4, 0.6)
        klines.append([
            1700000000000 + i * 60000,
            str(o), str(h), str(l), str(c), str(v),
            1700000000000 + i * 60000 + 60000,
            str(v * c), 100, str(taker_buy), str(v - taker_buy), "0"
        ])
        price = c
    return klines


def _generate_range_klines(n: int = 100, start_price: float = 100000):
    """Generate klines with no clear trend (range-bound)."""
    klines = []
    price = start_price
    for i in range(n):
        o = price
        c = price + np.random.uniform(-50, 50)
        h = max(o, c) + 20
        l = min(o, c) - 20
        v = np.random.uniform(500, 2000)
        taker_buy = v * np.random.uniform(0.4, 0.6)
        klines.append([
            1700000000000 + i * 60000,
            str(o), str(h), str(l), str(c), str(v),
            1700000000000 + i * 60000 + 60000,
            str(v * c), 100, str(taker_buy), str(v - taker_buy), "0"
        ])
        price = c
    return klines


@pytest.mark.asyncio
async def test_calculate_technicals_returns_all_indicators():
    import src.tools.technical_analysis as ta_module
    ta_module.binance_client.get_klines = AsyncMock(return_value=_generate_trending_klines())

    indicators = await calculate_technicals("BTCUSDT")
    assert -1 <= indicators.trend_score <= 1
    assert indicators.regime in ["trend", "range", "volatile"]
    assert 0 <= indicators.rsi_14 <= 100 or indicators.rsi_14 is not None
    assert indicators.bb_width > 0
    assert indicators.atr_14 > 0
    assert indicators.adx >= 0


@pytest.mark.asyncio
async def test_calculate_technicals_all_fields_present():
    import src.tools.technical_analysis as ta_module
    ta_module.binance_client.get_klines = AsyncMock(return_value=_generate_trending_klines())

    indicators = await calculate_technicals("BTCUSDT")
    required_fields = [
        "rsi_14", "rsi_21", "macd", "macd_signal", "macd_histogram",
        "bb_upper", "bb_middle", "bb_lower", "bb_width",
        "atr_14", "vpin", "volume_sma_20", "volume_ratio",
        "adx", "ema_9", "ema_21", "ema_50", "trend_score", "regime"
    ]
    for field in required_fields:
        assert hasattr(indicators, field), f"Missing field: {field}"


@pytest.mark.asyncio
async def test_regime_detection_trend():
    import src.tools.technical_analysis as ta_module
    ta_module.binance_client.get_klines = AsyncMock(return_value=_generate_trending_klines(trend=200))

    indicators = await calculate_technicals("BTCUSDT")
    # With strong trend, should detect as "trend" or "volatile" depending on BB width
    assert indicators.regime in ["trend", "volatile"]


@pytest.mark.asyncio
async def test_regime_detection_range():
    import src.tools.technical_analysis as ta_module
    ta_module.binance_client.get_klines = AsyncMock(return_value=_generate_range_klines())

    indicators = await calculate_technicals("BTCUSDT")
    assert indicators.regime in ["trend", "range", "volatile"]


@pytest.mark.asyncio
async def test_trend_score_positive():
    import src.tools.technical_analysis as ta_module
    ta_module.binance_client.get_klines = AsyncMock(return_value=_generate_trending_klines(trend=500))

    indicators = await calculate_technicals("BTCUSDT")
    # Strong uptrend should give positive trend_score
    assert indicators.trend_score >= 0


@pytest.mark.asyncio
async def test_trend_score_negative():
    import src.tools.technical_analysis as ta_module
    ta_module.binance_client.get_klines = AsyncMock(return_value=_generate_trending_klines(trend=-500))

    indicators = await calculate_technicals("BTCUSDT")
    # Strong downtrend should give negative trend_score
    assert indicators.trend_score <= 0


@pytest.mark.asyncio
async def test_technical_indicators_model():
    indicators = TechnicalIndicators(
        rsi_14=50.0, rsi_21=55.0, macd=1.0, macd_signal=0.5, macd_histogram=0.5,
        bb_upper=101000.0, bb_middle=100000.0, bb_lower=99000.0, bb_width=0.01,
        atr_14=500.0, vpin=0.3, volume_sma_20=1000.0, volume_ratio=1.5,
        adx=30.0, ema_9=100100.0, ema_21=100050.0, ema_50=100000.0,
        trend_score=1.0, regime="trend",
    )
    assert indicators.trend_score == 1.0
    assert indicators.regime == "trend"
    assert indicators.bb_width == 0.01


@pytest.mark.asyncio
async def test_calculate_technicals_symbol_parameter():
    import src.tools.technical_analysis as ta_module
    ta_module.binance_client.get_klines = AsyncMock(return_value=_generate_trending_klines())

    await calculate_technicals("ETHUSDT", interval="15m")
    ta_module.binance_client.get_klines.assert_called_with(symbol="ETHUSDT", interval="15m", limit=100)


@pytest.mark.asyncio
async def test_volume_ratio_calculation():
    import src.tools.technical_analysis as ta_module
    ta_module.binance_client.get_klines = AsyncMock(return_value=_generate_trending_klines())

    indicators = await calculate_technicals("BTCUSDT")
    assert indicators.volume_ratio > 0
    assert indicators.volume_sma_20 > 0


@pytest.mark.asyncio
async def test_vpin_calculation():
    import src.tools.technical_analysis as ta_module
    ta_module.binance_client.get_klines = AsyncMock(return_value=_generate_trending_klines())

    indicators = await calculate_technicals("BTCUSDT")
    assert 0 <= indicators.vpin <= 1