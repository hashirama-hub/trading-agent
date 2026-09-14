from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands


# Module-level client reference
binance_client: Optional[object] = None


class TechnicalIndicators(BaseModel):
    rsi_14: float
    rsi_21: float
    macd: float
    macd_signal: float
    macd_histogram: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_width: float
    atr_14: float
    vpin: float
    volume_sma_20: float
    volume_ratio: float
    adx: float
    ema_9: float
    ema_21: float
    ema_50: float
    trend_score: float
    regime: str


async def calculate_technicals(symbol: str, interval: str = "1h") -> TechnicalIndicators:
    """Compute all technical indicators for symbol."""
    klines = await binance_client.get_klines(symbol=symbol, interval=interval, limit=100)

    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # RSI
    rsi_14 = float(RSIIndicator(close, window=14).rsi().iloc[-1])
    rsi_21 = float(RSIIndicator(close, window=21).rsi().iloc[-1])

    # MACD
    macd_indicator = MACD(close)
    macd_val = float(macd_indicator.macd().iloc[-1])
    macd_signal_val = float(macd_indicator.macd_signal().iloc[-1])
    macd_hist = float(macd_indicator.macd_diff().iloc[-1])

    # Bollinger Bands
    bb = BollingerBands(close)
    bb_upper = float(bb.bollinger_hband().iloc[-1])
    bb_middle = float(bb.bollinger_mavg().iloc[-1])
    bb_lower = float(bb.bollinger_lband().iloc[-1])
    bb_width = float((bb_upper - bb_lower) / bb_middle)

    # ATR
    atr_14 = float(AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1])

    # ADX
    adx_val = float(ADXIndicator(high, low, close, window=14).adx().iloc[-1])

    # EMAs
    ema_9 = float(EMAIndicator(close, window=9).ema_indicator().iloc[-1])
    ema_21 = float(EMAIndicator(close, window=21).ema_indicator().iloc[-1])
    ema_50 = float(EMAIndicator(close, window=50).ema_indicator().iloc[-1])

    # VPIN (volume imbalance)
    taker_buy_vol = float(df["taker_buy_base"].astype(float).iloc[-20:].sum())
    total_vol = float(volume.astype(float).iloc[-20:].sum())
    vpin = abs(taker_buy_vol - (total_vol - taker_buy_vol)) / total_vol if total_vol > 0 else 0.0

    # Volume ratio
    volume_sma_20 = float(volume.rolling(20).mean().iloc[-1])
    volume_ratio = float(volume.iloc[-1] / volume_sma_20) if volume_sma_20 > 0 else 0.0

    # Trend score
    trend_score: float = 0.0
    if ema_9 > ema_21 > ema_50:
        trend_score = 1.0
    elif ema_9 < ema_21 < ema_50:
        trend_score = -1.0

    # Regime detection
    regime: str
    if adx_val > 25 and bb_width < 0.05:
        regime = "trend"
    elif adx_val < 20:
        regime = "range"
    else:
        regime = "volatile"

    return TechnicalIndicators(
        rsi_14=rsi_14,
        rsi_21=rsi_21,
        macd=macd_val,
        macd_signal=macd_signal_val,
        macd_histogram=macd_hist,
        bb_upper=bb_upper,
        bb_middle=bb_middle,
        bb_lower=bb_lower,
        bb_width=bb_width,
        atr_14=atr_14,
        vpin=vpin,
        volume_sma_20=volume_sma_20,
        volume_ratio=volume_ratio,
        adx=adx_val,
        ema_9=ema_9,
        ema_21=ema_21,
        ema_50=ema_50,
        trend_score=trend_score,
        regime=regime,
    )