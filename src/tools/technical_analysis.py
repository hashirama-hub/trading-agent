from typing import Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor

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


class MultiTimeframeIndicators(BaseModel):
    """Aggregated indicators across multiple timeframes."""
    current: TechnicalIndicators
    htf_trend: str  # Higher timeframe trend direction
    htf_regime: str  # Higher timeframe regime
    signal_alignment: float  # -1 to +1, how aligned timeframes are
    summary: str  # Human-readable summary for LLM


def _compute_indicators_from_df(df: pd.DataFrame) -> TechnicalIndicators:
    """Compute technical indicators from a DataFrame with OHLCV columns."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    rsi_14 = float(RSIIndicator(close, window=14).rsi().iloc[-1])
    rsi_21 = float(RSIIndicator(close, window=21).rsi().iloc[-1])

    macd_indicator = MACD(close)
    macd_val = float(macd_indicator.macd().iloc[-1])
    macd_signal_val = float(macd_indicator.macd_signal().iloc[-1])
    macd_hist = float(macd_indicator.macd_diff().iloc[-1])

    bb = BollingerBands(close)
    bb_upper = float(bb.bollinger_hband().iloc[-1])
    bb_middle = float(bb.bollinger_mavg().iloc[-1])
    bb_lower = float(bb.bollinger_lband().iloc[-1])
    bb_width = float((bb_upper - bb_lower) / bb_middle) if bb_middle > 0 else 0.0

    atr_14 = float(AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1])
    adx_val = float(ADXIndicator(high, low, close, window=14).adx().iloc[-1])

    ema_9 = float(EMAIndicator(close, window=9).ema_indicator().iloc[-1])
    ema_21 = float(EMAIndicator(close, window=21).ema_indicator().iloc[-1])
    ema_50 = float(EMAIndicator(close, window=50).ema_indicator().iloc[-1])

    taker_buy_vol = float(df["taker_buy_base"].astype(float).iloc[-20:].sum())
    total_vol = float(volume.astype(float).iloc[-20:].sum())
    vpin = abs(taker_buy_vol - (total_vol - taker_buy_vol)) / total_vol if total_vol > 0 else 0.0

    volume_sma_20 = float(volume.rolling(20).mean().iloc[-1])
    volume_ratio = float(volume.iloc[-1] / volume_sma_20) if volume_sma_20 > 0 else 0.0

    trend_score: float = 0.0
    if ema_9 > ema_21 > ema_50:
        trend_score = 1.0
    elif ema_9 < ema_21 < ema_50:
        trend_score = -1.0

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

    return _compute_indicators_from_df(df)


async def calculate_multi_timeframe_technicals(
    symbol: str,
    timeframes: List[str] = None,
) -> MultiTimeframeIndicators:
    """Compute technical indicators across multiple timeframes.

    Args:
        symbol: Trading pair
        timeframes: List of intervals (default: ["15m", "1h", "4h"])

    Returns:
        MultiTimeframeIndicators with aggregated analysis
    """
    if timeframes is None:
        timeframes = ["15m", "1h", "4h"]

    indicators_by_tf = {}
    for tf in timeframes:
        try:
            klines = await binance_client.get_klines(symbol=symbol, interval=tf, limit=100)
            df = pd.DataFrame(klines, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore",
            ])
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            indicators_by_tf[tf] = _compute_indicators_from_df(df)
        except Exception:
            continue

    if not indicators_by_tf:
        raise ValueError(f"Failed to compute indicators for {symbol}")

    current = indicators_by_tf.get("1h") or list(indicators_by_tf.values())[-1]

    htf = indicators_by_tf.get("4h") or indicators_by_tf.get("1h")
    ltf = indicators_by_tf.get("15m") or indicators_by_tf.get("5m")

    htf_trend = "bullish" if htf.trend_score > 0 else "bearish" if htf.trend_score < 0 else "neutral"
    htf_regime = htf.regime

    trend_scores = [ind.trend_score for ind in indicators_by_tf.values()]
    signal_alignment = float(np.mean(trend_scores)) if trend_scores else 0.0

    alignment_desc = "strong" if abs(signal_alignment) > 0.7 else "moderate" if abs(signal_alignment) > 0.3 else "weak"
    alignment_dir = "bullish" if signal_alignment > 0.3 else "bearish" if signal_alignment < -0.3 else "mixed"

    summary = (
        f"4H: {htf_regime} {htf_trend} (ADX={htf.adx:.1f}, RSI={htf.rsi_14:.1f}) | "
        f"1H: {current.regime} (ADX={current.adx:.1f}, RSI={current.rsi_14:.1f}, MACD_hist={current.macd_histogram:.2f}) | "
        f"15M: RSI={ltf.rsi_14:.1f} if ltf else 'N/A' | "
        f"Alignment: {alignment_desc} {alignment_dir} ({signal_alignment:.2f})"
    )

    return MultiTimeframeIndicators(
        current=current,
        htf_trend=htf_trend,
        htf_regime=htf_regime,
        signal_alignment=signal_alignment,
        summary=summary,
    )
