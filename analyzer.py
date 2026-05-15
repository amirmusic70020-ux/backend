"""
RadarFX — analyzer.py
Technical analysis engine.
Calculates indicators and returns a structured result for each timeframe.
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  INDICATOR CALCULATIONS (no external library needed — all pure pandas/numpy)
# ─────────────────────────────────────────────────────────────────────────────

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_macd(series: pd.Series, fast=12, slow=26, signal=9) -> dict:
    ema_fast   = calc_ema(series, fast)
    ema_slow   = calc_ema(series, slow)
    macd_line  = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram  = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "hist": histogram}


def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)

    dm_plus  = np.where((high - high.shift()) > (low.shift() - low),
                        np.maximum(high - high.shift(), 0), 0)
    dm_minus = np.where((low.shift() - low) > (high - high.shift()),
                        np.maximum(low.shift() - low, 0), 0)

    dm_plus  = pd.Series(dm_plus,  index=df.index)
    dm_minus = pd.Series(dm_minus, index=df.index)

    atr      = tr.ewm(alpha=1/period, adjust=False).mean()
    di_plus  = 100 * dm_plus.ewm(alpha=1/period, adjust=False).mean()  / atr
    di_minus = 100 * dm_minus.ewm(alpha=1/period, adjust=False).mean() / atr

    dx  = (100 * (di_plus - di_minus).abs()) / (di_plus + di_minus).replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx


def calc_bollinger(series: pd.Series, period: int = 20, std: float = 2.0) -> dict:
    mid   = series.rolling(period).mean()
    sigma = series.rolling(period).std()
    upper = mid + std * sigma
    lower = mid - std * sigma
    width = (upper - lower) / mid
    return {"upper": upper, "mid": mid, "lower": lower, "width": width}


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


# ─────────────────────────────────────────────────────────────────────────────
#  PATTERN DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_pattern(df: pd.DataFrame) -> str:
    """Detect the most recent candlestick pattern on last 3 candles."""
    if len(df) < 3:
        return "none"

    c = df.iloc[-1]   # current candle
    p = df.iloc[-2]   # previous candle

    body_c = abs(c["close"] - c["open"])
    body_p = abs(p["close"] - p["open"])

    # Bullish Engulfing
    if (c["close"] > c["open"] and p["close"] < p["open"]
            and c["open"] < p["close"] and c["close"] > p["open"]
            and body_c > body_p):
        return "bullish_engulfing"

    # Bearish Engulfing
    if (c["close"] < c["open"] and p["close"] > p["open"]
            and c["open"] > p["close"] and c["close"] < p["open"]
            and body_c > body_p):
        return "bearish_engulfing"

    # Bullish Pin Bar (hammer)
    total_range = c["high"] - c["low"]
    lower_wick  = min(c["open"], c["close"]) - c["low"]
    upper_wick  = c["high"] - max(c["open"], c["close"])
    if total_range > 0:
        if lower_wick / total_range > 0.6 and c["close"] > c["open"]:
            return "bullish_pin_bar"
        if upper_wick / total_range > 0.6 and c["close"] < c["open"]:
            return "bearish_pin_bar"

    # Doji
    if total_range > 0 and body_c / total_range < 0.1:
        return "doji"

    return "none"


# ─────────────────────────────────────────────────────────────────────────────
#  TREND DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_trend(df: pd.DataFrame) -> str:
    """
    Determine trend direction using EMA 50 vs EMA 200.
    Returns: 'bullish', 'bearish', or 'neutral'
    """
    if len(df) < 50:
        return "neutral"

    close     = df["close"]
    ema50     = calc_ema(close, 50).iloc[-1]
    ema200    = calc_ema(close, 200).iloc[-1] if len(df) >= 200 else calc_ema(close, 50).iloc[-1]
    last_close = close.iloc[-1]

    if last_close > ema50 and ema50 > ema200:
        return "bullish"
    elif last_close < ema50 and ema50 < ema200:
        return "bearish"
    else:
        return "neutral"


# ─────────────────────────────────────────────────────────────────────────────
#  RANGE DETECTION  ← "بازار رنجه، وارد نشو"
# ─────────────────────────────────────────────────────────────────────────────

def is_ranging(df: pd.DataFrame) -> bool:
    """
    Returns True if the market is in a range (choppy, no clear trend).
    Conditions:
      - ADX < 25  (weak trend)
      - Bollinger Band width in bottom 30% of recent history (squeezed)
    """
    if len(df) < 25:
        return False

    adx = calc_adx(df).iloc[-1]
    bb  = calc_bollinger(df["close"])
    bb_width     = bb["width"].iloc[-1]
    bb_width_avg = bb["width"].rolling(50).mean().iloc[-1]

    adx_weak     = adx < 25
    bb_squeezed  = bb_width < bb_width_avg * 0.75 if not pd.isna(bb_width_avg) else False

    return adx_weak and bb_squeezed


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ANALYSIS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def analyze(df: pd.DataFrame, timeframe: str = "1h") -> dict:
    """
    Run full technical analysis on a candle DataFrame.

    Returns:
        dict with all indicators and a bias ('bullish' | 'bearish' | 'neutral' | 'ranging')
    """
    if df is None or len(df) < 20:
        return {"timeframe": timeframe, "bias": "neutral", "error": "Not enough data"}

    close  = df["close"]
    result = {"timeframe": timeframe}

    # ── RSI ──────────────────────────────
    rsi = calc_rsi(close).iloc[-1]
    result["rsi"] = round(rsi, 2)
    result["rsi_signal"] = (
        "overbought" if rsi > 70 else
        "oversold"   if rsi < 30 else
        "neutral"
    )

    # ── MACD ─────────────────────────────
    macd = calc_macd(close)
    macd_val  = macd["macd"].iloc[-1]
    macd_sig  = macd["signal"].iloc[-1]
    macd_hist = macd["hist"].iloc[-1]
    result["macd"]         = round(macd_val, 6)
    result["macd_signal"]  = round(macd_sig, 6)
    result["macd_hist"]    = round(macd_hist, 6)
    result["macd_cross"]   = "bullish" if macd_val > macd_sig else "bearish"

    # ── EMA ──────────────────────────────
    ema20  = calc_ema(close, 20).iloc[-1]
    ema50  = calc_ema(close, 50).iloc[-1]
    ema200 = calc_ema(close, 200).iloc[-1] if len(df) >= 200 else ema50
    result["ema20"]  = round(ema20, 5)
    result["ema50"]  = round(ema50, 5)
    result["ema200"] = round(ema200, 5)

    # ── ADX ──────────────────────────────
    adx = calc_adx(df).iloc[-1]
    result["adx"] = round(adx, 2)
    result["adx_strength"] = (
        "strong trend" if adx > 40 else
        "moderate trend" if adx > 25 else
        "weak / ranging"
    )

    # ── Bollinger Bands ───────────────────
    bb = calc_bollinger(close)
    result["bb_upper"] = round(bb["upper"].iloc[-1], 5)
    result["bb_mid"]   = round(bb["mid"].iloc[-1],   5)
    result["bb_lower"] = round(bb["lower"].iloc[-1], 5)
    result["bb_width"] = round(bb["width"].iloc[-1], 5)

    # ── ATR ───────────────────────────────
    atr = calc_atr(df).iloc[-1]
    result["atr"] = round(atr, 5)

    # ── Trend ─────────────────────────────
    result["trend"] = detect_trend(df)

    # ── Pattern ───────────────────────────
    result["pattern"] = detect_pattern(df)

    # ── Support / Resistance ──────────────
    recent = df.tail(50)
    result["support"]    = round(recent["low"].min(),  5)
    result["resistance"] = round(recent["high"].max(), 5)

    # ── Range check ───────────────────────
    result["is_ranging"] = is_ranging(df)

    # ── Overall bias ──────────────────────
    if result["is_ranging"]:
        result["bias"] = "ranging"
    else:
        bull_signals = sum([
            result["trend"] == "bullish",
            result["macd_cross"] == "bullish",
            result["rsi_signal"] == "oversold",
            result["pattern"] in ("bullish_engulfing", "bullish_pin_bar"),
            close.iloc[-1] > ema50,
        ])
        bear_signals = sum([
            result["trend"] == "bearish",
            result["macd_cross"] == "bearish",
            result["rsi_signal"] == "overbought",
            result["pattern"] in ("bearish_engulfing", "bearish_pin_bar"),
            close.iloc[-1] < ema50,
        ])
        if bull_signals >= 3:
            result["bias"] = "bullish"
        elif bear_signals >= 3:
            result["bias"] = "bearish"
        else:
            result["bias"] = "neutral"

    return result


def analyze_multi(timeframe_data: dict) -> dict:
    """
    Analyze multiple timeframes and return all results.

    Args:
        timeframe_data: dict of { timeframe_str: DataFrame }

    Returns:
        dict of { timeframe_str: analysis_result }
    """
    results = {}
    for tf, df in timeframe_data.items():
        results[tf] = analyze(df, tf)
    return results


# ─── Quick test ───────────────────────────
if __name__ == "__main__":
    from data_feed import fetch_candles
    df = fetch_candles("1h", outputsize="compact")
    if not df.empty:
        result = analyze(df, "1h")
        print("\n── Analysis Result ──────────────────")
        for k, v in result.items():
            print(f"  {k:15s}: {v}")
