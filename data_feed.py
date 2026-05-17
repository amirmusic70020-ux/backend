"""
RadarFX — data_feed.py
Fetches OHLCV candles via yfinance for any supported currency pair / commodity.
100% free — no API key needed.
"""

import yfinance as yf
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  SUPPORTED PAIRS
#  pip_factor : multiplier to convert price distance → "pips"
#               EUR/USD, GBP/USD, USD/CHF → 10 000  (0.0001 = 1 pip)
#               USD/JPY                   →    100   (0.01   = 1 pip)
#               XAU/USD (Gold)            →     10   ($0.10  = 1 unit displayed)
#  decimals   : decimal places for price display
# ─────────────────────────────────────────────────────────────────────────────

PAIRS = {
    "EURUSD": {
        "ticker":     "EURUSD=X",
        "display":    "EUR/USD",
        "pip_factor": 10000,
        "decimals":   5,
        "pip_label":  "pips",
    },
    "GBPUSD": {
        "ticker":     "GBPUSD=X",
        "display":    "GBP/USD",
        "pip_factor": 10000,
        "decimals":   5,
        "pip_label":  "pips",
    },
    "XAUUSD": {
        "ticker":     "GC=F",
        "display":    "XAU/USD",
        "pip_factor": 10,
        "decimals":   2,
        "pip_label":  "pts",   # gold "points"
    },
    "USDCHF": {
        "ticker":     "USDCHF=X",
        "display":    "USD/CHF",
        "pip_factor": 10000,
        "decimals":   5,
        "pip_label":  "pips",
    },
    "USDJPY": {
        "ticker":     "USDJPY=X",
        "display":    "USD/JPY",
        "pip_factor": 100,
        "decimals":   3,
        "pip_label":  "pips",
    },
}

DEFAULT_PAIR = "EURUSD"

# ─────────────────────────────────────────────────────────────────────────────
#  TIMEFRAME MAP  (yfinance interval, history period)
# ─────────────────────────────────────────────────────────────────────────────

TIMEFRAME_MAP = {
    "5m":      ("5m",  "2d"),    # ~576 candles  — recent price action
    "15m":     ("15m", "6d"),    # ~576 candles
    "30m":     ("30m", "30d"),   # ~1440 candles
    "1h":      ("1h",  "60d"),   # ~1440 candles — more data for RL
    "4h":      ("1h",  "180d"),  # ~1080 candles  (resampled from 1h)
    "daily":   ("1d",  "2y"),    # ~500 candles — needs more for daily ML
    "weekly":  ("1wk", "5y"),
    "monthly": ("1mo", "10y"),
}


# ─────────────────────────────────────────────────────────────────────────────
#  CORE FETCH
# ─────────────────────────────────────────────────────────────────────────────

def fetch_candles(timeframe: str = "1h", pair: str = DEFAULT_PAIR,
                  period: str = None) -> pd.DataFrame:
    """
    Fetch OHLCV candles for any supported pair.

    Args:
        timeframe : one of TIMEFRAME_MAP keys ("15m","30m","1h","4h","daily","weekly","monthly")
        pair      : key from PAIRS dict (e.g. "EURUSD", "GBPUSD", "XAUUSD", "USDCHF", "USDJPY")
        period    : override default period (e.g. "730d", "1y", "2y")

    Returns:
        DataFrame with columns: open, high, low, close, volume
    """
    pair = pair.upper()
    if pair not in PAIRS:
        raise ValueError(f"Unsupported pair '{pair}'. Choose from: {list(PAIRS.keys())}")
    if timeframe not in TIMEFRAME_MAP:
        raise ValueError(f"Invalid timeframe '{timeframe}'. Choose from: {list(TIMEFRAME_MAP.keys())}")

    ticker_sym = PAIRS[pair]["ticker"]
    interval, default_period = TIMEFRAME_MAP[timeframe]
    period = period or default_period
    label = PAIRS[pair]["display"]

    print(f"[RadarFX] Fetching {label} {timeframe} candles...")

    try:
        ticker = yf.Ticker(ticker_sym)
        df = ticker.history(interval=interval, period=period)
    except Exception as e:
        print(f"[RadarFX] Error fetching {label}: {e}")
        return pd.DataFrame()

    if df.empty:
        print(f"[RadarFX] No data returned for {label} {timeframe}")
        return pd.DataFrame()

    # Normalise column names
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df.dropna()

    # Resample 1h → 4h
    if timeframe == "4h":
        df = df.resample("4h").agg({
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        }).dropna()

    dec = PAIRS[pair]["decimals"]
    print(f"[RadarFX] {label} {timeframe}: {len(df)} candles | "
          f"Latest: {df.index[-1]} | Close: {df['close'].iloc[-1]:.{dec}f}")
    return df


def get_multi_timeframe(timeframes: list = None, pair: str = DEFAULT_PAIR) -> dict:
    """Fetch candles for multiple timeframes for a given pair."""
    if timeframes is None:
        timeframes = ["daily", "4h", "1h", "30m", "15m"]

    results = {}
    for tf in timeframes:
        df = fetch_candles(tf, pair=pair)
        if not df.empty:
            results[tf] = df
        else:
            print(f"[RadarFX] Skipping {tf} — no data.")
    return results


def get_current_price(pair: str = DEFAULT_PAIR) -> float:
    """Get the latest price for any pair."""
    df = fetch_candles("1h", pair=pair)
    if df.empty:
        return 0.0
    dec = PAIRS[pair.upper()]["decimals"]
    return round(float(df["close"].iloc[-1]), dec)


# ─── Quick test ───────────────────────────
if __name__ == "__main__":
    for p in ["EURUSD", "GBPUSD", "XAUUSD", "USDCHF", "USDJPY"]:
        price = get_current_price(p)
        info  = PAIRS[p]
        print(f"  {info['display']:10s}  →  {price:.{info['decimals']}f}")
