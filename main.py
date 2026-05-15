"""
RadarFX — main.py
FastAPI server. Run with:  uvicorn main:app --reload

Endpoints:
  GET /signal?pair=EURUSD      → signal for one pair (default: EURUSD)
  GET /signal/all              → signals for all 5 pairs
  GET /price?pair=EURUSD       → current price for one pair
  GET /prices                  → current prices for all 5 pairs
  GET /health                  → health check
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import Optional

from data_feed     import get_multi_timeframe, get_current_price, PAIRS
from signal_engine import generate_signal, format_signal

app = FastAPI(
    title       = "RadarFX API",
    description = "AI-powered Forex signal engine — EUR/USD, GBP/USD, XAU/USD, USD/CHF, USD/JPY",
    version     = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

TIMEFRAMES = ["daily", "4h", "1h", "30m", "15m"]


# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status":  "RadarFX API v2 is running",
        "pairs":   list(PAIRS.keys()),
        "time":    datetime.utcnow().isoformat(),
    }


@app.get("/signal")
def get_signal(pair: Optional[str] = Query(default="EURUSD", description="Pair key: EURUSD, GBPUSD, XAUUSD, USDCHF, USDJPY")):
    """
    Returns the current signal for a single pair.
    Example: /signal?pair=GBPUSD
    """
    pair = pair.upper()
    if pair not in PAIRS:
        raise HTTPException(status_code=400, detail=f"Unknown pair '{pair}'. Available: {list(PAIRS.keys())}")

    tf_data = get_multi_timeframe(TIMEFRAMES, pair=pair)
    if not tf_data:
        raise HTTPException(status_code=503, detail="Could not fetch market data.")

    signal = generate_signal(tf_data, pair=pair)
    signal.pop("analyses", None)
    return signal


@app.get("/signal/all")
def get_all_signals():
    """
    Returns signals for all 5 supported pairs at once.
    Useful for the dashboard overview panel.
    """
    results = {}
    for pair_key in PAIRS:
        try:
            tf_data = get_multi_timeframe(TIMEFRAMES, pair=pair_key)
            if tf_data:
                signal = generate_signal(tf_data, pair=pair_key)
                signal.pop("analyses", None)
            else:
                signal = {
                    "signal":    "ERROR",
                    "pair":      PAIRS[pair_key]["display"],
                    "reason":    "No data returned",
                    "timestamp": datetime.utcnow().isoformat(),
                }
        except Exception as e:
            signal = {
                "signal":    "ERROR",
                "pair":      PAIRS[pair_key]["display"],
                "reason":    str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
        results[pair_key] = signal

    return {
        "signals":   results,
        "timestamp": datetime.utcnow().isoformat(),
        "count":     len(results),
    }


@app.get("/price")
def get_price(pair: Optional[str] = Query(default="EURUSD")):
    """Returns the latest price for a single pair."""
    pair = pair.upper()
    if pair not in PAIRS:
        raise HTTPException(status_code=400, detail=f"Unknown pair '{pair}'.")

    price = get_current_price(pair)
    info  = PAIRS[pair]
    return {
        "pair":     info["display"],
        "pair_key": pair,
        "price":    price,
        "time":     datetime.utcnow().isoformat(),
    }


@app.get("/prices")
def get_all_prices():
    """Returns the latest prices for all 5 pairs."""
    prices = {}
    for pair_key, info in PAIRS.items():
        try:
            price = get_current_price(pair_key)
        except Exception:
            price = None
        prices[pair_key] = {
            "pair":  info["display"],
            "price": price,
        }
    return {"prices": prices, "time": datetime.utcnow().isoformat()}


@app.get("/pairs")
def list_pairs():
    """Returns metadata for all supported pairs."""
    return {
        k: {
            "display":   v["display"],
            "pip_label": v["pip_label"],
            "decimals":  v["decimals"],
        }
        for k, v in PAIRS.items()
    }


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0", "time": datetime.utcnow().isoformat()}
