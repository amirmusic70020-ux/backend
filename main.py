"""
RadarFX v3 — main.py  (ML Prediction Engine)
FastAPI server. Run with:  uvicorn main:app --reload
"""

import os
import sys
import time
import base64
import threading
import asyncio
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(__file__))

from data_feed import fetch_candles, get_current_price, PAIRS
from ml_model  import predict as ml_predict, train, MODEL_DIR, FORECAST
from predictor import draw_chart
from news_feed import get_upcoming_events, is_safe_to_trade

_pred_cache: dict = {}
CACHE_TTL = 30 * 60
_train_lock = threading.Lock()


def model_exists(pair: str, tf: str) -> bool:
    key = f"{pair}_{tf}"
    return all(
        os.path.exists(os.path.join(MODEL_DIR, f"{key}_{q}.pkl"))
        for q in ["bear", "median", "bull"]
    )


def _train_now(pair: str, tf: str):
    if _train_lock.locked():
        return
    with _train_lock:
        df = fetch_candles(tf, pair=pair)
        if df is not None and not df.empty:
            train(df, pair=f"{pair}_{tf}")
            _pred_cache.pop(f"{pair}_{tf}", None)
            print(f"  [train] done: {pair} {tf}")
        else:
            print(f"  [train] no data: {pair} {tf}")


def train_if_missing(pair: str, tf: str):
    if not model_exists(pair, tf):
        print(f"  [startup] training {pair} {tf}...")
        _train_now(pair, tf)
        print(f"  [startup] ready: {pair} {tf}")


def _build_result(pair: str, tf: str) -> dict:
    df = fetch_candles(tf, pair=pair)
    if df is None or df.empty:
        raise RuntimeError(f"No data for {pair} {tf}")

    median_p, bull_p, bear_p, last_close = ml_predict(df, pair=f"{pair}_{tf}")
    decimals   = PAIRS[pair]["decimals"]
    pip_factor = PAIRS[pair]["pip_factor"]
    pip_label  = PAIRS[pair]["pip_label"]
    median_end = float(median_p[-1])
    pct_change = (median_end / last_close - 1) * 100
    direction  = "BUY" if pct_change > 0.05 else "SELL" if pct_change < -0.05 else "WAIT"

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    draw_chart(df, median_p, bull_p, bear_p,
               last_close, pair, tf=tf, save_path=tmp_path, show=False)
    with open(tmp_path, "rb") as f:
        chart_b64 = base64.b64encode(f.read()).decode()
    os.unlink(tmp_path)

    return {
        "pair":             pair,
        "display":          PAIRS[pair]["display"],
        "tf":               tf,
        "last_close":       round(last_close, decimals),
        "forecast_candles": FORECAST,
        "median_end":       round(median_end, decimals),
        "bull_end":         round(float(bull_p[-1]), decimals),
        "bear_end":         round(float(bear_p[-1]), decimals),
        "median_pct":       round(pct_change, 3),
        "bull_pct":         round((float(bull_p[-1]) / last_close - 1) * 100, 3),
        "bear_pct":         round((float(bear_p[-1]) / last_close - 1) * 100, 3),
        "median_pips":      round(abs(median_end - last_close) * pip_factor, 1),
        "direction":        direction,
        "pip_label":        pip_label,
        "chart_b64":        chart_b64,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
    }


def get_cached_or_run(pair: str, tf: str) -> dict:
    key = f"{pair}_{tf}"
    now = time.time()
    cached = _pred_cache.get(key)
    if cached and cached["expires"] > now:
        cached["result"]["from_cache"] = True
        return cached["result"]
    result = _build_result(pair, tf)
    result["from_cache"] = False
    _pred_cache[key] = {"result": result, "expires": now + CACHE_TTL}
    return result


def background_retrain(pair: str, tf: str):
    def _do():
        _train_now(pair, tf)
    threading.Thread(target=_do, daemon=True).start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[RadarFX] Startup - checking EURUSD 1h model...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, train_if_missing, "EURUSD", "1h")
    print("[RadarFX] Ready.\n")
    yield
    print("[RadarFX] Shutdown.")


app = FastAPI(
    title="RadarFX API",
    description="ML Forex prediction - EUR/USD GBP/USD XAU/USD USD/CHF USD/JPY",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_TFS = ["5m", "15m", "30m", "1h", "4h", "daily"]


@app.get("/")
def root():
    return {
        "status": "RadarFX API v3 - ML Prediction Engine",
        "pairs":  list(PAIRS.keys()),
        "tfs":    VALID_TFS,
        "docs":   "/docs",
        "time":   datetime.utcnow().isoformat(),
    }


@app.get("/predict")
def predict_endpoint(
    pair: str = Query(default="EURUSD"),
    tf:   str = Query(default="1h"),
    retrain: bool = Query(default=False),
    background_tasks: BackgroundTasks = None,
):
    """ML price prediction - returns median/bull/bear targets + base64 chart PNG."""
    pair = pair.upper()
    if pair not in PAIRS:
        raise HTTPException(400, f"Unknown pair '{pair}'. Choose: {list(PAIRS.keys())}")
    if tf not in VALID_TFS:
        raise HTTPException(400, f"Invalid tf '{tf}'. Choose: {VALID_TFS}")

    if not model_exists(pair, tf):
        try:
            train_if_missing(pair, tf)
        except Exception as e:
            raise HTTPException(503, f"Training failed: {e}")

    try:
        result = get_cached_or_run(pair, tf)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {e}")

    if retrain and background_tasks:
        background_tasks.add_task(background_retrain, pair, tf)

    return result


@app.get("/predict/all")
def predict_all(tf: str = Query(default="1h")):
    """Predictions for all pairs on one TF. Chart omitted."""
    if tf not in VALID_TFS:
        raise HTTPException(400, f"Invalid tf '{tf}'. Choose: {VALID_TFS}")

    results = {}
    for pk in PAIRS:
        try:
            train_if_missing(pk, tf)
            res = get_cached_or_run(pk, tf)
            results[pk] = {k: v for k, v in res.items() if k != "chart_b64"}
        except Exception as e:
            results[pk] = {
                "pair": pk, "display": PAIRS[pk]["display"],
                "error": str(e), "direction": "ERROR",
            }

    return {"predictions": results, "tf": tf, "timestamp": datetime.utcnow().isoformat()}


@app.post("/retrain")
def retrain_endpoint(
    pair: str = Query(default="EURUSD"),
    tf:   str = Query(default="1h"),
):
    """Trigger background retrain. Returns immediately."""
    pair = pair.upper()
    if pair not in PAIRS:
        raise HTTPException(400, f"Unknown pair '{pair}'.")
    background_retrain(pair, tf)
    return {"status": "retrain_started", "pair": pair, "tf": tf}


@app.get("/news")
def news_endpoint(
    pair:  Optional[str] = Query(default=None),
    hours: int           = Query(default=48),
):
    """Upcoming high/medium impact economic events."""
    try:
        events = get_upcoming_events(hours_ahead=hours, pair=pair)
        return {
            "events":    events,
            "count":     len(events),
            "hours":     hours,
            "pair":      pair,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/news/check")
def news_check(
    pair:  str = Query(...),
    hours: int = Query(default=2),
):
    """Check if safe to trade - no high-impact news in next N hours."""
    pair = pair.upper()
    if pair not in PAIRS:
        raise HTTPException(400, f"Unknown pair '{pair}'.")
    return is_safe_to_trade(pair, hours_ahead=hours)


@app.get("/price")
def get_price(pair: str = Query(default="EURUSD")):
    pair = pair.upper()
    if pair not in PAIRS:
        raise HTTPException(400, f"Unknown pair '{pair}'.")
    price = get_current_price(pair)
    return {
        "pair":     PAIRS[pair]["display"],
        "pair_key": pair,
        "price":    price,
        "time":     datetime.utcnow().isoformat(),
    }


@app.get("/prices")
def get_all_prices():
    prices = {}
    for pk, info in PAIRS.items():
        try:
            price = get_current_price(pk)
        except Exception:
            price = None
        prices[pk] = {"pair": info["display"], "price": price}
    return {"prices": prices, "time": datetime.utcnow().isoformat()}


@app.get("/pairs")
def list_pairs():
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
    models = {pk: model_exists(pk, "1h") for pk in PAIRS}
    return {
        "status":     "ok",
        "version":    "3.0.0",
        "models_1h":  models,
        "cache_keys": list(_pred_cache.keys()),
        "time":       datetime.utcnow().isoformat(),
    }
