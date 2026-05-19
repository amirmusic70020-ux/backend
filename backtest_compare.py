"""
RadarFX — backtest_compare.py
Compare v1 (current) vs v2 (2-year trained) models side by side.
Uses last 3 months of MT5 data as test set.

Usage:
  python backtest_compare.py              # all pairs, 1h
  python backtest_compare.py --pair EURUSD
  python backtest_compare.py --tf 4h

At the end shows:
  ✅ v2 better  → run: python switch_to_v2.py
  ❌ v1 better  → keep current models, delete v2
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from mt5_bridge import connect, disconnect, get_candles
from ml_model   import add_features, predict as ml_predict, LOOKBACK
from data_feed  import PAIRS

MODEL_DIR   = os.path.join(os.path.dirname(__file__), "models")
TEST_CANDLES = 2_200    # ~3 months of 1h data for test
MIN_MOVE_PIPS = 10
MIN_RR        = 1.0


# ─── Single backtest run ──────────────────────────────────────────────────────

def run_backtest(pair: str, tf: str, model_suffix: str = "") -> dict:
    """
    Simulate trading on last TEST_CANDLES candles.
    model_suffix: "" for v1, "_v2" for v2
    """
    pip_factor = PAIRS[pair]["pip_factor"]
    dec        = PAIRS[pair]["decimals"]
    label      = f"{pair}_{tf}{model_suffix}"

    # Check model exists
    for q in ["bear", "median", "bull"]:
        path = os.path.join(MODEL_DIR, f"{label}_{q}.pkl")
        if not os.path.exists(path):
            return {"error": f"Model not found: {label}_{q}.pkl"}

    # Fetch test data (3 months + enough history for features)
    n_fetch = TEST_CANDLES + LOOKBACK + 100
    df = get_candles(pair, tf, n=n_fetch)
    if df is None or df.empty:
        return {"error": "No data from MT5"}

    # Use only last TEST_CANDLES as test window
    df_test = df.tail(TEST_CANDLES).copy()

    trades = []
    wins = 0
    total_pips = 0.0

    for i in range(LOOKBACK + 50, len(df_test) - 20):
        # Use history up to this bar
        df_window = df.iloc[: -(len(df_test) - i)].copy() if len(df_test) - i > 0 else df.copy()
        if len(df_window) < LOOKBACK + 50:
            continue

        try:
            median_prices, bull_prices, bear_prices, last_close = ml_predict(
                df_window, label
            )
        except Exception:
            continue

        median_end  = float(median_prices[-1])
        bull_end    = float(bull_prices[-1])
        bear_end    = float(bear_prices[-1])
        median_move = (median_end - last_close) * pip_factor

        MIN_SL_PIPS = 8

        if median_move > MIN_MOVE_PIPS:
            signal = "BUY"
            tp     = round(bull_end, dec)
            sl_raw = round(bear_end, dec)
            sl     = sl_raw if sl_raw < last_close else round(last_close - MIN_SL_PIPS / pip_factor, dec)
        elif median_move < -MIN_MOVE_PIPS:
            signal = "SELL"
            tp     = round(bear_end, dec)
            sl_raw = round(bull_end, dec)
            sl     = sl_raw if sl_raw > last_close else round(last_close + MIN_SL_PIPS / pip_factor, dec)
        else:
            continue

        sl_pips = abs(last_close - sl) * pip_factor
        tp_pips = abs(last_close - tp) * pip_factor
        rr      = tp_pips / sl_pips if sl_pips > 0 else 0

        if rr < MIN_RR:
            continue

        # Simulate: check next 20 candles for TP/SL hit
        future = df_test.iloc[i: i + 20]
        result = "OPEN"
        pnl_pips = 0.0

        for _, bar in future.iterrows():
            if signal == "BUY":
                if bar["low"] <= sl:
                    result   = "LOSS"
                    pnl_pips = -sl_pips
                    break
                if bar["high"] >= tp:
                    result   = "WIN"
                    pnl_pips = tp_pips
                    break
            else:
                if bar["high"] >= sl:
                    result   = "LOSS"
                    pnl_pips = -sl_pips
                    break
                if bar["low"] <= tp:
                    result   = "WIN"
                    pnl_pips = tp_pips
                    break

        if result == "OPEN":
            continue   # skip unresolved trades

        trades.append({
            "signal":   signal,
            "result":   result,
            "pnl_pips": round(pnl_pips, 1),
            "rr":       round(rr, 2),
        })

        if result == "WIN":
            wins += 1
        total_pips += pnl_pips

    if not trades:
        return {"error": "No trades generated"}

    n          = len(trades)
    win_rate   = wins / n * 100
    avg_pips   = total_pips / n

    # Max drawdown
    equity = np.cumsum([t["pnl_pips"] for t in trades])
    peak   = np.maximum.accumulate(equity)
    dd     = equity - peak
    max_dd = float(dd.min())

    return {
        "trades":     n,
        "win_rate":   round(win_rate, 1),
        "total_pips": round(total_pips, 1),
        "avg_pips":   round(avg_pips, 1),
        "max_dd":     round(max_dd, 1),
        "error":      None,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RadarFX Model Comparator")
    parser.add_argument("--pair", type=str, default=None)
    parser.add_argument("--tf",   type=str, default="1h")
    args = parser.parse_args()

    pairs = [args.pair.upper()] if args.pair else list(PAIRS.keys())
    tf    = args.tf

    print(f"\n{'═'*65}")
    print(f"  RadarFX Backtest Compare  —  v1 vs v2  |  TF: {tf}")
    print(f"  Test window: last ~3 months of MT5 data")
    print(f"{'═'*65}")

    if not connect():
        print("✗ Cannot connect to MT5.")
        return

    v2_wins = 0
    v1_wins = 0

    for pair in pairs:
        display = PAIRS[pair]["display"]
        print(f"\n  {display}")

        r1 = run_backtest(pair, tf, model_suffix="")
        r2 = run_backtest(pair, tf, model_suffix="_v2")

        if r1.get("error"):
            print(f"    v1 : ✗ {r1['error']}")
        else:
            print(f"    v1 : {r1['trades']} trades | "
                  f"WR {r1['win_rate']}% | "
                  f"Pips {r1['total_pips']:+.0f} | "
                  f"DD {r1['max_dd']:.0f}")

        if r2.get("error"):
            print(f"    v2 : ✗ {r2['error']}")
        else:
            print(f"    v2 : {r2['trades']} trades | "
                  f"WR {r2['win_rate']}% | "
                  f"Pips {r2['total_pips']:+.0f} | "
                  f"DD {r2['max_dd']:.0f}")

        # Compare
        if not r1.get("error") and not r2.get("error"):
            score1 = r1["total_pips"] + r1["win_rate"] * 10
            score2 = r2["total_pips"] + r2["win_rate"] * 10
            if score2 > score1:
                print(f"    → ✅ v2 BETTER  (+{score2-score1:.0f} score)")
                v2_wins += 1
            else:
                print(f"    → ❌ v1 still better")
                v1_wins += 1

    disconnect()

    print(f"\n{'═'*65}")
    print(f"  Summary: v2 better in {v2_wins}/{len(pairs)} pairs")
    if v2_wins > v1_wins:
        print(f"\n  ✅ RECOMMENDATION: Switch to v2")
        print(f"  Run: python switch_to_v2.py")
    else:
        print(f"\n  ❌ Keep current v1 models — v2 not better")
        print(f"  Run: python switch_to_v2.py --delete  (to remove v2 files)")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    main()
