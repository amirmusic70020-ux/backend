"""
RadarFX ML — train.py
Trains Gradient Boosting models for price prediction.

Usage:
  python train.py                        # EURUSD, all timeframes
  python train.py EURUSD                 # one pair, all timeframes
  python train.py EURUSD --tf 1h         # one pair, one timeframe
  python train.py ALL --tf 4h            # all pairs, one timeframe
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from data_feed import fetch_candles, PAIRS
from ml_model  import train, add_features, make_sequences, LOOKBACK

# ─── تایم‌فریم‌ها و تعداد کندل پیش‌بینی برای هر کدام ──────────────────────
TF_CONFIG = {
    "5m":    {"forecast": 20, "label": "5 دقیقه  → پیش‌بینی ۱h۴۰m"},
    "15m":   {"forecast": 20, "label": "15 دقیقه → پیش‌بینی ۵h"},
    "30m":   {"forecast": 20, "label": "30 دقیقه → پیش‌بینی ۱۰h"},
    "1h":    {"forecast": 20, "label": "۱ ساعته  → پیش‌بینی ۲۰h"},
    "4h":    {"forecast": 20, "label": "۴ ساعته  → پیش‌بینی ۳.۳ روز"},
    "daily": {"forecast": 20, "label": "روزانه   → پیش‌بینی ۲۰ روز"},
}

ALL_TF = list(TF_CONFIG.keys())


def train_pair_tf(pair_key: str, tf: str) -> bool:
    pair_key = pair_key.upper()
    if pair_key not in PAIRS:
        print(f"  ✗ Unknown pair '{pair_key}'")
        return False
    if tf not in TF_CONFIG:
        print(f"  ✗ Unknown timeframe '{tf}'. Available: {ALL_TF}")
        return False

    display = PAIRS[pair_key]["display"]
    config  = TF_CONFIG[tf]

    print(f"\n  {'─'*50}")
    print(f"  {display} ({pair_key})  |  {config['label']}")
    print(f"  {'─'*50}")

    # Fetch data
    df = fetch_candles(tf, pair=pair_key)
    if df is None or df.empty:
        print(f"  ✗ No data")
        return False

    print(f"  Candles: {len(df)}  ({df.index[0].date()} → {df.index[-1].date()})")

    # Check sequences
    df_check = add_features(df)
    X, Y = make_sequences(df_check)

    if len(X) < 50:
        print(f"  ✗ Not enough sequences ({len(X)}). Need ≥50.")
        return False

    print(f"  Sequences: {len(X)}")

    t0 = time.time()
    train(df, pair=f"{pair_key}_{tf}")
    elapsed = time.time() - t0

    print(f"  ✓ Done in {elapsed:.0f}s")
    return True


def main():
    p = argparse.ArgumentParser(description="RadarFX ML Trainer")
    p.add_argument("pair", nargs="?", default="EURUSD",
                   help="Pair key (e.g. EURUSD) or ALL")
    p.add_argument("--tf", type=str, default="1h",
                   help=f"Timeframe: {ALL_TF} or ALL")
    args = p.parse_args()

    target_pair = args.pair.upper()
    target_tf   = args.tf.lower()

    pairs_list = list(PAIRS.keys()) if target_pair == "ALL" else [target_pair]
    tf_list    = ALL_TF             if target_tf   == "all" else [target_tf]

    print(f"\n{'═'*55}")
    print(f"  RadarFX ML Trainer")
    print(f"  Pairs      : {pairs_list}")
    print(f"  Timeframes : {tf_list}")
    print(f"{'═'*55}")

    results = {}
    for pair in pairs_list:
        for tf in tf_list:
            key = f"{pair}_{tf}"
            ok  = train_pair_tf(pair, tf)
            results[key] = "✓" if ok else "✗"

    print(f"\n{'═'*55}")
    print("  Results:")
    for key, status in results.items():
        print(f"    {key:15s}  {status}")
    print(f"{'═'*55}")
    print(f"\n  Next: python predictor.py EURUSD --tf 1h\n")


if __name__ == "__main__":
    main()
