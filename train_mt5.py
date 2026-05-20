"""
RadarFX — train_mt5.py
Train ML models using 2 years of live data directly from MT5.
Models saved with _v2 suffix — old models untouched.

Usage (on VPS with MT5 running):
  python train_mt5.py              # all pairs, 1h only
  python train_mt5.py --tf 1h      # all pairs, specific TF
  python train_mt5.py --pair EURUSD
  python train_mt5.py --all-tf     # all pairs, all timeframes (slow)
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from mt5_bridge import connect, disconnect, get_candles
from ml_model   import train, add_features, make_sequences
from data_feed  import PAIRS

# ─── Config ───────────────────────────────────────────────────────────────────

# Candles to fetch per timeframe (~3 months)
TF_CANDLES = {
    "1h":     2_200,   # 3 months x ~24h trading
    "4h":       550,   # 3 months x ~6
    "daily":     65,   # 3 months
    "15m":    8_500,
    "30m":    4_300,
    "5m":    25_000,
}

DEFAULT_TF   = ["1h"]
ALL_TF       = ["1h", "4h", "daily"]   # skip 5m/15m/30m unless asked
MODEL_SUFFIX = "_v2"


# ─── Train one pair/tf ────────────────────────────────────────────────────────

def train_one(pair: str, tf: str) -> bool:
    n = TF_CANDLES.get(tf, 17_500)
    display = PAIRS[pair]["display"]

    print(f"\n  {'-'*52}")
    print(f"  {display}  |  {tf}  |  fetching {n:,} candles from MT5...")
    print(f"  {'-'*52}")

    df = get_candles(pair, tf, n=n)
    if df is None or df.empty:
        print(f"  [X] No data from MT5 for {pair} {tf}")
        return False

    print(f"  Candles : {len(df):,}  ({df.index[0].date()} to {df.index[-1].date()})")

    df_feat = add_features(df)
    X, Y    = make_sequences(df_feat)

    if len(X) < 100:
        print(f"  [X] Not enough sequences ({len(X)}). Skipping.")
        return False

    print(f"  Sequences: {len(X):,}  -- training...")

    t0      = time.time()
    pair_key = f"{pair}_{tf}{MODEL_SUFFIX}"
    train(df, pair=pair_key)
    elapsed = time.time() - t0

    print(f"  [OK] Done in {elapsed:.0f}s  ->  models/{pair_key}_*.pkl")
    return True


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RadarFX MT5 Trainer (v2)")
    parser.add_argument("--pair", type=str, default=None,
                        help="Single pair e.g. EURUSD (default: all)")
    parser.add_argument("--tf",   type=str, default="1h",
                        help="Timeframe: 1h 4h daily (default: 1h)")
    parser.add_argument("--all-tf", action="store_true",
                        help="Train all timeframes (1h, 4h, daily)")
    args = parser.parse_args()

    pairs   = [args.pair.upper()] if args.pair else list(PAIRS.keys())
    tf_list = ALL_TF if args.all_tf else [args.tf]

    print(f"\n{'='*55}")
    print(f"  RadarFX MT5 Trainer  --  v2 Models")
    print(f"  Pairs     : {pairs}")
    print(f"  Timeframes: {tf_list}")
    print(f"  Data      : ~3 months from MT5")
    print(f"  Suffix    : {MODEL_SUFFIX}")
    print(f"{'='*55}")

    if not connect():
        print("\n✗ Cannot connect to MT5. Make sure terminal is open.")
        return

    results = {}
    for pair in pairs:
        for tf in tf_list:
            key = f"{pair}_{tf}{MODEL_SUFFIX}"
            ok  = train_one(pair, tf)
            results[key] = "✓" if ok else "✗"

    disconnect()

    print(f"\n{'='*55}")
    print("  Results:")
    for key, status in results.items():
        print(f"    {key:25s}  {status}")
    print(f"{'='*55}")
    print(f"\n  Next: python backtest_compare.py")
    print(f"  If v2 is better: python switch_to_v2.py\n")


if __name__ == "__main__":
    main()
