"""
RadarFX — switch_to_v2.py
Switch to v2 models (or delete them if v1 was better).

Usage:
  python switch_to_v2.py            # rename v2 → v1 (backup old as _v1_old)
  python switch_to_v2.py --delete   # delete v2 files, keep v1
"""

import os
import sys
import argparse
import shutil

sys.path.insert(0, os.path.dirname(__file__))
from data_feed import PAIRS

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
TF_LIST   = ["1h", "4h", "daily"]
QUANTILES = ["bear", "median", "bull"]


def switch_to_v2():
    print("\n  Switching to v2 models...\n")
    switched = 0
    for pair in PAIRS:
        for tf in TF_LIST:
            for q in QUANTILES:
                v2_path  = os.path.join(MODEL_DIR, f"{pair}_{tf}_v2_{q}.pkl")
                v1_path  = os.path.join(MODEL_DIR, f"{pair}_{tf}_{q}.pkl")
                bak_path = os.path.join(MODEL_DIR, f"{pair}_{tf}_{q}_v1_old.pkl")

                if not os.path.exists(v2_path):
                    continue

                # Backup old v1
                if os.path.exists(v1_path):
                    shutil.copy2(v1_path, bak_path)

                # Replace v1 with v2
                shutil.copy2(v2_path, v1_path)
                os.remove(v2_path)
                switched += 1

    print(f"  ✅ Switched {switched} model files to v2.")
    print(f"  Old v1 models backed up as *_v1_old.pkl")
    print(f"\n  Restart live_trader.py to use new models.\n")


def delete_v2():
    print("\n  Deleting v2 models (keeping v1)...\n")
    deleted = 0
    for pair in PAIRS:
        for tf in TF_LIST:
            for q in QUANTILES:
                v2_path = os.path.join(MODEL_DIR, f"{pair}_{tf}_v2_{q}.pkl")
                if os.path.exists(v2_path):
                    os.remove(v2_path)
                    deleted += 1

    print(f"  ✅ Deleted {deleted} v2 model files.")
    print(f"  Current v1 models unchanged.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true",
                        help="Delete v2 models instead of switching")
    args = parser.parse_args()

    if args.delete:
        delete_v2()
    else:
        switch_to_v2()


if __name__ == "__main__":
    main()
