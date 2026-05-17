"""
RadarFX — rl_train.py
Train DQN trading agents for any pair / timeframe.
Same CLI interface as train.py.

Usage
-----
    python rl_train.py EURUSD --tf 1h
    python rl_train.py EURUSD --tf all
    python rl_train.py all --tf 1h
    python rl_train.py all --tf all        # train everything (~30-60 min)
"""

import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from data_feed  import fetch_candles, PAIRS
from ml_model   import add_features
from rl_env     import ForexTradingEnv
from rl_agent   import DQNAgent

MODEL_DIR  = os.path.join(os.path.dirname(__file__), "models")
VALID_TFS  = ["5m", "15m", "30m", "1h", "4h", "daily"]
N_EPISODES  = 800       # episodes per pair/tf
TRAIN_START = 128      # min buffer size before learning begins
TRAIN_SPLIT = 0.80     # fraction of bars used for training


# ─── Train one pair × timeframe ───────────────────────────────────────────────

def train_one(pair: str, tf: str) -> bool:
    label = PAIRS[pair]["display"]
    print(f"\n{'═'*58}")
    print(f"  {label} | {tf}  →  DQN Reinforcement Learning")
    print(f"{'═'*58}")

    # ── Data ──────────────────────────────────────────────────
    df_raw = fetch_candles(tf, pair=pair)
    if df_raw is None or df_raw.empty:
        print("  ✗ No data — skipping")
        return False

    df = add_features(df_raw)
    if len(df) < 150:
        print(f"  ✗ Only {len(df)} rows after feature engineering (need ≥ 150)")
        return False

    split      = int(len(df) * TRAIN_SPLIT)
    df_train   = df.iloc[:split].reset_index(drop=True)
    df_val     = df.iloc[split:].reset_index(drop=True)

    pip_factor   = PAIRS[pair]["pip_factor"]
    spread_pips  = 5.0 if pair == "XAUUSD" else 2.0   # gold has wider spread

    train_env = ForexTradingEnv(df_train, pip_factor=pip_factor, spread_pips=spread_pips)
    val_env   = ForexTradingEnv(df_val,   pip_factor=pip_factor, spread_pips=spread_pips)

    agent = DQNAgent(state_dim=ForexTradingEnv.STATE_DIM)

    print(f"  Train bars : {len(df_train)} | Val bars: {len(df_val)}")
    print(f"  Episodes   : {N_EPISODES}  |  ε start: {agent.epsilon:.2f}")
    print()

    best_val_pips = -np.inf
    best_w        = None          # dict of best weight arrays
    t0            = time.time()

    for ep in range(1, N_EPISODES + 1):
        # ── Training episode ──────────────────────────────────
        state = train_env.reset()
        while True:
            action          = agent.act(state)
            next_s, r, done = train_env.step(action)
            agent.push(state, action, r, next_s, done)

            if len(agent.buf) >= TRAIN_START:
                agent.train_step()

            state = next_s
            if done:
                break

        agent.decay_epsilon()

        # ── Validation every 50 episodes ─────────────────────
        if ep % 50 == 0 or ep == N_EPISODES:
            s = val_env.reset()
            while True:
                a      = agent._greedy(s)
                s, _, d = val_env.step(a)
                if d:
                    break

            vp      = val_env.total_pips
            elapsed = time.time() - t0
            print(f"  Ep {ep:4d}/{N_EPISODES}  "
                  f"ε={agent.epsilon:.3f}  "
                  f"train={train_env.total_pips:+8.1f}p  "
                  f"val={vp:+8.1f}p  "
                  f"trades={val_env.n_trades}  "
                  f"{elapsed:.0f}s")

            if vp > best_val_pips:
                best_val_pips = vp
                best_w = {
                    "W1": agent.q.W1.copy(), "b1": agent.q.b1.copy(),
                    "W2": agent.q.W2.copy(), "b2": agent.q.b2.copy(),
                    "W3": agent.q.W3.copy(), "b3": agent.q.b3.copy(),
                }

    # ── Save best model ───────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, f"rl_{pair}_{tf}")

    if best_w:
        np.savez(save_path + ".npz", **best_w)
        print(f"\n  ✓ Best val: {best_val_pips:+.1f} pips")
    else:
        agent.save(save_path)

    print(f"  ✓ Saved  → {save_path}.npz")
    print(f"  ✓ Done in {time.time()-t0:.0f}s")
    return True


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RadarFX RL Training — DQN agent per pair/timeframe"
    )
    parser.add_argument("pair", help="Pair key (EURUSD, GBPUSD, …) or 'all'")
    parser.add_argument("--tf", default="1h",
                        help=f"Timeframe: one of {VALID_TFS} or 'all'")
    args = parser.parse_args()

    pairs = list(PAIRS.keys()) if args.pair.upper() == "ALL" else [args.pair.upper()]
    tfs   = VALID_TFS          if args.tf  == "all"         else [args.tf]

    results = {}
    for pair in pairs:
        if pair not in PAIRS:
            print(f"  ✗ Unknown pair '{pair}'. Choices: {list(PAIRS.keys())}")
            continue
        for tf in tfs:
            if tf not in VALID_TFS:
                print(f"  ✗ Invalid tf '{tf}'. Choices: {VALID_TFS}")
                continue
            ok = train_one(pair, tf)
            results[f"{pair}_{tf}"] = "✓" if ok else "✗"

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'═'*58}")
    print("  Results:")
    for k, v in results.items():
        print(f"    {k:<22s}  {v}")
    print(f"{'═'*58}\n")

    print("  Push RL models to GitHub:")
    print("    git add models/rl_*.npz")
    print("    git commit -m 'feat: add RL DQN trading agents'")
    print("    git push\n")


if __name__ == "__main__":
    main()
