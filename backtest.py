"""
RadarFX — backtest.py
Tests ML + RL models on real historical data.
Shows win rate, total pips, drawdown, and trade log.

Usage
-----
    python backtest.py EURUSD --tf 1h
    python backtest.py EURUSD --tf 4h --plot
    python backtest.py all --tf 1h
"""

import os, sys, argparse, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data_feed import fetch_candles, PAIRS
from ml_model  import add_features, FEATURE_COLS, LOOKBACK, predict as ml_predict
from rl_env    import ForexTradingEnv
from rl_agent  import DQNAgent

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


# ─── helpers ─────────────────────────────────────────────────────────────────

def rl_model_path(pair, tf):
    return os.path.join(MODEL_DIR, f"rl_{pair}_{tf}.npz")

def ml_model_exists(pair, tf):
    key = f"{pair}_{tf}"
    return all(os.path.exists(os.path.join(MODEL_DIR, f"{key}_{q}.pkl"))
               for q in ["bear","median","bull"])

def rl_model_exists(pair, tf):
    return os.path.exists(rl_model_path(pair, tf) + ".npz") or \
           os.path.exists(rl_model_path(pair, tf))

def pct(a, b):
    return f"{'+' if a>=b else ''}{((a/b-1)*100):.2f}%"


# ─── Backtest engine ─────────────────────────────────────────────────────────

def backtest(pair: str, tf: str, plot: bool = False) -> dict:
    pip_factor  = PAIRS[pair]["pip_factor"]
    pip_label   = PAIRS[pair]["pip_label"]
    decimals    = PAIRS[pair]["decimals"]
    spread_pips = 5.0 if pair == "XAUUSD" else 2.0

    # ── 1. Fetch data ────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Backtest: {PAIRS[pair]['display']} | {tf}")
    print(f"{'═'*60}")

    df_raw = fetch_candles(tf, pair=pair)
    if df_raw is None or df_raw.empty:
        print("  ✗ No data"); return {}

    df = add_features(df_raw)

    # Use last 20% of data as unseen test set
    split    = int(len(df) * 0.80)
    df_test  = df.iloc[split:].reset_index(drop=True)
    closes   = df_test["close"].values
    n        = len(df_test)

    if n < 30:
        print(f"  ✗ Test set too small ({n} bars)"); return {}

    print(f"  Test bars : {n}  ({df_test.index[0] if hasattr(df_test.index[0],'strftime') else 'start'} → end)")
    print(f"  Spread    : {spread_pips} {pip_label}\n")

    results = {}

    # ══════════════════════════════════════════════════════════════
    # STRATEGY 1 — ML Only
    # ══════════════════════════════════════════════════════════════
    if ml_model_exists(pair, tf):
        trades   = []
        equity   = [0.0]
        position = 0
        entry_px = 0.0

        for i in range(LOOKBACK, n - 1):
            # Predict on window ending at bar i
            df_window = df.iloc[:split + i]
            try:
                med, bull, bear, last_close = ml_predict(df_window, pair=f"{pair}_{tf}")
            except Exception:
                continue

            pct_chg = (float(med[-1]) / last_close - 1) * 100
            signal  = "BUY" if pct_chg > 0.05 else "SELL" if pct_chg < -0.05 else "HOLD"

            price = closes[i]

            # Simple: enter on signal, exit when signal flips
            if signal == "BUY" and position != 1:
                if position == -1:
                    pnl = (entry_px - price) * pip_factor - spread_pips
                    trades.append({"type":"SELL","pnl":pnl,"bars":i})
                    equity.append(equity[-1] + pnl)
                position = 1; entry_px = price

            elif signal == "SELL" and position != -1:
                if position == 1:
                    pnl = (price - entry_px) * pip_factor - spread_pips
                    trades.append({"type":"BUY","pnl":pnl,"bars":i})
                    equity.append(equity[-1] + pnl)
                position = -1; entry_px = price

        # Close any open position
        if position != 0:
            last = closes[-1]
            pnl  = ((last - entry_px) if position == 1 else (entry_px - last)) * pip_factor - spread_pips
            trades.append({"type":"CLOSE","pnl":pnl,"bars":n-1})
            equity.append(equity[-1] + pnl)

        ml_res = _stats(trades, equity, pip_label)
        results["ML"] = ml_res
        _print_stats("ML Only (Gradient Boosting)", ml_res, pip_label)
    else:
        print("  ⚠ ML model not found — skipping ML backtest")

    # ══════════════════════════════════════════════════════════════
    # STRATEGY 2 — RL Only
    # ══════════════════════════════════════════════════════════════
    path = rl_model_path(pair, tf)
    if not os.path.exists(path + ".npz"):
        path_candidate = path
    else:
        path_candidate = path + ".npz"

    rl_path = path + ".npz" if os.path.exists(path + ".npz") else path

    if os.path.exists(rl_path):
        agent = DQNAgent.load(rl_path, state_dim=ForexTradingEnv.STATE_DIM)

        env    = ForexTradingEnv(df_test, pip_factor=pip_factor, spread_pips=spread_pips)
        state  = env.reset()
        equity = [0.0]
        trades = []

        while True:
            action          = agent._greedy(state)
            prev_trades     = env.n_trades
            prev_pips       = env.total_pips
            state, r, done  = env.step(action)

            if env.n_trades > prev_trades:
                pnl = env.total_pips - prev_pips
                trades.append({"type": ["HOLD","BUY","SELL"][action], "pnl": pnl})
                equity.append(env.total_pips)

            if done: break

        rl_res = _stats(trades, equity, pip_label)
        results["RL"] = rl_res
        _print_stats("RL Only (DQN Agent)", rl_res, pip_label)
    else:
        print("  ⚠ RL model not found — skipping RL backtest")
        print(f"    Train first: python rl_train.py {pair} --tf {tf}")

    # ══════════════════════════════════════════════════════════════
    # STRATEGY 3 — ML + RL Combined (enter only when both agree)
    # ══════════════════════════════════════════════════════════════
    if "ML" in results and "RL" in results and os.path.exists(rl_path):
        agent    = DQNAgent.load(rl_path, state_dim=ForexTradingEnv.STATE_DIM)
        trades   = []
        equity   = [0.0]
        position = 0
        entry_px = 0.0

        feats  = df_test[FEATURE_COLS].values.astype(np.float32)

        for i in range(LOOKBACK, n - 1):
            # ML signal
            df_window = df.iloc[:split + i]
            try:
                med, _, _, last_close = ml_predict(df_window, pair=f"{pair}_{tf}")
                pct_chg  = (float(med[-1]) / last_close - 1) * 100
                ml_sig   = "BUY" if pct_chg > 0.05 else "SELL" if pct_chg < -0.05 else "HOLD"
            except Exception:
                continue

            # RL signal (current bar features, flat position)
            feat   = feats[i].copy()
            state  = np.concatenate([feat, np.zeros(3, dtype=np.float32)])
            q_vals = agent.q.forward(state.reshape(1,-1))[0]
            rl_act = int(np.argmax(q_vals))
            rl_sig = ["HOLD","BUY","SELL"][rl_act]

            price = closes[i]

            # Only enter when BOTH agree
            combined = "BUY"  if ml_sig == "BUY"  and rl_sig == "BUY"  else \
                       "SELL" if ml_sig == "SELL" and rl_sig == "SELL" else \
                       "HOLD"

            if combined == "BUY" and position != 1:
                if position == -1:
                    pnl = (entry_px - price) * pip_factor - spread_pips
                    trades.append({"type":"SELL","pnl":pnl})
                    equity.append(equity[-1] + pnl)
                position = 1; entry_px = price

            elif combined == "SELL" and position != -1:
                if position == 1:
                    pnl = (price - entry_px) * pip_factor - spread_pips
                    trades.append({"type":"BUY","pnl":pnl})
                    equity.append(equity[-1] + pnl)
                position = -1; entry_px = price

            elif combined == "HOLD" and position != 0:
                # Exit when conflicted
                if position == 1:
                    pnl = (price - entry_px) * pip_factor - spread_pips
                elif position == -1:
                    pnl = (entry_px - price) * pip_factor - spread_pips
                trades.append({"type":"EXIT","pnl":pnl})
                equity.append(equity[-1] + pnl)
                position = 0

        # Close open position
        if position != 0:
            last = closes[-1]
            pnl  = ((last-entry_px) if position==1 else (entry_px-last))*pip_factor - spread_pips
            trades.append({"type":"CLOSE","pnl":pnl})
            equity.append(equity[-1] + pnl)

        comb_res = _stats(trades, equity, pip_label)
        results["COMBINED"] = comb_res
        _print_stats("ML + RL Combined (both must agree)", comb_res, pip_label)

    # ── Summary comparison ───────────────────────────────────────
    if len(results) > 1:
        print(f"\n{'─'*60}")
        print(f"  {'SUMMARY':^56}")
        print(f"{'─'*60}")
        print(f"  {'Strategy':<28} {'Total':>8} {'Win%':>7} {'Trades':>7} {'MaxDD':>8}")
        print(f"  {'─'*56}")
        for name, r in results.items():
            print(f"  {name:<28} {r['total_pips']:>+8.1f} {r['win_rate']:>6.1f}% "
                  f"{r['n_trades']:>7} {r['max_dd']:>+8.1f}")
        print(f"{'═'*60}\n")

    # ── Optional matplotlib plot ─────────────────────────────────
    if plot and results:
        _plot(results, pair, tf, pip_label)

    return results


# ─── Stats helper ─────────────────────────────────────────────────────────────

def _stats(trades: list, equity: list, pip_label: str) -> dict:
    if not trades:
        return {"total_pips":0,"win_rate":0,"n_trades":0,"avg_pip":0,"max_dd":0,"equity":[0]}

    pnls     = [t["pnl"] for t in trades]
    wins     = [p for p in pnls if p > 0]
    total    = sum(pnls)
    win_rate = len(wins) / len(pnls) * 100 if pnls else 0
    avg_pip  = total / len(pnls) if pnls else 0

    # Max drawdown
    eq   = np.array(equity)
    peak = np.maximum.accumulate(eq)
    dd   = eq - peak
    max_dd = float(dd.min())

    return {
        "total_pips": round(total, 1),
        "win_rate":   round(win_rate, 1),
        "n_trades":   len(pnls),
        "avg_pip":    round(avg_pip, 1),
        "max_dd":     round(max_dd, 1),
        "equity":     equity,
    }


def _print_stats(title: str, r: dict, pip_label: str):
    bar  = "█" * min(int(r["win_rate"] / 5), 20)
    sign = "✅" if r["total_pips"] > 0 else "❌"
    print(f"  ┌─ {title}")
    print(f"  │  Total P&L   : {r['total_pips']:+.1f} {pip_label}  {sign}")
    print(f"  │  Win Rate    : {r['win_rate']:.1f}%  {bar}")
    print(f"  │  Trades      : {r['n_trades']}")
    print(f"  │  Avg/trade   : {r['avg_pip']:+.1f} {pip_label}")
    print(f"  │  Max Drawdown: {r['max_dd']:+.1f} {pip_label}")
    print(f"  └{'─'*50}")


def _plot(results: dict, pair: str, tf: str, pip_label: str):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        colors = {"ML":"#448aff", "RL":"#00e6b4", "COMBINED":"#ffca28"}
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")

        for name, r in results.items():
            eq = r["equity"]
            ax.plot(eq, label=f"{name} ({r['total_pips']:+.0f} {pip_label})",
                    color=colors.get(name,"white"), linewidth=2)

        ax.axhline(0, color="white", linewidth=0.5, alpha=0.3)
        ax.set_title(f"RadarFX Backtest — {PAIRS[pair]['display']} {tf}",
                     color="white", fontsize=14, pad=12)
        ax.set_xlabel("Trades", color="#888")
        ax.set_ylabel(f"Cumulative {pip_label}", color="#888")
        ax.tick_params(colors="#888")
        ax.spines["bottom"].set_color("#333")
        ax.spines["left"].set_color("#333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")

        save_path = os.path.join(os.path.dirname(__file__),
                                 f"backtest_{pair}_{tf}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  📊 Chart saved → {save_path}")
    except Exception as e:
        print(f"  (plot skipped: {e})")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RadarFX Backtest Engine")
    parser.add_argument("pair", help="Pair (EURUSD, …) or 'all'")
    parser.add_argument("--tf",   default="1h")
    parser.add_argument("--plot", action="store_true",
                        help="Save equity curve as PNG")
    args = parser.parse_args()

    pairs = list(PAIRS.keys()) if args.pair.upper()=="ALL" else [args.pair.upper()]

    for pair in pairs:
        if pair not in PAIRS:
            print(f"Unknown pair: {pair}"); continue
        backtest(pair, args.tf, plot=args.plot)


if __name__ == "__main__":
    main()
