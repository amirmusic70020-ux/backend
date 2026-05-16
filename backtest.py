"""
RadarFX — backtest.py
Backtests the signal engine on XAU/USD (Gold) for the past 7 days.

Method:
  - Fetches 60 days of 1h candles for XAU/USD
  - Walks forward candle by candle through the last 7 days
  - Every 4 hours: uses only past data to generate a signal (no lookahead)
  - If BUY or SELL fires: looks forward up to 96 candles (4 days) to see
    whether TP1 or SL is hit first
  - Prints a full report at the end

Run:
  python backtest.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from data_feed    import fetch_candles, PAIRS
from analyzer     import analyze, calc_atr
from signal_engine import calc_sl_tp, calc_confidence, get_tf_confirmations, TF_WEIGHTS, MIN_CONFIDENCE, MIN_AGREEMENT

# ─────────────────────────────────────────────────────────────────────────────
import sys as _sys
_arg = _sys.argv[1].upper() if len(_sys.argv) > 1 else "XAUUSD"
PAIR = _arg if _arg in PAIRS else "XAUUSD"
if _arg not in PAIRS:
    print(f"Unknown pair '{_arg}'. Using XAUUSD. Available: {list(PAIRS.keys())}")

PAIR_INFO   = PAIRS[PAIR]
DAYS_BACK   = 30         # test window (30 days for more signals)
CHECK_EVERY = 4          # hours between signal checks
MAX_HOLD    = 96         # max candles to wait for TP/SL (hours)

# Backtest uses slightly looser thresholds to find more signals
BT_MIN_CONFIDENCE = 55   # live system uses 65
BT_MIN_AGREEMENT  = 0.45 # live system uses 0.55
# ─────────────────────────────────────────────────────────────────────────────


def print_separator(char="─", width=60):
    print(char * width)


def generate_signal_from_slice(df_1h: pd.DataFrame, df_daily_full: pd.DataFrame,
                               df_weekly_full: pd.DataFrame = None, debug=False) -> dict | None:
    """
    Generate a signal using only data available at this point in time.
    df_daily_full : 2 years of daily data (for EMA200)
    df_weekly_full: weekly data (for trend filter)
    """
    if len(df_1h) < 50:
        return None

    # Build 4h candles from 1h data
    df_4h = df_1h.resample("4h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()

    timeframe_data = {}
    if df_weekly_full is not None and len(df_weekly_full) >= 20:
        timeframe_data["weekly"] = df_weekly_full
    if len(df_daily_full) >= 50: timeframe_data["daily"] = df_daily_full
    if len(df_4h)         >= 20: timeframe_data["4h"]    = df_4h
    if len(df_1h)         >= 50: timeframe_data["1h"]    = df_1h

    if not timeframe_data:
        return None

    from analyzer import analyze_multi
    analyses = analyze_multi(timeframe_data)

    bull_score = bear_score = total_w = any_ranging = 0
    for tf, result in analyses.items():
        w    = TF_WEIGHTS.get(tf, 1)
        bias = result.get("bias", "neutral")
        total_w += w
        if bias == "bullish":  bull_score  += w
        elif bias == "bearish": bear_score  += w
        elif bias == "ranging": any_ranging += w

    ranging_pct = any_ranging / max(total_w, 1)
    bull_pct    = bull_score  / max(total_w, 1)
    bear_pct    = bear_score  / max(total_w, 1)

    if debug:
        bias_summary = {tf: r.get("bias","?") for tf, r in analyses.items()}
        print(f"    biases={bias_summary} bull={bull_pct:.2f} bear={bear_pct:.2f} ranging={ranging_pct:.2f}")

    if ranging_pct > 0.4:
        if debug: print(f"    → BLOCKED: ranging ({ranging_pct:.0%})")
        return None

    # Block if weekly is ranging
    if analyses.get("weekly", {}).get("bias", "neutral") == "ranging":
        if debug: print(f"    → BLOCKED: weekly is ranging")
        return None

    # Require at least 1h OR 4h to confirm (no purely daily signals)
    bias_1h = analyses.get("1h", {}).get("bias", "neutral")
    bias_4h = analyses.get("4h", {}).get("bias", "neutral")
    if bias_1h == "neutral" and bias_4h == "neutral":
        if debug: print(f"    → BLOCKED: 1h+4h both neutral (no short-term confirm)")
        return None

    if bull_score > bear_score and bull_pct >= BT_MIN_AGREEMENT:
        direction = "buy"
    elif bear_score > bull_score and bear_pct >= BT_MIN_AGREEMENT:
        direction = "sell"
    else:
        if debug: print(f"    → BLOCKED: no agreement (bull={bull_pct:.0%} bear={bear_pct:.0%} min={BT_MIN_AGREEMENT:.0%})")
        return None

    # calc_confidence expects "buy"/"sell" and maps internally to "bullish"/"bearish"
    confidence = calc_confidence(analyses, direction)
    if debug: print(f"    → direction={direction} confidence={confidence}%")
    if confidence < BT_MIN_CONFIDENCE:
        if debug: print(f"    → BLOCKED: low confidence ({confidence}% < {BT_MIN_CONFIDENCE}%)")
        return None

    primary_an = analyses.get("1h") or analyses.get("4h") or list(analyses.values())[0]
    entry      = round(float(df_1h["close"].iloc[-1]), PAIR_INFO["decimals"])
    atr        = primary_an.get("atr", entry * 0.001)
    support    = primary_an.get("support",    entry - atr * 3)
    resistance = primary_an.get("resistance", entry + atr * 3)

    levels = calc_sl_tp(
        entry, direction, atr, support, resistance,
        pip_factor=PAIR_INFO["pip_factor"],
        decimals=PAIR_INFO["decimals"],
    )

    return {
        "signal":     "BUY" if direction == "buy" else "SELL",
        "entry":      entry,
        "sl":         levels["sl"],
        "tp1":        levels["tp1"],
        "tp2":        levels["tp2"],
        "risk_pts":   levels["risk_pips"],
        "reward_pts": levels["reward_pips"],
        "rr":         levels["rr"],
        "confidence": confidence,
    }


def check_outcome(future_candles: pd.DataFrame, signal: dict) -> dict:
    """
    Walk future candles to see if TP1 or SL is hit first.
    Returns: { result: WIN/LOSS/OPEN, pnl_pts, candles_held }
    """
    direction = signal["signal"]
    tp1 = signal["tp1"]
    sl  = signal["sl"]

    for i, (ts, row) in enumerate(future_candles.iterrows()):
        high = row["high"]
        low  = row["low"]

        if direction == "BUY":
            if low  <= sl:  return {"result": "LOSS", "pnl_pts": -signal["risk_pts"],   "candles": i+1, "exit": sl}
            if high >= tp1: return {"result": "WIN",  "pnl_pts":  signal["reward_pts"],  "candles": i+1, "exit": tp1}
        else:  # SELL
            if high >= sl:  return {"result": "LOSS", "pnl_pts": -signal["risk_pts"],   "candles": i+1, "exit": sl}
            if low  <= tp1: return {"result": "WIN",  "pnl_pts":  signal["reward_pts"],  "candles": i+1, "exit": tp1}

    return {"result": "OPEN", "pnl_pts": 0, "candles": len(future_candles), "exit": None}


# ─────────────────────────────────────────────────────────────────────────────

def run_backtest():
    print_separator("═")
    print(f"  RadarFX Backtest — {PAIR_INFO['display']}")
    print(f"  Window : Last {DAYS_BACK} days")
    print(f"  Signal check every {CHECK_EVERY} hours")
    print_separator("═")

    # Fetch data
    print(f"\n[1] Fetching {PAIR_INFO['display']} data...")
    df_full = fetch_candles("1h", pair=PAIR)
    if df_full.empty:
        print("ERROR: No data returned.")
        return
    print(f"    Fetching daily data (2 years) for EMA200...")
    df_daily_all = fetch_candles("daily", pair=PAIR)
    if df_daily_all.empty:
        print("ERROR: No daily data returned.")
        return
    print(f"    Fetching weekly data for trend filter...")
    df_weekly_all = fetch_candles("weekly", pair=PAIR)
    if df_weekly_all.empty:
        print("WARNING: No weekly data — continuing without it.")

    # Timezone-aware cutoff
    now      = pd.Timestamp.now(tz="UTC")
    cutoff   = now - pd.Timedelta(days=DAYS_BACK)
    df_test  = df_full[df_full.index >= cutoff]

    print(f"[2] Test window: {cutoff.date()} → {now.date()}")
    print(f"    Total candles in window: {len(df_test)}")

    if len(df_test) < 10:
        print("ERROR: Not enough candles in the test window.")
        return

    # Walk forward
    print(f"\n[3] Walking forward (every {CHECK_EVERY} hours)...\n")
    signals_found = []
    last_signal_time = None

    for i in range(50, len(df_test)):
        current_ts  = df_test.index[i]
        current_pos = df_full.index.get_loc(current_ts)

        # Check only every CHECK_EVERY hours
        if last_signal_time and (current_ts - last_signal_time).total_seconds() < CHECK_EVERY * 3600:
            continue

        # Use all data up to (not including) current candle
        history = df_full.iloc[:current_pos]
        if len(history) < 50:
            continue

        # Slice daily/weekly data up to current date (no lookahead)
        current_date  = current_ts.date()
        history_daily  = df_daily_all[df_daily_all.index.date < current_date]
        history_weekly = df_weekly_all[df_weekly_all.index.date < current_date] if not df_weekly_all.empty else pd.DataFrame()

        sig = generate_signal_from_slice(history, history_daily, history_weekly, debug=(i < 60))
        if sig is None:
            continue

        # Found a signal — check outcome on future candles
        future = df_test.iloc[i+1 : i+1+MAX_HOLD]
        if future.empty:
            continue

        outcome = check_outcome(future, sig)

        record = {
            "time":       current_ts,
            "signal":     sig["signal"],
            "entry":      sig["entry"],
            "sl":         sig["sl"],
            "tp1":        sig["tp1"],
            "confidence": sig["confidence"],
            "rr":         sig["rr"],
            "risk_pts":   sig["risk_pts"],
            **outcome,
        }
        signals_found.append(record)
        last_signal_time = current_ts

        # Print each signal as found
        result_str = record["result"]
        pnl_str    = f"+{record['pnl_pts']:.1f}" if record["pnl_pts"] > 0 else f"{record['pnl_pts']:.1f}"
        dec = PAIR_INFO["decimals"]
        print(
            f"  {str(current_ts)[:16]}  "
            f"{sig['signal']:4s}  "
            f"Entry:{sig['entry']:.{dec}f}  "
            f"SL:{sig['sl']:.{dec}f}  "
            f"TP1:{sig['tp1']:.{dec}f}  "
            f"Conf:{sig['confidence']}%  "
            f"→  {result_str:4s}  {pnl_str} pts  ({record['candles']}h)"
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    print_separator()
    print(f"\n  BACKTEST RESULTS — {PAIR_INFO['display']} — Last {DAYS_BACK} days")
    print_separator()

    if not signals_found:
        print("  No signals fired in this period.")
        print("  (Market may have been in a range — WAIT is the correct response)")
        print_separator()
        return

    total  = len(signals_found)
    wins   = [r for r in signals_found if r["result"] == "WIN"]
    losses = [r for r in signals_found if r["result"] == "LOSS"]
    open_  = [r for r in signals_found if r["result"] == "OPEN"]

    win_rate = len(wins) / (len(wins) + len(losses)) * 100 if (wins or losses) else 0
    total_pnl = sum(r["pnl_pts"] for r in signals_found)
    avg_conf  = sum(r["confidence"] for r in signals_found) / total

    print(f"  Total signals  : {total}")
    print(f"  Wins           : {len(wins)}")
    print(f"  Losses         : {len(losses)}")
    print(f"  Still open     : {len(open_)}")
    print(f"  Win rate       : {win_rate:.1f}%")
    print(f"  Total P&L      : {total_pnl:+.1f} pts")
    print(f"  Avg confidence : {avg_conf:.1f}%")

    if wins:
        avg_win_h = sum(r["candles"] for r in wins) / len(wins)
        print(f"  Avg hold (win) : {avg_win_h:.1f} hours")
    if losses:
        avg_loss_h = sum(r["candles"] for r in losses) / len(losses)
        print(f"  Avg hold (loss): {avg_loss_h:.1f} hours")

    print_separator()

    if win_rate >= 60:
        print("  Assessment: System performed well in this period.")
    elif win_rate >= 40:
        print("  Assessment: Mixed results — market may have been choppy.")
    else:
        print("  Assessment: Tough week for trend-following signals.")

    print_separator("═")
    print()


if __name__ == "__main__":
    run_backtest()
