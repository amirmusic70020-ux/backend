"""
RadarFX — signal_engine.py
Combines multi-timeframe analysis into one final signal.
Supports all 5 pairs: EUR/USD, GBP/USD, XAU/USD, USD/CHF, USD/JPY.
Output: BUY / SELL / WAIT  +  Entry, Stop Loss, Take Profit, Confidence score.
"""

import pandas as pd
from datetime import datetime

from analyzer import analyze_multi, calc_atr
from data_feed import PAIRS, DEFAULT_PAIR


# ─────────────────────────────────────────────────────────────────────────────
#  TIMEFRAME WEIGHTS  (higher = more important)
# ─────────────────────────────────────────────────────────────────────────────

TF_WEIGHTS = {
    "monthly": 5,
    "weekly":  4,
    "daily":   3,
    "4h":      2,
    "1h":      2,
    "30m":     1,
    "15m":     1,
    "5m":      1,
}

# Minimum confidence to fire a signal (0–100)
MIN_CONFIDENCE = 65

# Minimum fraction of weighted TFs that must agree
MIN_AGREEMENT = 0.55


# ─────────────────────────────────────────────────────────────────────────────
#  SL / TP CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────

def calc_sl_tp(entry: float, direction: str, atr: float,
               support: float, resistance: float,
               pip_factor: int = 10000,
               decimals: int = 5,
               rr_ratio: float = 2.0) -> dict:
    """
    Calculate Stop Loss and Take Profit using ATR + structure levels.

    Args:
        entry      : current price
        direction  : 'buy' or 'sell'
        atr        : Average True Range of the primary timeframe
        support    : key support level
        resistance : key resistance level
        pip_factor : units per pip (10000 for majors, 100 for JPY, 10 for XAU)
        decimals   : decimal places for rounding prices
        rr_ratio   : target risk:reward (default 2.0)

    Returns:
        dict with sl, tp1, tp2, risk_pips, reward_pips, rr
    """
    atr_buffer = atr * 1.2

    if direction == "buy":
        sl   = round(min(entry - atr_buffer, support  - atr * 0.3), decimals)
        risk = round(entry - sl, decimals)
        tp1  = round(entry + risk * rr_ratio,             decimals)
        tp2  = round(min(entry + risk * rr_ratio * 1.8, resistance), decimals)
    else:  # sell
        sl   = round(max(entry + atr_buffer, resistance + atr * 0.3), decimals)
        risk = round(sl - entry, decimals)
        tp1  = round(entry - risk * rr_ratio,             decimals)
        tp2  = round(max(entry - risk * rr_ratio * 1.8, support),   decimals)

    risk_pips   = round(risk * pip_factor, 1)
    reward_pips = round(risk * rr_ratio * pip_factor, 1)
    actual_rr   = round(reward_pips / risk_pips, 2) if risk_pips > 0 else 0

    return {
        "sl":          sl,
        "tp1":         tp1,
        "tp2":         tp2,
        "risk_pips":   risk_pips,
        "reward_pips": reward_pips,
        "rr":          actual_rr,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIDENCE SCORE  (0 → 100)
# ─────────────────────────────────────────────────────────────────────────────

def calc_confidence(analyses: dict, direction: str) -> int:
    total_weight = 0
    agree_weight = 0

    for tf, result in analyses.items():
        w    = TF_WEIGHTS.get(tf, 1)
        bias = result.get("bias", "neutral")
        total_weight += w

        if bias == direction:
            agree_weight += w
        elif bias in ("ranging", "neutral"):
            agree_weight += w * 0.3   # partial credit

    if total_weight == 0:
        return 0
    return min(max(int(round(agree_weight / total_weight * 100)), 0), 100)


# ─────────────────────────────────────────────────────────────────────────────
#  TIMEFRAME CONFIRMATIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_tf_confirmations(analyses: dict, direction: str) -> dict:
    confirmations = {}
    for tf, result in analyses.items():
        bias = result.get("bias", "neutral")
        if bias == direction:
            confirmations[tf] = "confirmed"
        elif bias == "ranging":
            confirmations[tf] = "ranging"
        elif bias == "neutral":
            confirmations[tf] = "neutral"
        else:
            confirmations[tf] = "against"
    return confirmations


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN SIGNAL GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_signal(timeframe_data: dict,
                    current_price: float = None,
                    pair: str = DEFAULT_PAIR) -> dict:
    """
    Generate the final RadarFX signal from multi-timeframe data.

    Args:
        timeframe_data : dict of { timeframe: DataFrame }
        current_price  : override entry price (optional)
        pair           : key from PAIRS dict (e.g. "EURUSD", "GBPUSD", "XAUUSD" …)

    Returns:
        Full signal dict with direction, entry, SL, TP, confidence, etc.
    """
    pair = pair.upper()
    pair_info  = PAIRS.get(pair, PAIRS[DEFAULT_PAIR])
    pip_factor = pair_info["pip_factor"]
    decimals   = pair_info["decimals"]
    display    = pair_info["display"]
    pip_label  = pair_info["pip_label"]

    analyses = analyze_multi(timeframe_data)

    # ── Score each direction ──────────────────────────────────────────────────
    bull_score  = 0
    bear_score  = 0
    total_w     = 0
    any_ranging = 0

    for tf, result in analyses.items():
        w    = TF_WEIGHTS.get(tf, 1)
        bias = result.get("bias", "neutral")
        total_w += w

        if bias == "bullish":
            bull_score  += w
        elif bias == "bearish":
            bear_score  += w
        elif bias == "ranging":
            any_ranging += w

    # ── Block signal if market is dominated by ranging ────────────────────────
    if any_ranging / max(total_w, 1) > 0.4:
        return {
            "signal":     "WAIT",
            "pair":       display,
            "reason":     "Market is ranging — no clear trend. Stay out.",
            "confidence": 0,
            "timestamp":  datetime.utcnow().isoformat(),
            "analyses":   analyses,
        }

    # ── Decide direction ──────────────────────────────────────────────────────
    if bull_score > bear_score and bull_score / max(total_w, 1) >= MIN_AGREEMENT:
        direction = "buy"
    elif bear_score > bull_score and bear_score / max(total_w, 1) >= MIN_AGREEMENT:
        direction = "sell"
    else:
        return {
            "signal":     "WAIT",
            "pair":       display,
            "reason":     "No clear agreement across timeframes.",
            "confidence": calc_confidence(analyses, "bullish"),
            "timestamp":  datetime.utcnow().isoformat(),
            "analyses":   analyses,
        }

    # ── Use primary timeframe for price levels ────────────────────────────────
    primary_tf = next(
        (tf for tf in ["1h", "30m", "daily", "15m"] if tf in timeframe_data), None
    )
    primary_df = timeframe_data.get(primary_tf) if primary_tf else list(timeframe_data.values())[0]
    primary_an = analyses.get(primary_tf, {})

    entry      = current_price or round(float(primary_df["close"].iloc[-1]), decimals)
    atr        = primary_an.get("atr",        entry * 0.001)     # fallback: 0.1% of price
    support    = primary_an.get("support",    entry - atr * 3)
    resistance = primary_an.get("resistance", entry + atr * 3)

    # ── Calculate SL / TP ─────────────────────────────────────────────────────
    levels = calc_sl_tp(
        entry, direction, atr, support, resistance,
        pip_factor=pip_factor,
        decimals=decimals,
    )

    # ── Confidence ────────────────────────────────────────────────────────────
    confidence = calc_confidence(analyses, direction)

    if confidence < MIN_CONFIDENCE:
        return {
            "signal":     "WAIT",
            "pair":       display,
            "reason":     f"Confidence too low ({confidence}%) — min {MIN_CONFIDENCE}%.",
            "confidence": confidence,
            "timestamp":  datetime.utcnow().isoformat(),
            "analyses":   analyses,
        }

    # ── Build reason string ───────────────────────────────────────────────────
    pattern     = primary_an.get("pattern", "none")
    pattern_str = {
        "bullish_engulfing": "Bullish Engulfing",
        "bearish_engulfing": "Bearish Engulfing",
        "bullish_pin_bar":   "Bullish Pin Bar (hammer)",
        "bearish_pin_bar":   "Bearish Pin Bar (shooting star)",
        "doji":              "Doji — indecision",
        "none":              "",
    }.get(pattern, "")

    reason_parts = [
        f"Trend: {primary_an.get('trend', '')}",
        f"MACD: {primary_an.get('macd_cross', '')} crossover",
        f"ADX: {primary_an.get('adx', 0):.1f}",
        f"RSI: {primary_an.get('rsi', 0):.1f}",
    ]
    if pattern_str:
        reason_parts.append(f"Pattern: {pattern_str}")

    confirmations = get_tf_confirmations(analyses, direction)

    signal = {
        "signal":          "BUY" if direction == "buy" else "SELL",
        "pair":            display,
        "pair_key":        pair,
        "entry":           entry,
        "sl":              levels["sl"],
        "tp1":             levels["tp1"],
        "tp2":             levels["tp2"],
        "risk_pips":       levels["risk_pips"],
        "reward_pips":     levels["reward_pips"],
        "pip_label":       pip_label,
        "rr":              levels["rr"],
        "confidence":      confidence,
        "reason":          " | ".join(reason_parts),
        "pattern":         pattern,
        "primary_tf":      primary_tf,
        "timeframe_confirmations": confirmations,
        "timestamp":       datetime.utcnow().isoformat(),
        "analyses":        analyses,
    }

    return signal


def format_signal(signal: dict) -> str:
    """Pretty-print a signal to the terminal."""
    pip_label = signal.get("pip_label", "pips")

    if signal["signal"] == "WAIT":
        return (
            f"\n{'='*50}\n"
            f"  RadarFX  {signal.get('pair','?')}  —  {signal['timestamp'][:16]}\n"
            f"{'='*50}\n"
            f"  WAIT — {signal['reason']}\n"
            f"{'='*50}\n"
        )

    lines = [
        f"\n{'='*50}",
        f"  RadarFX  {signal['pair']}  —  {signal['timestamp'][:16]}",
        f"{'='*50}",
        f"  {'BUY' if signal['signal']=='BUY' else 'SELL'}  {signal['pair']}",
        f"  Confidence : {signal['confidence']}%",
        f"  Entry      : {signal['entry']}",
        f"  Stop Loss  : {signal['sl']}   ({signal['risk_pips']} {pip_label})",
        f"  TP1        : {signal['tp1']}",
        f"  TP2        : {signal['tp2']}   ({signal['reward_pips']} {pip_label})",
        f"  R:R        : 1 : {signal['rr']}",
        f"  Pattern    : {signal['pattern']}",
        f"  Reason     : {signal['reason']}",
        f"\n  Timeframe confirmations:",
    ]
    for tf, status in signal["timeframe_confirmations"].items():
        icon = "+" if status == "confirmed" else "~" if status in ("neutral", "ranging") else "-"
        lines.append(f"    [{icon}] {tf:8s} -> {status}")
    lines.append(f"{'='*50}\n")
    return "\n".join(lines)


# ─── Quick test ───────────────────────────
if __name__ == "__main__":
    from data_feed import get_multi_timeframe

    for p in ["EURUSD", "GBPUSD", "XAUUSD", "USDCHF", "USDJPY"]:
        print(f"\n[RadarFX] Fetching {p}...")
        tf_data = get_multi_timeframe(["daily", "1h", "30m", "15m"], pair=p)
        if tf_data:
            signal = generate_signal(tf_data, pair=p)
            print(format_signal(signal))
        else:
            print(f"  No data for {p}")
