"""
RadarFX — telegram_bot.py
Scans all 5 pairs every CHECK_INTERVAL_MINUTES and sends BUY/SELL signals to Telegram.

Usage:
  python telegram_bot.py          → run forever (loop mode)
  python telegram_bot.py once     → scan all pairs once and exit
  python telegram_bot.py eurusd   → scan a single pair once and exit
"""

import requests
import time
import sys
from datetime import datetime

from data_feed     import get_multi_timeframe, PAIRS
from signal_engine import generate_signal

# ─────────────────────────────────────────────────────────────────────────────
BOT_TOKEN  = "8874736945:AAF-1g5U4m9jNIBQig9KaFQIYjcoehP5n5I"
CHAT_ID    = "81462940"
# ─────────────────────────────────────────────────────────────────────────────

# Which pairs to monitor (all 5 by default — comment out any you don't want)
ACTIVE_PAIRS = ["EURUSD", "GBPUSD", "XAUUSD", "USDCHF", "USDJPY"]

# Timeframes to fetch per pair
TIMEFRAMES = ["daily", "4h", "1h", "30m", "15m"]

# Check every X minutes
CHECK_INTERVAL_MINUTES = 60

# If True, never sends WAIT signals — only BUY or SELL
ONLY_SEND_TRADE_SIGNALS = True


# ─────────────────────────────────────────────────────────────────────────────
#  TELEGRAM HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def send_message(text: str):
    url  = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        r = requests.post(url, data=data, timeout=10)
        if r.status_code == 200:
            print("[RadarFX Bot] Message sent")
        else:
            print(f"[RadarFX Bot] Send failed: {r.text}")
    except Exception as e:
        print(f"[RadarFX Bot] Error: {e}")


def format_telegram_message(signal: dict) -> str:
    """Format a signal as a clean Telegram HTML message."""
    now       = datetime.now().strftime("%Y-%m-%d %H:%M")
    pair      = signal.get("pair", "?")
    pip_label = signal.get("pip_label", "pips")

    if signal["signal"] == "WAIT":
        return (
            f"RadarFX — WAIT\n"
            f"{pair}\n"
            f"{now}\n\n"
            f"{signal.get('reason', 'No clear signal.')}\n"
            f"Stay out. No trade."
        )

    direction = signal["signal"]
    arrow     = "BUY" if direction == "BUY" else "SELL"

    # Timeframe confirmations block
    tf_lines = ""
    for tf, status in signal.get("timeframe_confirmations", {}).items():
        icon = "[OK]" if status == "confirmed" else "[--]" if status in ("neutral", "ranging") else "[NO]"
        tf_lines += f"  {icon} {tf.upper()}\n"

    msg = (
        f"<b>RadarFX Signal</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Pair:</b>        {pair}\n"
        f"<b>Signal:</b>      {arrow}\n"
        f"<b>Confidence:</b>  {signal['confidence']}%\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Entry:</b>       {signal['entry']}\n"
        f"<b>Stop Loss:</b>   {signal['sl']}  ({signal['risk_pips']} {pip_label})\n"
        f"<b>TP1:</b>         {signal['tp1']}\n"
        f"<b>TP2:</b>         {signal['tp2']}  ({signal['reward_pips']} {pip_label})\n"
        f"<b>R:R:</b>         1 : {signal['rr']}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Timeframes:</b>\n{tf_lines}"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{now}\n"
        f"<i>Not financial advice.</i>"
    )
    return msg


# ─────────────────────────────────────────────────────────────────────────────
#  SCAN LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def scan_pair(pair_key: str) -> dict | None:
    """
    Fetch and analyse one pair.
    Returns the signal dict, or None on error.
    """
    try:
        tf_data = get_multi_timeframe(TIMEFRAMES, pair=pair_key)
        if not tf_data:
            print(f"[RadarFX Bot] No data for {pair_key}")
            return None
        signal = generate_signal(tf_data, pair=pair_key)
        signal.pop("analyses", None)
        return signal
    except Exception as e:
        print(f"[RadarFX Bot] Error scanning {pair_key}: {e}")
        return None


def run_once(pairs: list = None):
    """
    Scan all active pairs (or a custom list) once.
    Sends a Telegram message for each BUY or SELL signal found.
    """
    if pairs is None:
        pairs = ACTIVE_PAIRS

    now = datetime.now().strftime("%H:%M:%S")
    print(f"\n[RadarFX Bot] Scanning {len(pairs)} pair(s)... {now}")

    signals_sent = 0
    for pair_key in pairs:
        signal = scan_pair(pair_key)
        if signal is None:
            continue

        result = signal["signal"]
        print(f"  {pair_key:8s} → {result}  (confidence: {signal.get('confidence', 0)}%)")

        if ONLY_SEND_TRADE_SIGNALS and result == "WAIT":
            continue

        msg = format_telegram_message(signal)
        send_message(msg)
        signals_sent += 1
        time.sleep(1)   # small delay between messages if multiple signals

    if signals_sent == 0:
        print("[RadarFX Bot] No trade signals to send this round.")


def run_loop():
    """Run forever, scanning all pairs every CHECK_INTERVAL_MINUTES."""
    pairs_list = ", ".join(ACTIVE_PAIRS)
    print(f"[RadarFX Bot] Started — monitoring: {pairs_list}")
    print(f"[RadarFX Bot] Checking every {CHECK_INTERVAL_MINUTES} minutes.")

    send_message(
        f"<b>RadarFX Bot is online!</b>\n"
        f"Monitoring: {pairs_list}\n"
        f"Interval: every {CHECK_INTERVAL_MINUTES} minutes.\n"
        f"You will be notified when a BUY or SELL signal fires."
    )

    while True:
        try:
            run_once()
        except Exception as e:
            print(f"[RadarFX Bot] Unexpected error: {e}")

        print(f"[RadarFX Bot] Next check in {CHECK_INTERVAL_MINUTES} min...\n")
        time.sleep(CHECK_INTERVAL_MINUTES * 60)


# ─── Entry point ───────────────────────────
if __name__ == "__main__":
    args = [a.upper() for a in sys.argv[1:]]

    if not args:
        # Default: run loop for all pairs
        run_loop()
    elif args[0] == "ONCE":
        # python telegram_bot.py once  → scan all pairs once
        run_once()
    elif args[0] in PAIRS:
        # python telegram_bot.py eurusd  → scan one pair once
        run_once(pairs=[args[0]])
    else:
        print(f"Usage:")
        print(f"  python telegram_bot.py             — run forever (all pairs)")
        print(f"  python telegram_bot.py once        — scan all pairs once")
        print(f"  python telegram_bot.py <PAIR>      — scan one pair once")
        print(f"  Available pairs: {', '.join(PAIRS.keys())}")
