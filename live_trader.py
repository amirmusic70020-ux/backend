"""
RadarFX — live_trader.py
Automated forex trading bot powered by ML + RL.

Strategy:
  • ML (Gradient Boosting) = primary signal (BUY / SELL / WAIT)
  • RL (DQN Agent)         = filter — skip if strongly disagrees
  • SL  = worst-case percentile from ML (bear for BUY, bull for SELL)
  • TP  = best-case percentile from ML  (bull for BUY, bear for SELL)
  • Lot = 1% account risk per trade

Usage:
  python live_trader.py                  # trade all pairs, 1h TF
  python live_trader.py --pair EURUSD    # single pair
  python live_trader.py --tf 4h         # different timeframe
  python live_trader.py --dry           # dry run (no real orders)
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from mt5_bridge  import (connect, disconnect, get_candles, get_account,
                          get_balance, place_order, get_open_positions,
                          get_current_price, modify_position_sl)
from ml_model    import predict as ml_predict, add_features, FEATURE_COLS
from rl_agent    import DQNAgent
from data_feed   import PAIRS
from news_feed   import is_safe_to_trade
import notifier

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG — tweak these to your liking
# ─────────────────────────────────────────────────────────────────────────────

PAIRS_TO_TRADE   = ["EURUSD", "GBPUSD", "XAUUSD", "USDCHF", "USDJPY"]
DEFAULT_TF       = "1h"
CHECK_EVERY_SEC  = 60 * 60       # scan every 1 hour
RISK_PCT         = 0.01          # 1% of balance risked per trade
MIN_MOVE_PIPS    = 10            # minimum median move to enter (ignore tiny signals)
MAX_POS_PER_PAIR = 1             # max simultaneous positions per pair
USE_RL_FILTER    = False         # ML alone — better results per backtest
MIN_RR           = 1.0           # minimum R:R ratio to enter (reward/risk)
CANDLES_TO_FETCH = 300           # enough for feature engineering

# ─── Trading hours (UTC) — only trade during liquid sessions ─────────────────
# (start_hour, end_hour)  — end_hour is exclusive
TRADING_HOURS = {
    "EURUSD": (7,  17),   # London + NY overlap
    "GBPUSD": (7,  17),   # London session
    "USDCHF": (7,  17),   # London + NY
    "USDJPY": (0,   9),   # Tokyo + London open
    "XAUUSD": (7,  20),   # London + NY (gold trades wider hours)
}
# Days to avoid: 4=Friday after 20UTC, 6=Sunday before 21UTC
# We block: Friday 20:00+ and Sunday all day (handled in is_trading_hour)

LOG_FILE = os.path.join(os.path.dirname(__file__), "trader.log")

# Tracks previously known positions {ticket: position_dict} for close detection
_prev_positions: dict = {}

# Accumulates today's closed trades for the daily report
_daily_closed: list = []
_last_report_date: int = -1   # day-of-year when we last sent the report

# ─────────────────────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("RadarFX")


# ─────────────────────────────────────────────────────────────────────────────
#  RISK / LOT SIZING
# ─────────────────────────────────────────────────────────────────────────────

def is_trading_hour(pair: str) -> bool:
    """
    Returns True if current UTC time is within the liquid trading window
    for this pair. Blocks weekends and off-hours.
    """
    now  = datetime.now(timezone.utc)
    hour = now.hour
    wday = now.weekday()   # 0=Mon … 4=Fri, 5=Sat, 6=Sun

    # Block Saturday entirely
    if wday == 5:
        return False

    # Block Sunday before 21:00 UTC (market opens ~21:00 Sun)
    if wday == 6 and hour < 21:
        return False

    # Block Friday after 20:00 UTC (liquidity dries up)
    if wday == 4 and hour >= 20:
        return False

    start, end = TRADING_HOURS.get(pair, (0, 24))
    return start <= hour < end


def calc_lot(pair: str, sl_pips: float, balance: float) -> float:
    """
    Lot size = risk_amount / (sl_pips × pip_value_per_lot)

    Pip value per standard lot (approximate, USD account):
      Major pairs (EURUSD, GBPUSD, USDCHF): ~$10 / pip
      USDJPY                               : ~$9  / pip (close enough)
      XAUUSD (Gold)                        : ~$10 / 1 pt at 1 lot
    """
    risk_amount = balance * RISK_PCT

    if pair == "XAUUSD":
        pip_val_per_lot = 10.0    # $10 per 1.0 point per lot
    elif pair == "USDJPY":
        pip_val_per_lot = 9.0
    else:
        pip_val_per_lot = 10.0    # standard major

    raw_lot = risk_amount / (sl_pips * pip_val_per_lot)
    lot = round(max(0.01, min(raw_lot, 10.0)), 2)
    return lot


# ─────────────────────────────────────────────────────────────────────────────
#  ML SIGNAL
# ─────────────────────────────────────────────────────────────────────────────

def get_ml_signal(pair: str, tf: str, df) -> dict:
    """
    Run ML prediction. Returns:
      signal       : "BUY" | "SELL" | "WAIT"
      median_move  : pips
      entry        : current price
      tp           : take profit price
      sl           : stop loss price
      rr           : reward / risk ratio
    """
    try:
        model_key = f"{pair}_{tf}"
        median_prices, bull_prices, bear_prices, last_close = ml_predict(df, model_key)
    except FileNotFoundError:
        log.warning(f"[{pair}] No ML model for {tf} — run train.py first")
        return {"signal": "WAIT"}
    except Exception as e:
        log.error(f"[{pair}] ML predict error: {e}")
        return {"signal": "WAIT"}

    pip_factor  = PAIRS[pair]["pip_factor"]
    dec         = PAIRS[pair]["decimals"]

    median_end  = float(median_prices[-1])
    bull_end    = float(bull_prices[-1])
    bear_end    = float(bear_prices[-1])
    median_move = (median_end - last_close) * pip_factor   # pips (+/-)

    MIN_SL_PIPS = 8   # حداقل فاصله SL از entry

    if median_move > MIN_MOVE_PIPS:
        signal = "BUY"
        tp     = round(bull_end, dec)
        sl_raw = round(bear_end, dec)
        # SL باید زیر entry باشه برای BUY
        if sl_raw >= last_close:
            sl = round(last_close - MIN_SL_PIPS / pip_factor, dec)
        else:
            sl = sl_raw
    elif median_move < -MIN_MOVE_PIPS:
        signal = "SELL"
        tp     = round(bear_end, dec)
        sl_raw = round(bull_end, dec)
        # SL باید بالای entry باشه برای SELL
        if sl_raw <= last_close:
            sl = round(last_close + MIN_SL_PIPS / pip_factor, dec)
        else:
            sl = sl_raw
    else:
        return {"signal": "WAIT", "median_move": round(median_move, 1)}

    sl_pips = abs(last_close - sl) * pip_factor
    tp_pips = abs(last_close - tp) * pip_factor
    rr      = tp_pips / sl_pips if sl_pips > 0 else 0

    return {
        "signal":      signal,
        "entry":       round(last_close, dec),
        "tp":          tp,
        "sl":          sl,
        "sl_pips":     round(sl_pips, 1),
        "tp_pips":     round(tp_pips, 1),
        "median_move": round(median_move, 1),
        "rr":          round(rr, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  RL FILTER
# ─────────────────────────────────────────────────────────────────────────────

def get_rl_filter(pair: str, df) -> str:
    """
    Load RL agent and get its action for the current state.
    Returns "BUY", "SELL", "HOLD", or "SKIP" (if model not found).
    """
    model_path = os.path.join(os.path.dirname(__file__), "models", f"rl_{pair}_1h.npz")
    if not os.path.exists(model_path):
        return "SKIP"

    try:
        agent = DQNAgent.load(model_path, state_dim=15)

        df_feat     = add_features(df)
        state_feats = df_feat[FEATURE_COLS].values[-1]
        state       = np.array(list(state_feats) + [0.0, 0.0, 0.0], dtype=np.float32)
        sig         = agent.signal(state)
        return sig["action"]
    except Exception as e:
        log.warning(f"[{pair}] RL filter error: {e}")
        return "SKIP"


# ─────────────────────────────────────────────────────────────────────────────
#  BREAK-EVEN  — move SL to entry when price reaches 50% of TP
# ─────────────────────────────────────────────────────────────────────────────

BREAKEVEN_TRIGGER = 0.50   # move SL when 50% of TP distance reached

def check_breakeven():
    """
    For each open position, if price has moved 50% toward TP,
    move the SL to entry price (break-even).
    """
    positions = get_open_positions()
    for p in positions:
        ticket     = p["ticket"]
        entry      = p["open_price"]
        sl         = p["sl"]
        tp         = p["tp"]
        trade_type = p["type"]
        pair       = p["symbol"]
        dec        = PAIRS.get(pair, {}).get("decimals", 5)

        if tp == 0 or sl == 0:
            continue

        try:
            bid, ask = get_current_price(pair)
            price = bid if trade_type == "BUY" else ask
        except Exception:
            continue

        tp_distance    = abs(tp - entry)
        moved_distance = abs(price - entry)

        # Already at break-even or better
        if trade_type == "BUY" and sl >= entry:
            continue
        if trade_type == "SELL" and sl <= entry:
            continue

        # Check if 50% of TP distance reached
        if tp_distance == 0:
            continue
        progress = moved_distance / tp_distance

        if progress >= BREAKEVEN_TRIGGER:
            new_sl = round(entry, dec)
            result = modify_position_sl(ticket, new_sl=new_sl)
            if result.get("success"):
                log.info(f"[{pair}] Break-even activated #{ticket} — SL moved to {new_sl}")
            else:
                log.warning(f"[{pair}] Break-even failed #{ticket}: {result.get('message')}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLOSE DETECTION  — compare snapshot vs current positions each scan cycle
# ─────────────────────────────────────────────────────────────────────────────

def check_closed_positions():
    """
    Compare current MT5 positions against the previous snapshot.
    Any position that disappeared (TP hit, SL hit, manual close) sends
    a Telegram close notification.
    Returns the current positions dict so the caller can update the snapshot.
    """
    global _prev_positions

    current_raw  = get_open_positions()
    current_dict = {p["ticket"]: p for p in current_raw}

    for ticket, prev in _prev_positions.items():
        if ticket not in current_dict:
            # Position is gone — figure out rough P&L from last known price
            pair  = prev["symbol"].replace("m", "")   # strip micro suffix if any
            try:
                bid, ask = get_current_price(pair)
                if prev["type"] == "BUY":
                    close_price = bid
                else:
                    close_price = ask
            except Exception:
                close_price = 0.0

            label = PAIRS.get(pair, {}).get("display", pair)
            log.info(f"[{label}] Position #{ticket} closed | "
                     f"Entry: {prev['open_price']} -> Close: ~{close_price:.5f}")

            # Convert P&L to pips for public channel
            pip_factor = PAIRS.get(pair, {}).get("pip_factor", 10000)
            if prev["type"] == "BUY":
                pips_closed = round((close_price - prev["open_price"]) * pip_factor, 1)
            else:
                pips_closed = round((prev["open_price"] - close_price) * pip_factor, 1)

            # Accumulate for daily report
            _daily_closed.append({
                "pair":   pair,
                "action": prev["type"],
                "pips":   pips_closed,
            })

            try:
                notifier.notify_close(
                    pair        = pair,
                    action      = prev["type"],
                    entry       = prev["open_price"],
                    close_price = close_price,
                    pips        = pips_closed,
                )
            except Exception as e:
                log.warning(f"[Telegram] notify_close error: {e}")

    # NOTE: _prev_positions is updated in the main loop AFTER scanning,
    # so newly placed orders are captured in the snapshot too.
    return current_dict


# ─────────────────────────────────────────────────────────────────────────────
#  SCAN ONE PAIR
# ─────────────────────────────────────────────────────────────────────────────

def scan_pair(pair: str, tf: str, dry_run: bool = False):
    label = PAIRS[pair]["display"]

    # 1. Trading hours check
    if not is_trading_hour(pair):
        now = datetime.now(timezone.utc)
        log.info(f"[{label}] Outside trading hours ({now.strftime('%a %H:%M')} UTC) — skipping")
        return

    # 2. Check existing positions
    positions = get_open_positions(pair)
    if len(positions) >= MAX_POS_PER_PAIR:
        pnl = sum(p["profit"] for p in positions)
        log.info(f"[{label}] Already in {len(positions)} position(s) | P&L: {pnl:+.2f} — skipping")
        return

    # 3. News safety check — skip if high-impact event in next 2 hours
    try:
        if not is_safe_to_trade(pair, hours_ahead=2):
            log.info(f"[{label}] ⚠️  High-impact news in next 2h — skipping to protect position")
            return
    except Exception as e:
        log.warning(f"[{label}] News check failed ({e}) — proceeding anyway")

    # 4. Get live candles from MT5
    df = get_candles(pair, tf, n=CANDLES_TO_FETCH)
    if df.empty or len(df) < 100:
        log.warning(f"[{label}] Not enough candles ({len(df)}) — skipping")
        return

    # 5. ML signal
    ml = get_ml_signal(pair, tf, df)
    signal = ml["signal"]

    if signal == "WAIT":
        log.info(f"[{label}] ML → WAIT (move: {ml.get('median_move', 0):+.1f} pips)")
        return

    log.info(f"[{label}] ML → {signal} | "
             f"Move: {ml['median_move']:+.1f} pips | "
             f"SL: {ml['sl_pips']} pips | "
             f"TP: {ml['tp_pips']} pips | "
             f"R:R {ml['rr']}")

    # 6. R:R filter
    if ml["rr"] < MIN_RR:
        log.info(f"[{label}] R:R {ml['rr']} < {MIN_RR} minimum — skipping")
        return

    # 7. RL filter
    if USE_RL_FILTER:
        rl_action = get_rl_filter(pair, df)
        if rl_action == "SKIP":
            log.info(f"[{label}] RL model not found — proceeding without filter")
        elif rl_action != "HOLD" and rl_action != signal:
            log.info(f"[{label}] RL disagrees ({rl_action} vs ML {signal}) — SKIPPING trade")
            return
        else:
            log.info(f"[{label}] RL confirms: {rl_action} ✓")

    # 8. Lot sizing
    balance = get_balance()
    lot     = calc_lot(pair, ml["sl_pips"], balance)

    log.info(f"[{label}] → Placing {signal} | "
             f"Lot: {lot} | Entry: {ml['entry']} | "
             f"SL: {ml['sl']} | TP: {ml['tp']} | "
             f"Balance: {balance:.2f}")

    # 9. Place order
    if dry_run:
        log.info(f"[{label}] 🔵 DRY RUN — order NOT sent")
        return

    result = place_order(
        pair   = pair,
        action = signal,
        lot    = lot,
        sl     = ml["sl"],
        tp     = ml["tp"],
        comment = f"RadarFX-ML-{tf}",
    )

    if result["success"]:
        log.info(f"[{label}] ✅ Order placed! Ticket: {result['ticket']} @ {result['price']}")
        try:
            notifier.notify_open(
                pair    = pair,
                action  = signal,
                entry   = ml["entry"],
                sl      = ml["sl"],
                tp      = ml["tp"],
                lot     = lot,
                sl_pips = ml["sl_pips"],
                tp_pips = ml["tp_pips"],
                rr      = ml["rr"],
                df      = df,          # chart generation
            )
        except Exception as e:
            log.warning(f"[Telegram] notify_open error: {e}")
    else:
        log.error(f"[{label}] ❌ Order failed: {result['message']}")


# ─────────────────────────────────────────────────────────────────────────────
#  PRINT STATUS
# ─────────────────────────────────────────────────────────────────────────────

def print_status():
    acc = get_account()
    log.info(f"━━━ Account: {acc.get('balance', 0):.2f} {acc.get('currency','')} | "
             f"Equity: {acc.get('equity', 0):.2f} | "
             f"Free margin: {acc.get('free_margin', 0):.2f} ━━━")

    positions = get_open_positions()
    if positions:
        log.info(f"Open positions ({len(positions)}):")
        for p in positions:
            log.info(f"  #{p['ticket']}  {p['symbol']} {p['type']}  "
                     f"{p['volume']} lots  P&L: {p['profit']:+.2f}")
    else:
        log.info("No open positions.")


# ─────────────────────────────────────────────────────────────────────────────
#  DAILY REPORT  — fires once per day around 21:00 UTC
# ─────────────────────────────────────────────────────────────────────────────

def _maybe_send_daily_report(dry_run: bool):
    global _daily_closed, _last_report_date
    now = datetime.now(timezone.utc)

    # Send at 21:00 UTC, once per calendar day
    if now.hour != 21:
        return
    if now.timetuple().tm_yday == _last_report_date:
        return   # already sent today

    _last_report_date = now.timetuple().tm_yday
    balance   = get_balance()
    positions = get_open_positions()

    log.info(f"Sending daily report — {len(_daily_closed)} closed trade(s) today")

    if not dry_run:
        try:
            # Convert open positions to pip-based format
            open_pips = []
            for p in positions:
                pf  = PAIRS.get(p["symbol"], {}).get("pip_factor", 10000)
                if p["type"] == "BUY":
                    pip_running = round((p.get("profit", 0) / (p["volume"] * 10)), 1) if p["volume"] else 0
                else:
                    pip_running = round((p.get("profit", 0) / (p["volume"] * 10)), 1) if p["volume"] else 0
                open_pips.append({
                    "symbol": p["symbol"],
                    "type":   p["type"],
                    "pips":   pip_running,
                })
            notifier.notify_daily_report(_daily_closed, open_pips)
        except Exception as e:
            log.warning(f"[Telegram] daily report error: {e}")

    _daily_closed = []   # reset for the next day


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run(pairs: list, tf: str, dry_run: bool):
    log.info("═" * 60)
    log.info("  RadarFX Live Trader  —  ML + RL Powered")
    log.info(f"  Pairs     : {', '.join(pairs)}")
    log.info(f"  Timeframe : {tf}")
    log.info(f"  Risk/trade: {RISK_PCT*100:.0f}% of balance")
    log.info(f"  RL filter : {'ON' if USE_RL_FILTER else 'OFF'}")
    log.info(f"  Dry run   : {'YES — no real orders' if dry_run else 'NO — LIVE TRADING'}")
    log.info("═" * 60)

    if not connect():
        log.error("Cannot connect to MT5. Make sure terminal is open and logged in.")
        return

    try:
        notifier.notify_bot_start(pairs, tf)
    except Exception as e:
        log.warning(f"[Telegram] notify_bot_start error: {e}")

    try:
        while True:
            now = datetime.now(timezone.utc)
            log.info(f"\n{'─'*50}")
            log.info(f"Scan @ {now.strftime('%Y-%m-%d %H:%M')} UTC")
            log.info(f"{'─'*50}")

            # Check for closed positions before scanning new ones
            current_snapshot = {}
            if not dry_run:
                try:
                    current_snapshot = check_closed_positions()
                except Exception as e:
                    log.warning(f"Close-check error: {e}")

                # Move SL to break-even if 50% of TP reached
                try:
                    check_breakeven()
                except Exception as e:
                    log.warning(f"Break-even check error: {e}")

            # Daily report — send once at 21:00 UTC
            _maybe_send_daily_report(dry_run)

            print_status()

            for pair in pairs:
                try:
                    scan_pair(pair, tf, dry_run=dry_run)
                except Exception as e:
                    log.error(f"[{pair}] Unexpected error: {e}", exc_info=True)

            # Update position snapshot AFTER scanning so new orders are captured
            if not dry_run:
                try:
                    fresh = get_open_positions()
                    _prev_positions.update({p["ticket"]: p for p in fresh})
                    # also remove tickets no longer open
                    open_tickets = {p["ticket"] for p in fresh}
                    for t in list(_prev_positions.keys()):
                        if t not in open_tickets:
                            _prev_positions.pop(t, None)
                except Exception as e:
                    log.warning(f"Snapshot update error: {e}")

            next_scan = datetime.fromtimestamp(
                time.time() + CHECK_EVERY_SEC, tz=timezone.utc
            )
            log.info(f"\nNext scan: {next_scan.strftime('%H:%M')} UTC "
                     f"(in {CHECK_EVERY_SEC//60} min)")
            time.sleep(CHECK_EVERY_SEC)

    except KeyboardInterrupt:
        log.info("\n[RadarFX] Stopped by user.")
    finally:
        try:
            notifier.notify_bot_stop()
        except Exception:
            pass
        disconnect()


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RadarFX Live Trader")
    parser.add_argument("--pair", type=str, default=None,
                        help="Single pair to trade (default: all)")
    parser.add_argument("--tf",   type=str, default=DEFAULT_TF,
                        help="Timeframe: 15m 30m 1h 4h daily (default: 1h)")
    parser.add_argument("--dry",  action="store_true",
                        help="Dry run — analyse only, no real orders")
    args = parser.parse_args()

    pairs = [args.pair.upper()] if args.pair else PAIRS_TO_TRADE

    run(pairs=pairs, tf=args.tf, dry_run=args.dry)
