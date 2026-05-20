"""
RadarFX — notifier.py
Telegram signal channel — public format (no balance/dollars, pips only).
Sends chart image + clean signal card.
"""

import io
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime, timezone

BOT_TOKEN  = "8874736945:AAF-1g5U4m9jNIBQig9KaFQIYjcoehP5n5I"
CHANNEL_ID = "@RadarFXSignal"


# ─── Core senders ─────────────────────────────────────────────────────────────

def _send(text: str):
    if not CHANNEL_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, json={
            "chat_id":    CHANNEL_ID,
            "text":       text,
            "parse_mode": "HTML",
        }, timeout=10)
    except Exception as e:
        print(f"[Telegram] send failed: {e}")


def _send_photo(img_bytes: bytes, caption: str):
    if not CHANNEL_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
        requests.post(url,
            data={"chat_id": CHANNEL_ID, "caption": caption,
                  "parse_mode": "HTML"},
            files={"photo": ("chart.png", img_bytes, "image/png")},
            timeout=15,
        )
    except Exception as e:
        print(f"[Telegram] send_photo failed: {e}")


# ─── Chart generator ──────────────────────────────────────────────────────────

def _make_chart(df, pair: str, action: str,
                entry: float, sl: float, tp: float) -> bytes:
    """
    Generates a clean dark candlestick chart with entry/SL/TP lines.
    Returns PNG bytes.
    """
    candles = df.tail(60).copy().reset_index()
    n = len(candles)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    # ── Candles ───────────────────────────────────────────────────────────────
    for i, row in candles.iterrows():
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        color = "#26a69a" if c >= o else "#ef5350"   # teal / red
        # Body
        ax.bar(i, abs(c - o), bottom=min(o, c), color=color,
               width=0.6, linewidth=0)
        # Wick
        ax.plot([i, i], [l, h], color=color, linewidth=0.8)

    # ── Entry / SL / TP lines ─────────────────────────────────────────────────
    x_end = n + 4
    ax.hlines(entry, -1, x_end, colors="#ffffff", linewidths=1.2,
              linestyles="--", label=f"Entry {entry}")
    ax.hlines(sl,    -1, x_end, colors="#ef5350", linewidths=1.2,
              linestyles=":",  label=f"SL {sl}")
    ax.hlines(tp,    -1, x_end, colors="#26a69a", linewidths=1.2,
              linestyles=":",  label=f"TP {tp}")

    # Labels on the right
    ax.text(x_end + 0.2, entry, f" {entry}", va="center",
            color="#ffffff", fontsize=8)
    ax.text(x_end + 0.2, sl,    f" {sl}",    va="center",
            color="#ef5350", fontsize=8)
    ax.text(x_end + 0.2, tp,    f" {tp}",    va="center",
            color="#26a69a", fontsize=8)

    # ── Title & style ─────────────────────────────────────────────────────────
    direction = "LONG ▲" if action == "BUY" else "SHORT ▼"
    ax.set_title(f"{pair}  —  {direction}", color="#ffffff",
                 fontsize=13, fontweight="bold", pad=10)

    ax.tick_params(colors="#555555", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#222222")
    ax.set_xlim(-1, x_end + 6)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    ax.text(0.01, 0.02, "RadarFX ML Signal",
            transform=ax.transAxes, color="#333333",
            fontsize=7, va="bottom")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─── Trade opened ─────────────────────────────────────────────────────────────

def notify_open(pair: str, action: str, entry: float, sl: float,
                tp: float, lot: float, sl_pips: float, tp_pips: float,
                rr: float, df=None):

    direction = "BUY  📈" if action == "BUY" else "SELL  📉"
    arrow_sl  = "▲" if action == "BUY" else "▼"   # SL is below for BUY
    dash      = "—" * 20

    caption = (
        f"📊 <b>{pair}  ·  Instant Order</b>\n"
        f"\n"
        f"{'▼' if action=='SELL' else '▲'} <b>{pair} {action.lower()} now at</b>  {dash}  <code>{entry}</code>\n"
        f"💎 SL set at  {dash}  <code>{sl}</code> ❌  <i>({sl_pips:.0f} pips)</i>\n"
        f"💎 TP set at  {dash}  <code>{tp}</code> ✅  <i>({tp_pips:.0f} pips)</i>\n"
        f"\n"
        f"R : R   <b>1 : {rr}</b>\n"
        f"⚠️ Manage your own risk — max 1–2% per trade"
    )

    if df is not None and not df.empty:
        try:
            img = _make_chart(df, pair, action, entry, sl, tp)
            _send_photo(img, caption)
            return
        except Exception as e:
            print(f"[Chart] generation failed: {e}")

    # Fallback — text only
    _send(caption)


# ─── Trade closed ─────────────────────────────────────────────────────────────

def notify_close(pair: str, action: str, entry: float, close_price: float,
                 pips: float):
    """pips > 0 = win, pips < 0 = loss"""

    if pips > 0:
        emoji  = "✅"
        result = f"+{pips:.0f} pips"
    elif pips < 0:
        emoji  = "❌"
        result = f"{pips:.0f} pips"
    else:
        emoji  = "➖"
        result = "0 pips"

    direction = "Long" if action == "BUY" else "Short"

    text = (
        f"{emoji} <b>{pair}  {result}</b>\n"
        f"\n"
        f"{direction}  <code>{entry}</code> → <code>{close_price}</code>"
    )
    _send(text)


# ─── Bot lifecycle ────────────────────────────────────────────────────────────

def notify_bot_start(pairs: list, tf: str):
    pairs_str = "  ·  ".join(pairs)
    _send(f"👁  Watching  {pairs_str}")


def notify_bot_stop():
    _send("💤  RadarFX going offline")


# ─── Daily summary ────────────────────────────────────────────────────────────

def notify_daily_report(closed_trades: list, open_positions: list):
    """
    closed_trades : list of dicts  {pair, action, pips}
    open_positions: list of dicts  {symbol, type, pips}
    """
    today = datetime.now(timezone.utc).strftime("%d %b")
    lines = [f"📊 <b>Daily Pips Update — {today}</b>\n"]

    if closed_trades:
        total = sum(t["pips"] for t in closed_trades)
        for t in closed_trades:
            e   = "✅" if t["pips"] >= 0 else "❌"
            pip = f"+{t['pips']:.0f}" if t["pips"] >= 0 else f"{t['pips']:.0f}"
            act = t["action"].lower()
            lines.append(f"{t['pair']}  {act}   {pip} pips {e}")

        lines.append("")
        lines.append("·" * 28)

        sign = "+" if total >= 0 else ""
        fire = "🔥" if total > 0 else "📉"
        lines.append(f"{fire} Result Of The Day {fire}")
        lines.append("")
        if total >= 0:
            lines.append(f"Total Gain:  <b>+{total:.0f} Pips ✅</b>")
        else:
            lines.append(f"Total Loss:  <b>{total:.0f} Pips ❌</b>")
    else:
        lines.append("No trades closed today.")

    if open_positions:
        lines.append("")
        lines.append("Still running:")
        for p in open_positions:
            pips = p.get("pips", 0.0)
            e    = "🔵" if pips >= 0 else "🟡"
            pip  = f"+{pips:.0f}" if pips >= 0 else f"{pips:.0f}"
            lines.append(f"  {e}  {p.get('symbol','')}  {p.get('type','')}  {pip} pips")

    _send("\n".join(lines))
