"""
RadarFX — news_feed.py
Fetches economic calendar events via Finnhub API.

Setup:
  Key is stored in backend/.env file as FINNHUB_KEY=...
  Get free key at: https://finnhub.io

Usage:
  python news_feed.py              # next 48h events
  python news_feed.py --hours 72   # next 3 days
  python news_feed.py --pair EURUSD --check  # safe to trade?
"""

import requests
import os
import json
from datetime import datetime, timezone, timedelta
import argparse

# ─── Load API Key ──────────────────────────────────────────────────────────────
def _load_key() -> str:
    # Try environment variable first
    key = os.environ.get("FINNHUB_KEY", "")
    if key:
        return key
    # Try .env file
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("FINNHUB_KEY="):
                    return line.strip().split("=", 1)[1]
    return ""

FINNHUB_KEY = _load_key()

# ─── Pair → Currency mapping ──────────────────────────────────────────────────
PAIR_CURRENCIES = {
    "EURUSD": ["EUR", "USD"],
    "GBPUSD": ["GBP", "USD"],
    "XAUUSD": ["USD"],
    "USDCHF": ["USD", "CHF"],
    "USDJPY": ["USD", "JPY"],
}

CACHE_FILE = os.path.join(os.path.dirname(__file__), ".news_cache.json")
CACHE_TTL  = 30 * 60   # 30 minutes cache


# ─── Cache ────────────────────────────────────────────────────────────────────
def _load_cache(key: str):
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE) as f:
            data = json.load(f)
        if data.get("key") != key:
            return None
        age = datetime.now().timestamp() - data.get("fetched_at", 0)
        if age < CACHE_TTL:
            return data.get("events", [])
    except Exception:
        pass
    return None


def _save_cache(events: list, key: str):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump({
                "fetched_at": datetime.now().timestamp(),
                "key": key,
                "events": events
            }, f)
    except Exception:
        pass


# ─── Fetch from Finnhub ───────────────────────────────────────────────────────
def fetch_calendar(days_ahead: int = 7) -> list:
    """Fetch economic calendar from Finnhub for next N days."""
    if not FINNHUB_KEY:
        print("  [news] No Finnhub API key found. Check backend/.env")
        return []

    cache_key = f"{days_ahead}"
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    now      = datetime.now(timezone.utc)
    date_from = now.strftime("%Y-%m-%d")
    date_to   = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    url = (f"https://finnhub.io/api/v1/calendar/economic"
           f"?from={date_from}&to={date_to}&token={FINNHUB_KEY}")

    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            events = data.get("economicCalendar", [])
            _save_cache(events, cache_key)
            return events
        else:
            print(f"  [news] Finnhub error: {r.status_code}")
            return []
    except Exception as e:
        print(f"  [news] Could not fetch calendar: {e}")
        return []


# ─── Filter & Format ──────────────────────────────────────────────────────────
def get_upcoming_events(hours_ahead: int = 48, pair: str = None) -> list:
    """Get high/medium impact events in the next N hours."""
    raw      = fetch_calendar(days_ahead=max(hours_ahead // 24 + 1, 7))
    now      = datetime.now(timezone.utc)
    deadline = now + timedelta(hours=hours_ahead)

    currencies = None
    if pair and pair.upper() in PAIR_CURRENCIES:
        currencies = PAIR_CURRENCIES[pair.upper()]

    results = []
    for ev in raw:
        impact = str(ev.get("impact", "")).lower()
        if impact not in ("high", "medium", "1", "2", "3"):
            continue

        # Parse time
        try:
            dt_str = ev.get("time", ev.get("date", ""))
            if not dt_str:
                continue
            # Finnhub format: "2026-05-16 14:30:00" or "2026-05-16T14:30:00Z"
            dt_str = dt_str.replace("T", " ").replace("Z", "").strip()
            if len(dt_str) == 10:  # only date, no time
                dt_str += " 00:00:00"
            dt = datetime.strptime(dt_str[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except Exception:
            continue

        if not (now <= dt <= deadline):
            continue

        # Filter by currency
        ev_currency = str(ev.get("country", ev.get("currency", ""))).upper()
        if currencies and ev_currency not in currencies:
            continue

        # Normalize impact label
        impact_label = "high" if impact in ("high", "1") else "medium"

        minutes_away = int((dt - now).total_seconds() / 60)
        results.append({
            "time":         dt.strftime("%Y-%m-%d %H:%M UTC"),
            "currency":     ev_currency,
            "event":        ev.get("event", ev.get("title", "Unknown")),
            "impact":       impact_label,
            "minutes_away": minutes_away,
            "forecast":     ev.get("estimate", ev.get("forecast", "—")),
            "previous":     ev.get("prev", ev.get("previous", "—")),
            "unit":         ev.get("unit", ""),
        })

    results.sort(key=lambda x: x["minutes_away"])
    return results


def is_safe_to_trade(pair: str, hours_ahead: int = 2) -> dict:
    """Check if safe to trade — no high-impact news within N hours."""
    events = get_upcoming_events(hours_ahead=hours_ahead, pair=pair)
    high   = [e for e in events if e["impact"] == "high"]

    if high:
        c = high[0]
        return {
            "safe":   False,
            "reason": f"⛔ High-impact news in {c['minutes_away']}min: {c['event']} ({c['currency']})",
            "events": high,
        }
    return {
        "safe":   True,
        "reason": f"✅ No high-impact news in next {hours_ahead}h — clear to trade",
        "events": events,
    }


# ─── Display ──────────────────────────────────────────────────────────────────
def print_events(events: list, title: str = "Upcoming Events"):
    print(f"\n  {'─'*58}")
    print(f"  {title}")
    print(f"  {'─'*58}")
    if not events:
        print("  No events found in this window.")
    for e in events:
        icon = "🔴" if e["impact"] == "high" else "🟡"
        mins = e["minutes_away"]
        time_str = f"+{mins}min" if mins < 60 else f"+{mins//60}h{mins%60:02d}m"
        print(f"  {icon} {e['time']}  ({time_str:>8})  "
              f"{e['currency']:4}  {e['event']}")
        if e["forecast"] != "—":
            print(f"       Forecast: {e['forecast']}  |  Previous: {e['previous']}")
    print(f"  {'─'*58}")


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RadarFX News Feed")
    p.add_argument("--hours", type=int, default=48)
    p.add_argument("--pair",  type=str, default=None)
    p.add_argument("--check", action="store_true",
                   help="Check if safe to trade right now")
    args = p.parse_args()

    print(f"\n{'═'*58}")
    print(f"  RadarFX Economic Calendar  (Finnhub)")
    print(f"{'═'*58}")

    if args.check and args.pair:
        result = is_safe_to_trade(args.pair, hours_ahead=2)
        print(f"\n  {result['reason']}\n")
        if result["events"]:
            print_events(result["events"], "Relevant Upcoming Events")
    else:
        events = get_upcoming_events(hours_ahead=args.hours, pair=args.pair)
        title  = f"Next {args.hours}h Events"
        if args.pair:
            title += f"  ({args.pair})"
        print_events(events, title)
