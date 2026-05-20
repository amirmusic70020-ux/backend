"""
RadarFX — mt5_bridge.py
Low-level MetaTrader 5 interface: connect, fetch candles, place/close orders.
Requires MT5 terminal to be running and logged in.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ─── MT5 Timeframe map ────────────────────────────────────────────────────────
MT5_TIMEFRAMES = {
    "5m":    mt5.TIMEFRAME_M5,
    "15m":   mt5.TIMEFRAME_M15,
    "30m":   mt5.TIMEFRAME_M30,
    "1h":    mt5.TIMEFRAME_H1,
    "4h":    mt5.TIMEFRAME_H4,
    "daily": mt5.TIMEFRAME_D1,
}

# ─── FXTM symbol names (may need suffix like 'm' for micro accounts) ──────────
SYMBOL_MAP = {
    "EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD",
    "XAUUSD": "XAUUSD",
    "USDCHF": "USDCHF",
    "USDJPY": "USDJPY",
}


# ─── Connection ───────────────────────────────────────────────────────────────

def connect() -> bool:
    """
    Connect to the running MT5 terminal.
    MT5 must already be open and logged in — no credentials needed.
    """
    if not mt5.initialize():
        print(f"[MT5] ❌ Failed to connect: {mt5.last_error()}")
        return False

    info = mt5.account_info()
    if info is None:
        print(f"[MT5] ❌ No account found: {mt5.last_error()}")
        return False

    print(f"[MT5] ✅ Connected:")
    print(f"       Account : {info.login} ({info.name})")
    print(f"       Broker  : {info.company}")
    print(f"       Balance : {info.balance:.2f} {info.currency}")
    print(f"       Leverage: 1:{info.leverage}")
    return True


def disconnect():
    mt5.shutdown()
    print("[MT5] Disconnected.")


# ─── Account info ─────────────────────────────────────────────────────────────

def get_account() -> dict:
    info = mt5.account_info()
    if info is None:
        return {}
    return {
        "login":    info.login,
        "balance":  info.balance,
        "equity":   info.equity,
        "margin":   info.margin,
        "free_margin": info.margin_free,
        "currency": info.currency,
        "leverage": info.leverage,
    }


def get_balance() -> float:
    info = mt5.account_info()
    return info.balance if info else 0.0


# ─── Live candles from MT5 ────────────────────────────────────────────────────

def get_candles(pair: str, tf: str = "1h", n: int = 300) -> pd.DataFrame:
    """
    Fetch the last N OHLCV candles for a pair directly from MT5.
    Returns DataFrame with columns: open, high, low, close, volume
    """
    symbol = SYMBOL_MAP.get(pair.upper(), pair.upper())
    tf_mt5 = MT5_TIMEFRAMES.get(tf)
    if tf_mt5 is None:
        print(f"[MT5] Unknown timeframe: {tf}")
        return pd.DataFrame()

    # Make sure symbol is visible in MarketWatch
    if not mt5.symbol_select(symbol, True):
        print(f"[MT5] Symbol not found: {symbol}")
        return pd.DataFrame()

    rates = mt5.copy_rates_from_pos(symbol, tf_mt5, 0, n)
    if rates is None or len(rates) == 0:
        print(f"[MT5] No candles for {symbol} {tf}: {mt5.last_error()}")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df.dropna()
    return df


def get_current_price(pair: str) -> tuple:
    """Returns (bid, ask) for a pair."""
    symbol = SYMBOL_MAP.get(pair.upper(), pair.upper())
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return 0.0, 0.0
    return tick.bid, tick.ask


# ─── Open positions ───────────────────────────────────────────────────────────

def get_open_positions(pair: str = None) -> list:
    """
    Returns list of open position dicts.
    Filter by pair if specified.
    """
    if pair:
        symbol = SYMBOL_MAP.get(pair.upper(), pair.upper())
        positions = mt5.positions_get(symbol=symbol)
    else:
        positions = mt5.positions_get()

    if positions is None:
        return []

    result = []
    for p in positions:
        result.append({
            "ticket":  p.ticket,
            "symbol":  p.symbol,
            "type":    "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
            "volume":  p.volume,
            "open_price": p.price_open,
            "sl":      p.sl,
            "tp":      p.tp,
            "profit":  p.profit,
            "comment": p.comment,
            "time":    datetime.fromtimestamp(p.time, tz=timezone.utc),
        })
    return result


# ─── Place order ──────────────────────────────────────────────────────────────

def place_order(pair: str, action: str, lot: float,
                sl: float, tp: float,
                comment: str = "RadarFX") -> dict:
    """
    Place a market order.
    action: "BUY" or "SELL"
    Returns dict with success, ticket, message.
    """
    symbol = SYMBOL_MAP.get(pair.upper(), pair.upper())
    tick   = mt5.symbol_info_tick(symbol)
    if tick is None:
        return {"success": False, "message": f"No tick for {symbol}"}

    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        return {"success": False, "message": f"No symbol info for {symbol}"}

    # Round lot to symbol's volume step
    vol_step = sym_info.volume_step
    lot = round(round(lot / vol_step) * vol_step, 2)
    lot = max(sym_info.volume_min, min(lot, sym_info.volume_max))

    if action.upper() == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
        price      = tick.ask
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price      = tick.bid

    # Round SL/TP to symbol's digits
    digits = sym_info.digits
    sl = round(sl, digits)
    tp = round(tp, digits)

    request = {
        "action":      mt5.TRADE_ACTION_DEAL,
        "symbol":      symbol,
        "volume":      lot,
        "type":        order_type,
        "price":       price,
        "sl":          sl,
        "tp":          tp,
        "deviation":   20,          # max slippage in points
        "magic":       20250518,    # RadarFX magic number
        "comment":     comment,
        "type_time":   mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    result = mt5.order_send(request)

    if result is None:
        return {"success": False, "message": str(mt5.last_error())}

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        return {
            "success": True,
            "ticket":  result.order,
            "price":   result.price,
            "message": "Order placed successfully",
        }
    else:
        return {
            "success": False,
            "retcode": result.retcode,
            "message": result.comment,
        }


# ─── Modify SL/TP ────────────────────────────────────────────────────────────

def modify_position_sl(ticket: int, new_sl: float, new_tp: float = None) -> dict:
    """Move SL (and optionally TP) for an open position."""
    positions = mt5.positions_get()
    if positions is None:
        return {"success": False, "message": "No positions"}

    pos = next((p for p in positions if p.ticket == ticket), None)
    if pos is None:
        return {"success": False, "message": f"Ticket {ticket} not found"}

    sym_info = mt5.symbol_info(pos.symbol)
    digits   = sym_info.digits if sym_info else 5
    tp_use   = round(new_tp, digits) if new_tp is not None else pos.tp

    request = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "symbol":   pos.symbol,
        "position": ticket,
        "sl":       round(new_sl, digits),
        "tp":       tp_use,
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        return {"success": True, "ticket": ticket}
    return {"success": False, "message": result.comment if result else "Failed"}


# ─── Partial close ────────────────────────────────────────────────────────────

def partial_close_position(ticket: int, close_ratio: float = 0.5) -> dict:
    """Close a fraction of an open position. Default: 50%."""
    positions = mt5.positions_get()
    if positions is None:
        return {"success": False, "message": "No positions"}

    pos = next((p for p in positions if p.ticket == ticket), None)
    if pos is None:
        return {"success": False, "message": f"Ticket {ticket} not found"}

    sym_info = mt5.symbol_info(pos.symbol)
    if sym_info is None:
        return {"success": False, "message": "No symbol info"}

    vol_step = sym_info.volume_step
    vol_min  = sym_info.volume_min
    close_vol = round(pos.volume * close_ratio / vol_step) * vol_step
    close_vol = round(max(vol_min, close_vol), 2)

    if close_vol >= pos.volume:
        close_vol = round(pos.volume / 2, 2)

    tick = mt5.symbol_info_tick(pos.symbol)
    if pos.type == mt5.ORDER_TYPE_BUY:
        close_type = mt5.ORDER_TYPE_SELL
        price      = tick.bid
    else:
        close_type = mt5.ORDER_TYPE_BUY
        price      = tick.ask

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       pos.symbol,
        "volume":       close_vol,
        "type":         close_type,
        "position":     ticket,
        "price":        price,
        "deviation":    20,
        "magic":        20250518,
        "comment":      "RadarFX partial",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        return {"success": True, "ticket": ticket, "closed_vol": close_vol}
    return {"success": False, "message": result.comment if result else "Failed"}


# ─── Close position ───────────────────────────────────────────────────────────

def close_position(ticket: int) -> dict:
    """Close a position by ticket number."""
    positions = mt5.positions_get()
    if positions is None:
        return {"success": False, "message": "No positions"}

    pos = next((p for p in positions if p.ticket == ticket), None)
    if pos is None:
        return {"success": False, "message": f"Ticket {ticket} not found"}

    symbol = pos.symbol
    tick   = mt5.symbol_info_tick(symbol)

    if pos.type == mt5.ORDER_TYPE_BUY:
        close_type = mt5.ORDER_TYPE_SELL
        price      = tick.bid
    else:
        close_type = mt5.ORDER_TYPE_BUY
        price      = tick.ask

    request = {
        "action":      mt5.TRADE_ACTION_DEAL,
        "symbol":      symbol,
        "volume":      pos.volume,
        "type":        close_type,
        "position":    ticket,
        "price":       price,
        "deviation":   20,
        "magic":       20250518,
        "comment":     "RadarFX close",
        "type_time":   mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        return {"success": True, "ticket": ticket, "profit": pos.profit}
    return {"success": False, "message": result.comment if result else "Failed"}


def close_all_positions(pair: str = None) -> list:
    """Close all open positions (optionally filtered by pair)."""
    positions = get_open_positions(pair)
    results = []
    for p in positions:
        r = close_position(p["ticket"])
        results.append(r)
    return results


# ─── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if connect():
        acc = get_account()
        print(f"\nAccount: {acc}")

        print("\nFetching EUR/USD 1h candles...")
        df = get_candles("EURUSD", "1h", n=5)
        print(df.tail())

        bid, ask = get_current_price("EURUSD")
        print(f"\nEUR/USD  Bid: {bid}  Ask: {ask}")

        positions = get_open_positions()
        print(f"\nOpen positions: {len(positions)}")
        for p in positions:
            print(f"  {p['symbol']} {p['type']} {p['volume']} lots | P&L: {p['profit']:.2f}")

        disconnect()
