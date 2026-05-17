"""
RadarFX — rl_env.py
Forex trading environment for DQN reinforcement learning.
Pure numpy/pandas — no extra dependencies.

Actions: 0=HOLD  1=BUY (open/stay long)  2=SELL (open/stay short)
State  : 12 market features + 3 position features = 15 dims
Reward : realized pips P&L with spread cost deducted
"""

import numpy as np
import pandas as pd
from ml_model import add_features, FEATURE_COLS, LOOKBACK


class ForexTradingEnv:
    """
    Simulates a simple forex account: one position at a time (flat / long / short).

    State vector (15 dims):
        [0:12]  market features from ml_model (ret1 … bb_width)
        [12]    position  (–1 = short, 0 = flat, +1 = long)
        [13]    unrealized PnL as fraction of entry price
        [14]    steps_in_trade / 50  (capped at 1.0)

    Reward: pips realised when a trade closes (minus spread).
            Small negative each step while in an open trade to
            discourage infinite holding.
    """

    HOLD = 0
    BUY  = 1   # open long, or close short
    SELL = 2   # open short, or close long

    N_ACTIONS = 3
    STATE_DIM = len(FEATURE_COLS) + 3   # 12 + 3 = 15

    def __init__(self, df: pd.DataFrame,
                 pip_factor: float = 10_000,
                 spread_pips: float = 2.0,
                 hold_penalty: float = 0.05,
                 max_steps_per_trade: int = 40):
        """
        Args
        ----
        df                 : DataFrame already processed by add_features()
        pip_factor         : price distance → pips  (10000 for forex, 10 for gold)
        spread_pips        : round-trip cost charged when opening a trade
        hold_penalty       : tiny pip penalty per step while in a trade
        max_steps_per_trade: force-close if held longer than this
        """
        self.df           = df.reset_index(drop=True)
        self.pip_factor   = pip_factor
        self.spread       = spread_pips
        self.hold_penalty = hold_penalty
        self.max_hold     = max_steps_per_trade

        self.feats  = df[FEATURE_COLS].values.astype(np.float32)
        self.closes = df["close"].values.astype(np.float32)
        self.n      = len(df)

        self._reset_internals()

    # ──────────────────────────────────────────────────────────────────────────
    def _reset_internals(self):
        self.idx            = LOOKBACK
        self.position       = 0        # 0 flat | +1 long | −1 short
        self.entry_price    = 0.0
        self.steps_in_trade = 0
        self.total_pips     = 0.0
        self.n_trades       = 0
        self.done           = False

    def reset(self) -> np.ndarray:
        self._reset_internals()
        return self._obs()

    # ──────────────────────────────────────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        feat  = self.feats[self.idx].copy()          # (12,)
        price = self.closes[self.idx]

        unreal = 0.0
        if self.position != 0 and self.entry_price > 0:
            unreal = self.position * (price - self.entry_price) / (self.entry_price + 1e-10)

        extra = np.array([
            float(self.position),
            float(np.clip(unreal, -1.0, 1.0)),
            float(min(self.steps_in_trade / 50.0, 1.0)),
        ], dtype=np.float32)

        return np.concatenate([feat, extra])

    # ──────────────────────────────────────────────────────────────────────────
    def step(self, action: int):
        """Returns (next_obs, reward, done)."""
        price  = self.closes[self.idx]
        reward = 0.0

        # ── Execute action ────────────────────────────────────────────────
        if action == self.BUY:
            if self.position == 0:                      # open long
                self.position    = 1
                self.entry_price = price
                self.steps_in_trade = 0
                reward = -self.spread                   # pay spread

            elif self.position == -1:                   # close short → realise
                pnl = (self.entry_price - price) * self.pip_factor
                reward = pnl - self.spread
                self._close_trade(pnl)

            # already long → do nothing (treat as HOLD)

        elif action == self.SELL:
            if self.position == 0:                      # open short
                self.position    = -1
                self.entry_price = price
                self.steps_in_trade = 0
                reward = -self.spread

            elif self.position == 1:                    # close long → realise
                pnl = (price - self.entry_price) * self.pip_factor
                reward = pnl - self.spread
                self._close_trade(pnl)

            # already short → do nothing

        # HOLD: small penalty while in a trade to discourage over-holding
        if self.position != 0:
            reward -= self.hold_penalty
            self.steps_in_trade += 1

            # Force-close if held too long
            if self.steps_in_trade >= self.max_hold:
                if self.position == 1:
                    pnl = (price - self.entry_price) * self.pip_factor
                else:
                    pnl = (self.entry_price - price) * self.pip_factor
                reward += pnl - self.spread
                self._close_trade(pnl)

        # ── Advance bar ───────────────────────────────────────────────────
        self.idx += 1
        done = (self.idx >= self.n - 1)

        if done and self.position != 0:
            last = self.closes[-1]
            if self.position == 1:
                pnl = (last - self.entry_price) * self.pip_factor
            else:
                pnl = (self.entry_price - last) * self.pip_factor
            reward += pnl - self.spread
            self._close_trade(pnl)

        return self._obs(), reward, done

    def _close_trade(self, pnl: float):
        self.total_pips     += pnl
        self.n_trades       += 1
        self.position        = 0
        self.entry_price     = 0.0
        self.steps_in_trade  = 0

    # ──────────────────────────────────────────────────────────────────────────
    @property
    def avg_pips(self) -> float:
        return self.total_pips / max(self.n_trades, 1)


# ── Quick sanity test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_feed import fetch_candles

    df_raw = fetch_candles("1h", pair="EURUSD")
    df     = add_features(df_raw)
    env    = ForexTradingEnv(df)
    obs    = env.reset()
    print(f"State dim : {len(obs)}  (expected {ForexTradingEnv.STATE_DIM})")
    print(f"Obs sample: {obs[:5]}")

    # Random policy episode
    done = False
    while not done:
        a = np.random.randint(0, 3)
        obs, r, done = env.step(a)
    print(f"Random policy: total_pips={env.total_pips:+.1f}  trades={env.n_trades}")
    print("rl_env.py ✓")
