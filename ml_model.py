"""
RadarFX ML — ml_model.py  (scikit-learn edition)
Gradient Boosting with Quantile Regression for 3-line price forecasting.

3 prediction lines:
  Blue  (bull)   = 85th percentile
  Green (median) = 50th percentile
  Red   (bear)   = 15th percentile

No TensorFlow / PyTorch required.
"""

import os
import pickle
import numpy as np
import pandas as pd

# ─── Constants ────────────────────────────────────────────────────────────────
LOOKBACK   = 50     # past candles used as input
FORECAST   = 20     # future candles predicted
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "models")

FEATURE_COLS = [
    "ret1", "ret5", "ret10",
    "hl_ratio", "oc_ratio",
    "ma5_dev", "ma20_dev", "ma50_dev",
    "rsi", "atr_norm", "bb_pos", "bb_width",
]


# ─── Feature Engineering ──────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]

    df["ret1"]  = close.pct_change(1)
    df["ret5"]  = close.pct_change(5)
    df["ret10"] = close.pct_change(10)

    df["hl_ratio"] = (df["high"] - df["low"]) / (close + 1e-10)
    df["oc_ratio"] = (close - df["open"])     / (close + 1e-10)

    df["ma5_dev"]  = close.rolling(5).mean()  / close - 1
    df["ma20_dev"] = close.rolling(20).mean() / close - 1
    df["ma50_dev"] = close.rolling(50).mean() / close - 1

    delta     = close.diff()
    gain      = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss      = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs        = gain / loss.replace(0, np.nan)
    df["rsi"] = (100 - 100 / (1 + rs)) / 100

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - close.shift()).abs(),
        (df["low"]  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr_norm"] = tr.ewm(alpha=1/14, adjust=False).mean() / (close + 1e-10)

    bb_mid        = close.rolling(20).mean()
    bb_std        = close.rolling(20).std()
    bb_range      = (4 * bb_std).replace(0, np.nan)
    df["bb_pos"]  = (close - bb_mid) / (bb_range / 2 + 1e-10)
    df["bb_width"]= bb_range / (bb_mid + 1e-10)

    df.dropna(inplace=True)
    return df


# ─── Sequence Builder ─────────────────────────────────────────────────────────

def make_sequences(df: pd.DataFrame):
    """
    X: flattened window of LOOKBACK * n_features
    y: list of FORECAST future returns (one model per step)
    """
    feats  = df[FEATURE_COLS].values.astype(np.float32)
    closes = df["close"].values.astype(np.float32)

    X, Y = [], []
    for i in range(LOOKBACK, len(df) - FORECAST):
        x_flat = feats[i - LOOKBACK : i].flatten()
        future_ret = closes[i : i + FORECAST] / closes[i] - 1
        X.append(x_flat)
        Y.append(future_ret)

    return np.array(X), np.array(Y)


# ─── Train ────────────────────────────────────────────────────────────────────

def train(df: pd.DataFrame, pair: str = "EURUSD") -> None:
    """
    Train 3 GradientBoosting models per forecast step (quantile 15/50/85).
    Saves to models/<pair>_q{quantile}_step{k}.pkl
    """
    from sklearn.ensemble import GradientBoostingRegressor

    print(f"\n  Preparing features...")
    df_feat = add_features(df)
    X, Y    = make_sequences(df_feat)

    print(f"  Sequences: {len(X)}  |  Features: {X.shape[1]}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    quantiles = [("bear", 0.15), ("median", 0.50), ("bull", 0.85)]

    for q_name, q_val in quantiles:
        models_this_q = []
        print(f"\n  Training {q_name} models (q={q_val})...")

        for step in range(FORECAST):
            y_step = Y[:, step]

            model = GradientBoostingRegressor(
                loss        = "quantile",
                alpha       = q_val,
                n_estimators= 100,
                max_depth   = 3,
                learning_rate=0.1,
                subsample   = 0.8,
                min_samples_leaf=10,
                random_state= 42,
            )
            model.fit(X, y_step)
            models_this_q.append(model)

            if (step + 1) % 5 == 0:
                print(f"    Step {step+1}/{FORECAST} done")

        # Save all steps for this quantile
        save_path = os.path.join(MODEL_DIR, f"{pair}_{q_name}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(models_this_q, f)
        print(f"  [OK] Saved -> {save_path}")

    print(f"\n  [OK] Training complete for {pair}")


# ─── Predict ──────────────────────────────────────────────────────────────────

def predict(df: pd.DataFrame, pair: str = "EURUSD") -> tuple:
    """
    Returns (median_prices, bull_prices, bear_prices, last_close)
    Each array has length FORECAST.
    """
    df_feat    = add_features(df)
    last_close = float(df_feat["close"].iloc[-1])

    if len(df_feat) < LOOKBACK:
        raise ValueError(f"Need ≥{LOOKBACK} candles, got {len(df_feat)}")

    x_flat = df_feat[FEATURE_COLS].values[-LOOKBACK:].flatten().reshape(1, -1)

    results = {}
    for q_name in ["bear", "median", "bull"]:
        model_path = os.path.join(MODEL_DIR, f"{pair}_{q_name}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model for {pair} ({q_name}).\n"
                f"Run: python train.py {pair}"
            )
        with open(model_path, "rb") as f:
            step_models = pickle.load(f)

        ret_preds = np.array([m.predict(x_flat)[0] for m in step_models])
        results[q_name] = last_close * (1 + ret_preds)

    return results["median"], results["bull"], results["bear"], last_close


# ─── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_feed import fetch_candles

    pair = sys.argv[1].upper() if len(sys.argv) > 1 else "EURUSD"
    print(f"[ml_model] Feature test — {pair}")
    df = fetch_candles("1h", pair=pair)
    df_f = add_features(df)
    X, Y = make_sequences(df_f)
    print(f"  Candles    : {len(df)}")
    print(f"  Sequences  : {len(X)}")
    print(f"  X shape    : {X.shape}")
    print(f"  Y shape    : {Y.shape}")
    print("  ✓ Feature pipeline OK — ready to train")
