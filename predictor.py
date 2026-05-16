"""
RadarFX ML — predictor.py
Loads a trained LSTM model, predicts the next 20 candles,
and generates a professional dark-theme chart.

Usage:
  python predictor.py              # EURUSD
  python predictor.py XAUUSD
  python predictor.py GBPUSD --save chart.png
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data_feed import fetch_candles, PAIRS
from ml_model  import predict as ml_predict, FORECAST, LOOKBACK


# ─── Chart ────────────────────────────────────────────────────────────────────

# Dark theme colours (matches the screenshots)
BG_COLOR     = "#0d1117"
GRID_COLOR   = "#1e2530"
CANDLE_UP    = "#26a69a"
CANDLE_DOWN  = "#ef5350"
BULL_COLOR   = "#448aff"     # blue  — bullish scenario
MID_COLOR    = "#66bb6a"     # green — median/neutral
BEAR_COLOR   = "#ff5252"     # red   — bearish scenario
TEXT_COLOR   = "#b0bec5"
PRICE_COLOR  = "#eceff1"


def draw_chart(df_hist: pd.DataFrame,
               median_p: np.ndarray,
               bull_p:   np.ndarray,
               bear_p:   np.ndarray,
               last_close: float,
               pair: str,
               tf: str = "1h",
               save_path: str = None,
               show: bool = True) -> str:
    """
    Draw last 80 candles + 20 prediction candles.
    Returns the save path (str) or None.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")   # non-interactive backend for server use
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    # ── Prepare candle data ──────────────────────────────────────────────
    n_hist = 80
    df_plot = df_hist.tail(n_hist).copy()

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # ── Draw candlesticks ────────────────────────────────────────────────
    x_hist = np.arange(len(df_plot))
    for i, (_, row) in enumerate(df_plot.iterrows()):
        color = CANDLE_UP if row["close"] >= row["open"] else CANDLE_DOWN
        # Wick
        ax.plot([i, i], [row["low"], row["high"]],
                color=color, linewidth=0.8, zorder=2)
        # Body
        body_lo = min(row["open"], row["close"])
        body_hi = max(row["open"], row["close"])
        rect = mpatches.FancyBboxPatch(
            (i - 0.35, body_lo), 0.7, max(body_hi - body_lo, 1e-8),
            boxstyle="square,pad=0", linewidth=0,
            facecolor=color, zorder=3
        )
        ax.add_patch(rect)

    # ── Prediction x-axis (continues after candles) ──────────────────────
    x_pred = np.arange(n_hist, n_hist + FORECAST)

    # Shade between bull and bear (uncertainty band)
    ax.fill_between(x_pred, bear_p, bull_p,
                    color=MID_COLOR, alpha=0.08, zorder=1)

    # ── Draw 3 prediction lines ──────────────────────────────────────────
    # Start each line from last_close so they diverge naturally
    def prepend_last(arr):
        return np.concatenate([[last_close], arr])

    x_line = np.concatenate([[n_hist - 1], x_pred])

    ax.plot(x_line, prepend_last(bull_p),
            color=BULL_COLOR, linewidth=1.8, linestyle="-",
            label="Bullish scenario", zorder=5)

    ax.plot(x_line, prepend_last(median_p),
            color=MID_COLOR,  linewidth=2.0, linestyle="-",
            label="Median forecast", zorder=6)

    ax.plot(x_line, prepend_last(bear_p),
            color=BEAR_COLOR, linewidth=1.8, linestyle="--",
            label="Bearish scenario", zorder=5, alpha=0.85)

    # ── Divider line at "now" ────────────────────────────────────────────
    ax.axvline(x=n_hist - 1, color="#546e7a", linewidth=1.2,
               linestyle=":", alpha=0.7, zorder=4)
    ax.text(n_hist - 1 + 0.3, ax.get_ylim()[1] if ax.get_ylim()[1] else last_close,
            " NOW", color="#546e7a", fontsize=9, va="top")

    # ── Current price label ──────────────────────────────────────────────
    decimals = PAIRS[pair]["decimals"]
    ax.axhline(y=last_close, color="#546e7a", linewidth=0.8,
               linestyle=":", alpha=0.5)
    ax.text(len(df_plot) + FORECAST + 0.5, last_close,
            f" {last_close:.{decimals}f}",
            color=PRICE_COLOR, fontsize=9, va="center", ha="left")

    # ── Grid & axes ──────────────────────────────────────────────────────
    ax.set_xlim(-1, n_hist + FORECAST + 2)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.{decimals}f}")
    )
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.7)
    ax.set_xlabel("Candles (1h)", color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel("Price", color=TEXT_COLOR, fontsize=10)

    # ── X-axis labels: show timestamps ───────────────────────────────────
    step = 10
    x_ticks = list(range(0, n_hist, step)) + [n_hist + FORECAST - 1]
    x_labels = []
    for t in x_ticks:
        if t < len(df_plot):
            x_labels.append(str(df_plot.index[t])[5:16])
        else:
            x_labels.append(f"+{t - n_hist + 1}h")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8, color=TEXT_COLOR)

    # ── Title & legend ───────────────────────────────────────────────────
    pair_display = PAIRS[pair]["display"]
    tf_labels = {"5m":"5min","15m":"15min","30m":"30min","1h":"1H","4h":"4H","daily":"Daily"}
    tf_label = tf_labels.get(tf, tf)
    ax.set_title(
        f"RadarFX  ·  {pair_display}  ·  {tf_label} Forecast  (+{FORECAST} candles)",
        color=PRICE_COLOR, fontsize=13, fontweight="bold", pad=12
    )

    legend_elements = [
        Line2D([0], [0], color=BULL_COLOR, linewidth=2, label="Bullish (85th pct)"),
        Line2D([0], [0], color=MID_COLOR,  linewidth=2, label="Median  (50th pct)"),
        Line2D([0], [0], color=BEAR_COLOR, linewidth=2,
               linestyle="--", label="Bearish (15th pct)"),
    ]
    legend = ax.legend(handles=legend_elements, loc="upper left",
                       framealpha=0.3, facecolor=BG_COLOR,
                       labelcolor=TEXT_COLOR, fontsize=9)

    # ── Price annotations for endpoint predictions ────────────────────────
    for arr, color, offset in [
        (bull_p,   BULL_COLOR, +0.0002),
        (median_p, MID_COLOR,  0),
        (bear_p,   BEAR_COLOR, -0.0002),
    ]:
        final = arr[-1]
        ax.annotate(
            f"{final:.{decimals}f}",
            xy=(n_hist + FORECAST - 1, final),
            xytext=(n_hist + FORECAST + 0.5, final + offset * final),
            color=color, fontsize=8, va="center",
        )

    plt.tight_layout()

    # ── Save ─────────────────────────────────────────────────────────────
    if save_path is None:
        charts_dir = os.path.join(os.path.dirname(__file__), "..", "charts")
        os.makedirs(charts_dir, exist_ok=True)
        save_path = os.path.join(charts_dir, f"{pair}_forecast.png")

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR)
    print(f"  ✓ Chart saved → {save_path}")

    if show:
        plt.show()
    plt.close()

    return save_path


# ─── Run Prediction ───────────────────────────────────────────────────────────

def run_prediction(pair: str, tf: str = "1h", show: bool = True, save_path: str = None):
    """
    Full prediction pipeline for one pair.
    Returns dict with prediction arrays and chart path.
    """
    pair = pair.upper()
    if pair not in PAIRS:
        raise ValueError(f"Unknown pair '{pair}'. Available: {list(PAIRS.keys())}")

    display = PAIRS[pair]["display"]
    decimals = PAIRS[pair]["decimals"]

    print(f"\n{'═'*55}")
    print(f"  RadarFX ML Prediction  —  {display}")
    print(f"{'═'*55}")

    # Fetch enough history
    print(f"  Fetching {tf} data...")
    df = fetch_candles(tf, pair=pair)
    if df is None or df.empty:
        raise RuntimeError("No data returned.")
    print(f"  Candles: {len(df)}  ({df.index[-1]})")

    # Predict
    model_key = f"{pair}_{tf}"
    print(f"  Running prediction for {model_key}...")
    median_p, bull_p, bear_p, last_close = ml_predict(df, pair=model_key)

    # Print text summary
    print(f"\n  Current price : {last_close:.{decimals}f}")
    print(f"  ── +{FORECAST}h Forecast ──────────────────────────────")
    print(f"  Bullish (85%) : {bull_p[-1]:.{decimals}f}  "
          f"({(bull_p[-1]/last_close-1)*100:+.2f}%)")
    print(f"  Median  (50%) : {median_p[-1]:.{decimals}f}  "
          f"({(median_p[-1]/last_close-1)*100:+.2f}%)")
    print(f"  Bearish (15%) : {bear_p[-1]:.{decimals}f}  "
          f"({(bear_p[-1]/last_close-1)*100:+.2f}%)")
    print(f"{'═'*55}")

    # Draw chart
    chart_path = draw_chart(df, median_p, bull_p, bear_p,
                            last_close, pair, tf=tf,
                            save_path=save_path, show=show)

    return {
        "pair":        pair,
        "last_close":  last_close,
        "median":      median_p.tolist(),
        "bull":        bull_p.tolist(),
        "bear":        bear_p.tolist(),
        "chart_path":  chart_path,
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RadarFX ML Predictor")
    p.add_argument("pair",    nargs="?", default="EURUSD")
    p.add_argument("--tf",    type=str,  default="1h",
                   help="Timeframe: 5m, 15m, 30m, 1h, 4h, daily")
    p.add_argument("--save",  type=str,  default=None)
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    run_prediction(
        pair      = args.pair,
        tf        = args.tf,
        show      = not args.no_show,
        save_path = args.save,
    )
