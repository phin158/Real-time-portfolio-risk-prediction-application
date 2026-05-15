"""
scripts/plot_results.py — Tạo toàn bộ biểu đồ đánh giá cho báo cáo.

Sinh ra các file PNG:
  model/checkpoints/learning_curve.png
  model/checkpoints/var_violation_bar.png
  model/checkpoints/evaluation_report.json  (bổ sung Kupiec Test + RMSE)

Usage:
    python scripts/plot_results.py
"""
from __future__ import annotations

import json
import os
import sys
import logging
from math import log

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless — không cần màn hình
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HISTORY_PATH  = "model/checkpoints/training_history.json"
BACKTEST_PATH = "model/checkpoints/backtest_results.json"
OUT_DIR       = "model/checkpoints"
os.makedirs(OUT_DIR, exist_ok=True)


# ── Kupiec Likelihood Ratio Test ──────────────────────────────────────────────
def kupiec_lr_test(n: int, violations: int, confidence: float) -> dict:
    """
    Kupiec (1995) Proportion-of-Failures (POF) test.
    H0: violation rate == expected rate (1-confidence)
    LR ~ chi2(1); p < 0.05 → reject H0 (model inaccurate)
    """
    from scipy.stats import chi2

    p0 = 1.0 - confidence          # expected violation prob
    x  = max(violations, 1e-9)     # avoid log(0)
    p1 = x / n                     # observed violation prob

    # Log-likelihood under H0 (fixed p0) vs H1 (MLE p1)
    ll_h0 = x * log(p0) + (n - x) * log(1 - p0)
    ll_h1 = x * log(p1) + (n - x) * log(1 - p1)
    lr    = -2.0 * (ll_h0 - ll_h1)          # always ≥ 0
    lr    = max(lr, 0.0)                     # numerical guard

    p_value = float(chi2.sf(lr, df=1))
    return {
        "n":            n,
        "violations":   int(violations),
        "p_expected":   round(p0, 4),
        "p_actual":     round(float(violations / n), 6),
        "lr_statistic": round(lr, 4),
        "p_value":      round(p_value, 4),
        "pass":         p_value >= 0.05,
    }


# ── 1. Learning Curve ─────────────────────────────────────────────────────────
def plot_learning_curve() -> None:
    logger.info("Plotting learning curve from %s ...", HISTORY_PATH)
    with open(HISTORY_PATH) as f:
        h = json.load(f)

    train_losses = h["train_losses"]
    val_losses   = h["val_losses"]
    meta         = h["metadata"]
    best_epoch   = meta["best_epoch"]
    epochs       = list(range(1, len(train_losses) + 1))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_losses, "o-", color="#2196F3", linewidth=2,
            markersize=4, label="Train Loss")
    ax.plot(epochs, val_losses,   "s-", color="#FF5722", linewidth=2,
            markersize=4, label="Validation Loss")

    ax.axvline(best_epoch, color="#4CAF50", linestyle="--", linewidth=1.5,
               label=f"Best Epoch = {best_epoch}")
    ax.scatter([best_epoch], [meta["best_val_loss"]],
               color="#4CAF50", zorder=5, s=80)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (Quantile + MSE)", fontsize=12)
    ax.set_title("TFT Model — Training & Validation Loss Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotation box
    info = (f"Best Val Loss: {meta['best_val_loss']:.6f}\n"
            f"Epochs ran: {meta['epochs_ran']}\n"
            f"Train samples: {meta['n_train_samples']:,}\n"
            f"Time: {meta['total_training_seconds']}s")
    ax.text(0.98, 0.97, info, transform=ax.transAxes, fontsize=8.5,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    out = os.path.join(OUT_DIR, "learning_curve.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out)


# ── 2. VaR Violation Bar Chart ────────────────────────────────────────────────
def plot_violation_bars(kupiec_results: dict) -> None:
    logger.info("Plotting VaR violation bar chart ...")

    symbols  = list(kupiec_results.keys())
    vr99     = [kupiec_results[s]["var_99"]["p_actual"] * 100 for s in symbols]
    vr95     = [kupiec_results[s]["var_95"]["p_actual"] * 100 for s in symbols]

    x    = np.arange(len(symbols))
    w    = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars99 = ax.bar(x - w/2, vr99, w, color="#EF5350", alpha=0.85, label="VaR 99% Violation Rate")
    bars95 = ax.bar(x + w/2, vr95, w, color="#FF8A65", alpha=0.85, label="VaR 95% Violation Rate")

    ax.axhline(1.0, color="#D32F2F", linestyle="--", linewidth=1.5,
               label="Basel III limit 99% (1%)")
    ax.axhline(5.0, color="#E64A19", linestyle=":",  linewidth=1.5,
               label="Basel III limit 95% (5%)")

    ax.bar_label(bars99, fmt="%.2f%%", fontsize=8, padding=2)
    ax.bar_label(bars95, fmt="%.2f%%", fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(symbols, fontsize=11)
    ax.set_ylabel("Violation Rate (%)", fontsize=12)
    ax.set_title("VaR Backtesting — Basel III Lopez Test\n(lower is better; must be below dashed lines)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(max(vr99), max(vr95)) * 1.4 + 1)
    ax.grid(axis="y", alpha=0.3)

    out = os.path.join(OUT_DIR, "var_violation_bar.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out)


# ── 3. Extended Evaluation Report (Kupiec + RMSE) ────────────────────────────
def build_evaluation_report() -> dict:
    logger.info("Building extended evaluation report with Kupiec Test ...")
    with open(BACKTEST_PATH) as f:
        backtest = json.load(f)

    kupiec_results: dict = {}
    rows = []

    for sym, res in backtest["per_symbol"].items():
        n     = res["n_predictions"]
        v99   = res["violations_99"]
        v95   = res["violations_95"]

        k99 = kupiec_lr_test(n, v99, confidence=0.99)
        k95 = kupiec_lr_test(n, v95, confidence=0.95)

        # RMSE between mean VaR and mean actual return std (proxy)
        # Real RMSE needs tick-level predictions; use mean_var as proxy
        actual_std = abs(res["std_actual_return"])
        mean_var99 = res["mean_var99"]
        rmse_proxy = float(abs(mean_var99 - actual_std))

        kupiec_results[sym] = {"var_99": k99, "var_95": k95}
        rows.append({
            "Symbol":            sym,
            "N":                 n,
            "ViolRate99 (%)":    round(k99["p_actual"] * 100, 3),
            "Kupiec p-val 99%":  k99["p_value"],
            "Kupiec Pass 99%":   "✅ Pass" if k99["pass"] else "❌ Fail",
            "ViolRate95 (%)":    round(k95["p_actual"] * 100, 3),
            "Kupiec p-val 95%":  k95["p_value"],
            "Kupiec Pass 95%":   "✅ Pass" if k95["pass"] else "❌ Fail",
            "|MeanVaR - Std|":   round(rmse_proxy, 6),
        })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("EXTENDED EVALUATION REPORT")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)

    # Save JSON
    out_path = os.path.join(OUT_DIR, "evaluation_report.json")
    with open(out_path, "w") as f:
        json.dump({"kupiec": kupiec_results, "table": rows}, f, indent=2)
    logger.info("Saved: %s", out_path)

    return kupiec_results


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        from scipy.stats import chi2   # check scipy available
    except ImportError:
        print("Installing scipy...")
        os.system("pip install scipy -q")

    plot_learning_curve()
    kupiec = build_evaluation_report()
    plot_violation_bars(kupiec)

    print(f"\n✅ All outputs saved to {OUT_DIR}/")
    print("  → learning_curve.png")
    print("  → var_violation_bar.png")
    print("  → evaluation_report.json")
