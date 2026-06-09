"""
scripts/backtest_var.py — Portfolio-level VaR Backtesting (Basel III / Kupiec Test).

Computes:
1. Portfolio-level VaR backtest (using weighted portfolio returns)
2. Per-symbol VaR backtest
3. Kupiec Proportion of Failures (POF) test for statistical validity
4. CVaR violation analysis
5. Basel III traffic light zone classification

Usage:
    python scripts/backtest_var.py

Output: model/checkpoints/backtest_results.json
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from feature_engineering.engineer import FeatureEngineer
from model.predictor import RiskPredictor
from model.baseline import BaselineRiskModel
from portfolio.risk_aggregator import PortfolioRiskAggregator
from data_pipeline.schemas import ValidatedTick
from datetime import timezone
from config.settings import get_settings

# ── MLflow (Phase 7) — optional, does not break existing tests
try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LOOKBACK     = 60
OUTPUT_PATH  = "model/checkpoints/backtest_results.json"
DATA_PATH    = "data/raw/training_data.parquet"

# Equal weights for portfolio-level backtest (can be changed)
DEFAULT_EQUAL_WEIGHTS = True


def kupiec_pof_test(n_obs: int, n_violations: int, confidence: float) -> dict:
    """
    Kupiec Proportion of Failures (POF) Test.

    Tests whether the observed violation rate is statistically consistent
    with the expected rate under the VaR model.

    H0: p_actual = p_expected  (model is correctly calibrated)
    H1: p_actual != p_expected (model is mis-calibrated)

    Test statistic: LR = -2 * ln(L(p0) / L(p_hat))
    Under H0: LR ~ chi^2(1)
    Critical value at 95% confidence: 3.841

    Args:
        n_obs:        Total number of observations.
        n_violations: Number of VaR violations.
        confidence:   VaR confidence level (e.g., 0.95 for VaR 95%).

    Returns:
        Dict with test results.
    """
    expected_rate = 1.0 - confidence
    actual_rate   = n_violations / n_obs if n_obs > 0 else 0.0

    # Log-likelihood ratio
    p0 = expected_rate
    p_hat = actual_rate

    # Guard against log(0)
    eps = 1e-10
    p0    = max(eps, min(1 - eps, p0))
    p_hat = max(eps, min(1 - eps, p_hat))

    n_ok = n_obs - n_violations
    lr_stat = 2.0 * (
        n_violations * math.log(p_hat / p0)
        + n_ok * math.log((1 - p_hat) / (1 - p0))
    ) if n_obs > 0 else 0.0

    # Chi-squared(1) critical values
    chi2_95 = 3.841
    chi2_99 = 6.635

    reject_95 = lr_stat > chi2_95
    reject_99 = lr_stat > chi2_99

    return {
        "n_obs":           n_obs,
        "n_violations":    n_violations,
        "expected_rate":   round(expected_rate, 4),
        "actual_rate":     round(actual_rate, 4),
        "violation_ratio": round(actual_rate / expected_rate, 3) if expected_rate > 0 else None,
        "kupiec_lr_stat":  round(lr_stat, 4),
        "reject_h0_at_95": reject_95,  # True = model is mis-calibrated at 5% significance
        "reject_h0_at_99": reject_99,
        "pass": not reject_95,         # True = model passes Kupiec test
    }


def basel_zone(violation_rate: float, var_confidence: float) -> str:
    """
    Basel III traffic light classification for VaR model.

    For VaR 99%:
        Green  : 0–4 violations per 250 obs (rate ≤ 1.6%)
        Yellow : 5–9 violations
        Red    : ≥ 10 violations (rate > 3.6%)
    """
    expected = 1.0 - var_confidence
    if violation_rate <= expected:
        return "🟢 Green"
    elif violation_rate <= 3 * expected:
        return "🟡 Yellow"
    else:
        return "🔴 Red"


def run_backtest() -> dict:
    logger.info("Loading dataset from %s ...", DATA_PATH)

    if not os.path.exists(DATA_PATH):
        logger.error(
            "Training data not found at %s. Run scripts/generate_training_data.py first.",
            DATA_PATH,
        )
        sys.exit(1)

    df = pd.read_parquet(DATA_PATH)
    symbols = df["symbol"].unique().tolist()
    logger.info("Symbols: %s  |  Total rows: %d", symbols, len(df))

    feature_cols = [
        "log_return", "vol_30", "vol_60", "vol_390",
        "rsi_14", "macd_line", "macd_signal", "macd_hist", "zscore_30",
        "volume_change", "volume_zscore_30", "dollar_volume",
    ]

    # Determine which predictor to use
    predictor = RiskPredictor()
    baseline  = BaselineRiskModel()
    aggregator = PortfolioRiskAggregator()

    use_tft = predictor.checkpoint_loaded
    pred_method = "tft" if use_tft else "statistical_baseline"
    logger.info("Prediction method: %s", pred_method)

    # Equal portfolio weights
    n_syms = len(symbols)
    equal_weights = np.ones(n_syms) / n_syms

    # ── Per-symbol backtest ───────────────────────────────────────────────────
    results_per_symbol: dict = {}
    all_portfolio_actual_returns: list[float] = []
    all_portfolio_var95: list[float] = []
    all_portfolio_var99: list[float] = []

    for sym in symbols:
        sym_df = df[df["symbol"] == sym].sort_values("timestamp")
        sym_df = sym_df.dropna(subset=feature_cols)

        if len(sym_df) < LOOKBACK + 1:
            logger.warning("Not enough data for %s — skipping", sym)
            continue

        n_preds = 0
        viol_99 = 0
        viol_95 = 0
        var99_list: list[float] = []
        var95_list: list[float] = []
        actual_returns: list[float] = []

        features  = sym_df[feature_cols].values.astype("float32")
        np.nan_to_num(features, copy=False)
        log_rets  = sym_df["log_return"].values

        logger.info("Backtesting %s (%d rows)...", sym, len(sym_df))

        for i in range(LOOKBACK, len(features) - 1):
            window   = features[i - LOOKBACK: i]     # (60, 9)
            actual_r = float(log_rets[i + 1])

            # Build tensor: (LOOKBACK, 1, 9)
            tensor = window[np.newaxis, :, :].transpose(1, 0, 2)

            if use_tft:
                preds = predictor.predict(tensor, [sym])
            else:
                preds = baseline.predict(tensor, [sym])

            if sym not in preds:
                continue

            var99 = preds[sym]["var_99"]
            var95 = preds[sym]["var_95"]
            actual_loss = -actual_r

            viol_99 += int(actual_loss > var99)
            viol_95 += int(actual_loss > var95)
            var99_list.append(var99)
            var95_list.append(var95)
            actual_returns.append(actual_r)
            n_preds += 1

        if n_preds == 0:
            continue

        kupiec_99 = kupiec_pof_test(n_preds, viol_99, 0.99)
        kupiec_95 = kupiec_pof_test(n_preds, viol_95, 0.95)
        zone_99 = basel_zone(kupiec_99["actual_rate"], 0.99)
        zone_95 = basel_zone(kupiec_95["actual_rate"], 0.95)

        results_per_symbol[sym] = {
            "n_predictions":     n_preds,
            "kupiec_var99":      kupiec_99,
            "kupiec_var95":      kupiec_95,
            "basel_zone_99":     zone_99,
            "basel_zone_95":     zone_95,
            "mean_var99":        round(float(np.mean(var99_list)), 6),
            "mean_var95":        round(float(np.mean(var95_list)), 6),
            "mean_actual_return":round(float(np.mean(actual_returns)), 6),
            "std_actual_return": round(float(np.std(actual_returns)), 6),
        }

        logger.info(
            "  %s | n=%d | ViolRate99=%.2f%% %s | ViolRate95=%.2f%% %s | Kupiec99 pass=%s",
            sym, n_preds,
            kupiec_99["actual_rate"] * 100, zone_99,
            kupiec_95["actual_rate"] * 100, zone_95,
            kupiec_99["pass"],
        )

    # ── Portfolio-level backtest ──────────────────────────────────────────────
    # Align all symbols and compute weighted portfolio returns
    logger.info("\n--- Running portfolio-level backtest ---")

    all_sym_dfs = {}
    for sym in symbols:
        sdf = df[df["symbol"] == sym].sort_values("timestamp")
        sdf = sdf.dropna(subset=feature_cols).set_index("timestamp")
        all_sym_dfs[sym] = sdf["log_return"]

    # Align timestamps
    port_returns_df = pd.DataFrame(all_sym_dfs).dropna()
    port_returns = (port_returns_df.values @ equal_weights).tolist()

    port_n = len(port_returns)
    port_viol_99 = 0
    port_viol_95 = 0
    port_var99_list: list[float] = []
    port_var95_list: list[float] = []
    port_cvar95_list: list[float] = []
    port_cvar99_list: list[float] = []

    # Rolling portfolio VaR using the aggregator
    window_size = LOOKBACK

    for i in range(window_size, port_n - 1):
        hist_returns = port_returns_df.values[i - window_size: i]  # (window, n_syms)
        actual_port_r = port_returns[i + 1]

        result = aggregator.compute_portfolio_risk(
            returns_matrix=hist_returns,
            weights=equal_weights,
            symbols=symbols,
        )

        pvar99 = result["portfolio_var_99_parametric"]
        pvar95 = result["portfolio_var_95_parametric"]
        pcvar95 = result["portfolio_cvar_95"]
        pcvar99 = result["portfolio_cvar_99"]

        actual_loss = -actual_port_r
        port_viol_99 += int(actual_loss > pvar99)
        port_viol_95 += int(actual_loss > pvar95)
        port_var99_list.append(pvar99)
        port_var95_list.append(pvar95)
        port_cvar95_list.append(pcvar95)
        port_cvar99_list.append(pcvar99)

    port_n_preds = len(port_var99_list)
    port_kupiec_99 = kupiec_pof_test(port_n_preds, port_viol_99, 0.99) if port_n_preds > 0 else {}
    port_kupiec_95 = kupiec_pof_test(port_n_preds, port_viol_95, 0.95) if port_n_preds > 0 else {}

    # ── Plot Portfolio VaR Timeseries ─────────────────────────────────────────
    if port_n_preds > 0:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        actual_tested_returns = port_returns[window_size + 1 : port_n]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(actual_tested_returns, label='Portfolio Return', color='gray', alpha=0.6, linewidth=1)
        ax.plot([-v for v in port_var95_list], label='VaR 95% Boundary', color='#FF9800', linestyle='-', linewidth=1.5)
        ax.plot([-v for v in port_var99_list], label='VaR 99% Boundary', color='#F44336', linestyle='-', linewidth=1.5)
        
        # Highlight violations for 99%
        viol_x = [i for i, (r, v) in enumerate(zip(actual_tested_returns, port_var99_list)) if r < -v]
        viol_y = [actual_tested_returns[i] for i in viol_x]
        ax.scatter(viol_x, viol_y, color='red', marker='x', s=50, label='VaR 99% Violation', zorder=5)
        
        ax.set_title("Portfolio VaR Backtesting (Actual Returns vs. VaR Limits)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Log Return", fontsize=12)
        ax.set_xlabel("Time (1-minute intervals)", fontsize=12)
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        plot_path = os.path.join(os.path.dirname(OUTPUT_PATH), "portfolio_var_timeseries.png")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
        logger.info("Portfolio VaR timeseries plot saved to %s", plot_path)


    portfolio_summary = {
        "n_predictions":       port_n_preds,
        "weights":             {s: float(w) for s, w in zip(symbols, equal_weights)},
        "kupiec_var99":        port_kupiec_99,
        "kupiec_var95":        port_kupiec_95,
        "basel_zone_99":       basel_zone(port_kupiec_99.get("actual_rate", 0), 0.99),
        "basel_zone_95":       basel_zone(port_kupiec_95.get("actual_rate", 0), 0.95),
        "mean_portfolio_var99":round(float(np.mean(port_var99_list)), 6) if port_var99_list else 0,
        "mean_portfolio_var95":round(float(np.mean(port_var95_list)), 6) if port_var95_list else 0,
        "mean_portfolio_cvar95":round(float(np.mean(port_cvar95_list)), 6) if port_cvar95_list else 0,
        "mean_portfolio_cvar99":round(float(np.mean(port_cvar99_list)), 6) if port_cvar99_list else 0,
        "actual_portfolio_returns_summary": {
            "mean": round(float(np.mean(port_returns)), 6),
            "std":  round(float(np.std(port_returns)), 6),
        },
    }

    output = {
        "prediction_method": pred_method,
        "portfolio_backtest": portfolio_summary,
        "per_symbol_backtest": results_per_symbol,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Detailed backtest results saved to %s", OUTPUT_PATH)

    # Export simple summary as requested
    summary_output = {
        "method_used": pred_method,
        "warning": "checkpoint missing or insufficient data" if not use_tft else "None",
        "portfolio_summary": {
            "number_of_observations": port_n_preds,
            "VaR_99": {
                "violation_rate": port_kupiec_99.get("actual_rate", 0),
                "expected_violation_rate": port_kupiec_99.get("expected_rate", 0),
                "number_of_violations": port_kupiec_99.get("n_violations", 0),
                "pass_fail": "PASS" if port_kupiec_99.get("pass", False) else "FAIL"
            },
            "VaR_95": {
                "violation_rate": port_kupiec_95.get("actual_rate", 0),
                "expected_violation_rate": port_kupiec_95.get("expected_rate", 0),
                "number_of_violations": port_kupiec_95.get("n_violations", 0),
                "pass_fail": "PASS" if port_kupiec_95.get("pass", False) else "FAIL"
            }
        }
    }
    summary_path = os.path.join(os.path.dirname(OUTPUT_PATH), "backtest_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_output, f, indent=2)
    logger.info("Simple backtest summary saved to %s", summary_path)

    # ── Print Report ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VaR BACKTEST REPORT (Basel III + Kupiec POF Test)")
    print(f"Prediction method: {pred_method}")
    print("=" * 70)

    print("\n📊 Per-Symbol Results:")
    print(f"{'Symbol':<8} {'N':>6} {'ViolRate99':>11} {'Zone99':<12} {'ViolRate95':>11} {'Zone95':<12} {'Kupiec99'}")
    print("-" * 75)
    for sym, r in results_per_symbol.items():
        k99 = r["kupiec_var99"]
        k95 = r["kupiec_var95"]
        print(
            f"{sym:<8} {k99['n_obs']:>6} "
            f"{k99['actual_rate']*100:>10.2f}% {r['basel_zone_99']:<12} "
            f"{k95['actual_rate']*100:>10.2f}% {r['basel_zone_95']:<12} "
            f"{'PASS' if k99['pass'] else 'FAIL'}"
        )

    print("\n📈 Portfolio-Level Results (equal weights):")
    if port_n_preds > 0:
        print(f"  Observations : {port_n_preds}")
        print(f"  VaR 99% : rate={port_kupiec_99.get('actual_rate', 0)*100:.2f}% "
              f"| {portfolio_summary['basel_zone_99']} | Kupiec={'PASS' if port_kupiec_99.get('pass') else 'FAIL'}")
        print(f"  VaR 95% : rate={port_kupiec_95.get('actual_rate', 0)*100:.2f}% "
              f"| {portfolio_summary['basel_zone_95']} | Kupiec={'PASS' if port_kupiec_95.get('pass') else 'FAIL'}")
        print(f"  Mean CVaR 95%: {portfolio_summary['mean_portfolio_cvar95']:.6f}")
        print(f"  Mean CVaR 99%: {portfolio_summary['mean_portfolio_cvar99']:.6f}")
    print("=" * 70)

    return output


def run_backtest_and_log(run_id: str = None) -> dict:
    """
    Run backtest and log all Kupiec + Basel III results into MLflow.

    This is the entry point used by Airflow (and optionally the CLI).
    The core backtest logic is entirely delegated to run_backtest() —
    no changes to any financial calculations.

    Args:
        run_id: If provided, results are logged into the existing MLflow run
                (linked to the training run that produced the checkpoint).
                If None, a new standalone backtest run is created.

    Returns:
        The full backtest output dict (same as run_backtest()).
    """
    # Step 1: Run the actual backtest (all logic unchanged)
    output = run_backtest()

    # Step 2: If MLflow is not available or not desired, return early
    if not _MLFLOW_AVAILABLE:
        logger.warning("MLflow not installed — skipping metric logging.")
        return output

    cfg = get_settings()

    # Step 3: Log into MLflow
    try:
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        mlflow.set_experiment(cfg.mlflow_experiment_name)

        # Re-open the training run if run_id provided, else start a new one
        context = (
            mlflow.start_run(run_id=run_id)
            if run_id
            else mlflow.start_run(run_name="backtest_only")
        )

        with context:
            portfolio = output.get("portfolio_backtest", {})
            per_symbol = output.get("per_symbol_backtest", {})

            # ── Log per-symbol metrics ────────────────────────────────────────
            for sym, result in per_symbol.items():
                k99 = result.get("kupiec_var99", {})
                k95 = result.get("kupiec_var95", {})
                metrics = {
                    f"{sym}_violation_rate_99": k99.get("actual_rate", 0.0),
                    f"{sym}_violation_rate_95": k95.get("actual_rate", 0.0),
                    f"{sym}_kupiec_lr_99":      k99.get("kupiec_lr_stat", 0.0),
                    f"{sym}_kupiec_passed_99":  float(k99.get("pass", False)),
                    f"{sym}_kupiec_passed_95":  float(k95.get("pass", False)),
                }
                mlflow.log_metrics(metrics)

            # ── Log portfolio-level metrics ────────────────────────────────────
            pk99 = portfolio.get("kupiec_var99", {})
            pk95 = portfolio.get("kupiec_var95", {})

            n_green  = sum(1 for r in per_symbol.values() if "Green"  in r.get("basel_zone_99", ""))
            n_yellow = sum(1 for r in per_symbol.values() if "Yellow" in r.get("basel_zone_99", ""))
            n_red    = sum(1 for r in per_symbol.values() if "Red"    in r.get("basel_zone_99", ""))

            mlflow.log_metrics({
                "portfolio_violation_rate_99": pk99.get("actual_rate", 0.0),
                "portfolio_violation_rate_95": pk95.get("actual_rate", 0.0),
                "portfolio_kupiec_passed_99":  float(pk99.get("pass", False)),
                "portfolio_kupiec_passed_95":  float(pk95.get("pass", False)),
                "portfolio_kupiec_lr_99":      pk99.get("kupiec_lr_stat", 0.0),
                "portfolio_mean_var99":         portfolio.get("mean_portfolio_var99", 0.0),
                "portfolio_mean_cvar99":        portfolio.get("mean_portfolio_cvar99", 0.0),
                "n_green_symbols":             float(n_green),
                "n_yellow_symbols":            float(n_yellow),
                "n_red_symbols":               float(n_red),
            })

            # ── Log tags and artifact ────────────────────────────────────────────
            mlflow.set_tag("basel_zone_99", portfolio.get("basel_zone_99", "Unknown"))
            mlflow.set_tag("kupiec_passed", str(pk99.get("pass", False)))
            mlflow.set_tag("status", "backtested")
            mlflow.set_tag("prediction_method", output.get("prediction_method", "unknown"))

            if os.path.exists(OUTPUT_PATH):
                mlflow.log_artifact(OUTPUT_PATH, artifact_path="backtest")

        logger.info(
            "MLflow backtest metrics logged. "
            "portfolio_kupiec_passed_99=%s | basel_zone=%s",
            pk99.get("pass"), portfolio.get("basel_zone_99"),
        )

    except Exception as e:
        logger.warning("MLflow backtest logging failed (%s) — results still saved to disk.", e)

    return output


if __name__ == "__main__":
    run_backtest()
