"""
scripts/backtest_var.py — VaR Backtesting (Basel III Lopez Test)

Chạy RiskPredictor trên toàn bộ tập dữ liệu lịch sử,
so sánh VaR dự đoán với log_return thực tế,
tính violation rate và kiểm tra chuẩn Basel III.

Usage:
    python scripts/backtest_var.py

Output: model/checkpoints/backtest_results.json
"""
from __future__ import annotations

import json
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from feature_engineering.engineer import FeatureEngineer
from model.predictor import RiskPredictor
from data_pipeline.schemas import ValidatedTick
from datetime import timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LOOKBACK     = 60   # seq_len phải khớp với training
OUTPUT_PATH  = "model/checkpoints/backtest_results.json"
DATA_PATH    = "data/raw/training_data.parquet"


def run_backtest() -> dict:
    logger.info("Loading dataset from %s ...", DATA_PATH)
    df = pd.read_parquet(DATA_PATH)

    symbols = df["symbol"].unique().tolist()
    logger.info("Symbols: %s  |  Total rows: %d", symbols, len(df))

    predictor = RiskPredictor()
    engineer  = FeatureEngineer(symbols=symbols, history_cap=len(df) + 100)

    # Replay ticks into FeatureEngineer (populate history)
    logger.info("Replaying ticks through FeatureEngineer...")
    feature_cols = [
        "log_return","vol_30","vol_60","vol_390",
        "rsi_14","macd_line","macd_signal","macd_hist","zscore_30",
    ]

    for _, row in df.iterrows():
        ts = row["timestamp"]
        if hasattr(ts, "tzinfo") and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        tick = ValidatedTick(
            symbol=row["symbol"],
            timestamp=ts,
            open=100.0, high=100.0, low=100.0,   # dummy OHLC — not used in predictor
            close=100.0 * (1 + float(row["log_return"])),
            volume=500_000.0,
            is_valid=True,
        )
        engineer.update(tick)

    # --- Backtesting ---
    results_per_symbol: dict = {}
    all_violations_99: list[bool] = []
    all_violations_95: list[bool] = []

    for sym in symbols:
        sym_df = df[df["symbol"] == sym].sort_values("timestamp")
        sym_df = sym_df.dropna(subset=feature_cols)

        if len(sym_df) < LOOKBACK + 1:
            logger.warning("Not enough data for %s — skipping", sym)
            continue

        n_predictions = 0
        violations_99 = 0
        violations_95 = 0
        var99_list: list[float] = []
        var95_list: list[float] = []
        actual_returns: list[float] = []

        logger.info("Backtesting %s (%d rows)...", sym, len(sym_df))

        features = sym_df[feature_cols].values.astype("float32")
        np.nan_to_num(features, copy=False)
        log_rets = sym_df["log_return"].values

        for i in range(LOOKBACK, len(features) - 1):
            window   = features[i - LOOKBACK : i]            # (60, 9)
            actual_r = float(log_rets[i + 1])                # next bar's actual return

            # Build single-symbol tensor: (LOOKBACK, 1, 9)
            tensor = window[np.newaxis, :, :].transpose(1, 0, 2)  # (LOOKBACK,1,9)
            preds  = predictor.predict(tensor, [sym])

            if sym not in preds:
                continue

            var99 = preds[sym]["var_99"]
            var95 = preds[sym]["var_95"]

            # A violation happens when actual LOSS exceeds VaR (loss = -return)
            actual_loss = -actual_r
            is_v99 = actual_loss > var99
            is_v95 = actual_loss > var95

            var99_list.append(var99)
            var95_list.append(var95)
            actual_returns.append(actual_r)
            violations_99 += int(is_v99)
            violations_95 += int(is_v95)
            n_predictions += 1

        if n_predictions == 0:
            continue

        vr99 = violations_99 / n_predictions
        vr95 = violations_95 / n_predictions

        # Basel III: VaR 99% is "acceptable" if violation rate ≤ 1%
        # Zone: Green < 1%,  Yellow 1-5%,  Red > 5%
        zone99 = "🟢 Green" if vr99 <= 0.01 else ("🟡 Yellow" if vr99 <= 0.05 else "🔴 Red")
        zone95 = "🟢 Green" if vr95 <= 0.05 else ("🟡 Yellow" if vr95 <= 0.10 else "🔴 Red")

        sym_result = {
            "n_predictions":     n_predictions,
            "violations_99":     violations_99,
            "violation_rate_99": round(vr99, 5),
            "basel_zone_99":     zone99,
            "violations_95":     violations_95,
            "violation_rate_95": round(vr95, 5),
            "basel_zone_95":     zone95,
            "mean_var99":        round(float(np.mean(var99_list)), 6),
            "mean_var95":        round(float(np.mean(var95_list)), 6),
            "mean_actual_return":round(float(np.mean(actual_returns)), 6),
            "std_actual_return": round(float(np.std(actual_returns)), 6),
        }
        results_per_symbol[sym] = sym_result
        all_violations_99.extend([actual_loss > var99 for actual_loss, var99
                                   in zip([-r for r in actual_returns], var99_list)])
        all_violations_95.extend([actual_loss > var95 for actual_loss, var95
                                   in zip([-r for r in actual_returns], var95_list)])

        logger.info(
            "  %s | n=%d | ViolRate99=%.2f%% %s | ViolRate95=%.2f%% %s",
            sym, n_predictions,
            vr99 * 100, zone99,
            vr95 * 100, zone95,
        )

    # Portfolio-level summary
    total_preds = len(all_violations_99)
    portfolio_vr99 = sum(all_violations_99) / total_preds if total_preds > 0 else None
    portfolio_vr95 = sum(all_violations_95) / total_preds if total_preds > 0 else None

    output = {
        "summary": {
            "total_predictions":          total_preds,
            "portfolio_violation_rate_99": round(portfolio_vr99, 5) if portfolio_vr99 else None,
            "portfolio_violation_rate_95": round(portfolio_vr95, 5) if portfolio_vr95 else None,
            "portfolio_basel_zone_99":    (
                "🟢 Green" if portfolio_vr99 and portfolio_vr99 <= 0.01
                else ("🟡 Yellow" if portfolio_vr99 and portfolio_vr99 <= 0.05 else "🔴 Red")
            ),
        },
        "per_symbol": results_per_symbol,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Backtest results saved to %s", OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("VaR BACKTEST REPORT (Basel III Lopez Test)")
    print("=" * 60)
    print(f"{'Symbol':<8} {'N':>6} {'ViolRate99':>12} {'Zone99':<14} {'ViolRate95':>12} {'Zone95'}")
    print("-" * 60)
    for sym, r in results_per_symbol.items():
        print(
            f"{sym:<8} {r['n_predictions']:>6} "
            f"{r['violation_rate_99']*100:>11.2f}% {r['basel_zone_99']:<14} "
            f"{r['violation_rate_95']*100:>11.2f}% {r['basel_zone_95']}"
        )
    print("-" * 60)
    if portfolio_vr99 is not None:
        print(
            f"{'PORTFOLIO':<8} {total_preds:>6} "
            f"{portfolio_vr99*100:>11.2f}% {output['summary']['portfolio_basel_zone_99']}"
        )
    print("=" * 60)
    return output


if __name__ == "__main__":
    run_backtest()
