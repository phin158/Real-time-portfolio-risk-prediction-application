"""
scripts/train_tft.py — Complete TFT Training Pipeline with Time-based Split.

This script:
1. Downloads historical 1-minute OHLCV data from yfinance.
2. Enforces strict time-based split (train / val / test / replay).
   NO look-ahead bias: features at time t never use data after t.
3. Creates PortfolioRiskDataset with proper labels (future realized returns).
4. Trains TFT with configurable loss weights (lambda_quantile, lambda_vol).
5. Logs all loss components separately (quantile_loss, vol_loss, total_loss).
6. Saves checkpoint + metadata JSON alongside the model file.
7. Validates the checkpoint can be reloaded successfully.

Usage:
    # Basic training with default config from .env
    python scripts/train_tft.py

    # Override split dates
    python scripts/train_tft.py --train-end 2025-01-15 --val-end 2025-01-20

    # Custom epochs and batch size
    python scripts/train_tft.py --epochs 30 --batch-size 32

NOTE ON DATA:
    yfinance 1-minute data is limited to ~30 days. For a proper
    train/val/test split, use enough data (e.g., --period 30d).
    Do NOT use daily-trained weights for 1-minute inference directly
    unless feature scales and horizons are explicitly adapted.

TIME-BASED SPLIT DIAGRAM:
    ─────────────────────────────────────────────────────────────
    |      TRAIN       |   VAL   |   TEST  |   REPLAY (demo)  |
    ─────────────────────────────────────────────────────────────
    t=0                t1        t2         t3                t_now
    ^── never use replay data for training ─────────────────────^
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import get_settings
from data_pipeline.schemas import ValidatedTick
from feature_engineering.engineer import FeatureEngineer
from model.tft import TemporalFusionTransformer, tft_loss_components
from model.predictor import RiskPredictor

# ── MLflow (Phase 7) — imported lazily so pytest can run without MLflow server
try:
    import mlflow
    import mlflow.pytorch
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Dataset ───────────────────────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    """
    Per-symbol time-series dataset with NO look-ahead leakage.

    Each sample: (x, y)
    - x: features[i : i+seq_len]           — historical context
    - y: [future_return, future_vol]        — labels computed AFTER x ends

    Feature columns match FeatureEngineer's FEATURE_NAMES.
    """

    FEATURE_COLS = [
        "log_return", "vol_30", "vol_60", "vol_390",
        "rsi_14", "macd_line", "macd_signal", "macd_hist", "zscore_30",
        # Volume features (Phase 5)
        "volume_change", "volume_zscore_30", "dollar_volume",
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 60,
        pred_len: int = 5,
        symbols: Optional[list] = None,
    ) -> None:
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.samples = []

        if symbols:
            df = df[df["symbol"].isin(symbols)]

        for sym, group in df.groupby("symbol"):
            group = group.sort_values("timestamp").reset_index(drop=True)
            avail_cols = [c for c in self.FEATURE_COLS if c in group.columns]
            features = group[avail_cols].values.astype(np.float32)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            n = len(features)
            for i in range(n - seq_len - pred_len + 1):
                x = features[i: i + seq_len]
                future = features[i + seq_len: i + seq_len + pred_len]
                future_returns = future[:, 0]  # log_return column
                y_return = float(future_returns[0])
                y_vol = float(np.std(future_returns, ddof=1)) if pred_len > 1 else 0.0
                self.samples.append(
                    (x, np.array([y_return, y_vol], dtype=np.float32))
                )

        logger.info("Dataset: %d samples for symbols=%s", len(self.samples), symbols)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)


# ── Data Fetching ─────────────────────────────────────────────────────────────

def fetch_and_engineer_features(
    tickers: list[str],
    period: str = "30d",
) -> pd.DataFrame:
    """
    Download 1-minute OHLCV from yfinance, run through FeatureEngineer,
    and return a combined DataFrame sorted by timestamp.

    Returns:
        DataFrame with columns: symbol, timestamp, + FEATURE_COLS.
        Sorted chronologically — safe for time-based splitting.
    """
    logger.info("Downloading 1-min data for %s (period=%s) ...", tickers, period)
    all_ticks: list[ValidatedTick] = []

    for sym in tickers:
        try:
            df = yf.download(sym, period=period, interval="1m", progress=False)
            if df.empty:
                logger.warning("No data returned for %s", sym)
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            for ts, row in df.iterrows():
                if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                    utc_ts = ts.to_pydatetime().astimezone(timezone.utc)
                else:
                    utc_ts = ts.to_pydatetime().replace(tzinfo=timezone.utc)
                tick = ValidatedTick(
                    symbol=sym, timestamp=utc_ts,
                    open=float(row["Open"]),  high=float(row["High"]),
                    low=float(row["Low"]),    close=float(row["Close"]),
                    volume=float(row["Volume"]), is_valid=True,
                )
                all_ticks.append(tick)
        except Exception as e:
            logger.error("Error fetching %s: %s", sym, e)

    if not all_ticks:
        raise RuntimeError("No data fetched. Check tickers and internet connection.")

    all_ticks.sort(key=lambda t: t.timestamp)
    logger.info("Fetched %d total ticks across %d symbols.", len(all_ticks), len(tickers))

    # Run through FeatureEngineer
    engineer = FeatureEngineer(symbols=tickers, history_cap=len(all_ticks) + 1000)
    for tick in all_ticks:
        engineer.update(tick)

    all_frames = []
    for sym in tickers:
        fdf = engineer.get_feature_df(sym, lookback=len(all_ticks))
        if fdf.empty:
            continue
        fdf = fdf.reset_index()
        fdf.rename(columns={"index": "timestamp"}, inplace=True)
        fdf.insert(0, "symbol", sym)
        all_frames.append(fdf)

    if not all_frames:
        raise RuntimeError("FeatureEngineer produced no features. Check warm-up period.")

    combined = pd.concat(all_frames, ignore_index=True)
    combined.sort_values(["timestamp", "symbol"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    logger.info("Feature dataset shape: %s", combined.shape)
    return combined


# ── Time-based Split ──────────────────────────────────────────────────────────

def time_split(
    df: pd.DataFrame,
    train_end: Optional[str],
    val_end: Optional[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame by timestamp (no shuffling — preserves time order).

    Returns:
        (train_df, val_df, test_df)
        test_df contains everything after val_end.

    If dates not provided, falls back to 70/15/15 ratio split.
    """
    ts = pd.to_datetime(df["timestamp"], utc=True)

    if train_end and val_end:
        t1 = pd.Timestamp(train_end, tz="UTC")
        t2 = pd.Timestamp(val_end, tz="UTC")
        train_df = df[ts < t1].copy()
        val_df   = df[(ts >= t1) & (ts < t2)].copy()
        test_df  = df[ts >= t2].copy()
        logger.info(
            "Time split: train=%d, val=%d, test=%d rows",
            len(train_df), len(val_df), len(test_df),
        )
    else:
        # Fallback: ratio split (less ideal for time series, but better than random)
        n = len(df)
        t1 = int(n * 0.70)
        t2 = int(n * 0.85)
        train_df = df.iloc[:t1].copy()
        val_df   = df.iloc[t1:t2].copy()
        test_df  = df.iloc[t2:].copy()
        logger.info(
            "Ratio split (no dates provided): train=%d, val=%d, test=%d rows",
            len(train_df), len(val_df), len(test_df),
        )

    return train_df, val_df, test_df


# ── Training Loop ─────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 7) -> None:
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def train(
    train_ds: TimeSeriesDataset,
    val_ds: TimeSeriesDataset,
    num_features: int,
    hidden_size: int,
    checkpoint_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    lambda_quantile: float,
    lambda_vol: float,
    device: str,
    mlflow_run=None,  # ← active MLflow run object, passed in from main()
) -> dict:
    """
    Run training loop with early stopping and separate loss logging.

    Returns:
        Training history dict with per-epoch loss components.

    mlflow_run: if provided (active mlflow.ActiveRun), per-epoch metrics are
    logged automatically.  Passing None disables MLflow logging entirely so
    that unit tests and offline usage are unaffected.
    """
    dev = torch.device(device)
    model = TemporalFusionTransformer(num_features=num_features, hidden_size=hidden_size).to(dev)
    optimizer = Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=7)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False,  # No shuffle for time series
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    history = {"train": [], "val": []}
    best_val_loss = float("inf")

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    logger.info(
        "Training: device=%s, epochs=%d, batch=%d, lr=%.1e, λ_q=%.2f, λ_v=%.2f",
        device, epochs, batch_size, lr, lambda_quantile, lambda_vol,
    )
    logger.info("Train samples: %d | Val samples: %d", len(train_ds), len(val_ds))

    for epoch in range(epochs):
        # ── Training ────────────────────────────────────────────────────
        model.train()
        ep_q_loss = 0.0
        ep_vol_loss = 0.0
        ep_total = 0.0
        n_train = 0

        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad()
            preds = model(x)
            components = tft_loss_components(
                preds, y,
                lambda_quantile=lambda_quantile,
                lambda_volatility=lambda_vol,
            )
            loss = components["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = x.size(0)
            ep_q_loss   += components["quantile_loss"].item() * bs
            ep_vol_loss += components["volatility_loss"].item() * bs
            ep_total    += loss.item() * bs
            n_train     += bs

        if n_train > 0:
            ep_q_loss   /= n_train
            ep_vol_loss /= n_train
            ep_total    /= n_train

        # ── Validation ──────────────────────────────────────────────────
        model.eval()
        val_q, val_vol, val_total_sum, n_val = 0.0, 0.0, 0.0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(dev), y.to(dev)
                preds = model(x)
                comps = tft_loss_components(
                    preds, y,
                    lambda_quantile=lambda_quantile,
                    lambda_volatility=lambda_vol,
                )
                bs = x.size(0)
                val_q     += comps["quantile_loss"].item() * bs
                val_vol   += comps["volatility_loss"].item() * bs
                val_total_sum += comps["total_loss"].item() * bs
                n_val     += bs

        if n_val > 0:
            val_q /= n_val
            val_vol /= n_val
            val_total = val_total_sum / n_val
        else:
            val_total = float("inf")

        ep_record = {
            "epoch": epoch + 1,
            "train_quantile_loss": round(ep_q_loss, 6),
            "train_volatility_loss": round(ep_vol_loss, 6),
            "train_total_loss": round(ep_total, 6),
            "val_quantile_loss": round(val_q, 6),
            "val_volatility_loss": round(val_vol, 6),
            "val_total_loss": round(val_total, 6),
        }
        history["train"].append(ep_record)

        logger.info(
            "Epoch %d/%d — train: q=%.5f vol=%.5f total=%.5f | val: q=%.5f vol=%.5f total=%.5f",
            epoch + 1, epochs,
            ep_q_loss, ep_vol_loss, ep_total,
            val_q, val_vol, val_total,
        )

        # ── MLflow: log per-epoch metrics ────────────────────────────────
        if mlflow_run is not None and _MLFLOW_AVAILABLE:
            mlflow.log_metrics({
                "train_loss":         ep_total,
                "train_quantile_loss": ep_q_loss,
                "train_vol_loss":     ep_vol_loss,
                "val_loss":           val_total,
                "val_quantile_loss":  val_q,
                "val_vol_loss":       val_vol,
            }, step=epoch)

        # ── Checkpoint if best ──────────────────────────────────────────
        if val_total < best_val_loss:
            best_val_loss = val_total
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("  ✅ Checkpoint saved (val_total=%.6f)", best_val_loss)
            # ── MLflow: log best checkpoint as artifact ──────────────────
            if mlflow_run is not None and _MLFLOW_AVAILABLE:
                mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
                mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)

        if early_stopping(val_total):
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    return history


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> Optional[str]:
    """
    Entry point for CLI usage and Airflow integration.

    Returns:
        run_id (str | None): MLflow run ID if MLflow is available and tracking
        URI is reachable, otherwise None.  Airflow tasks use this to link the
        backtest into the same run.
    """
    parser = argparse.ArgumentParser(
        description="Train TFT model for portfolio risk prediction."
    )
    # NOTE: Default period is 30d (production). Use --period 7d only for
    # quick smoke tests / verification. Do NOT change this default.
    parser.add_argument("--period", default="30d",
        help="yfinance download period (e.g. 7d, 30d). Default: 30d (production)")
    parser.add_argument("--train-end", default="",
        help="Training data ends at this date (YYYY-MM-DD). Default: use .env")
    parser.add_argument("--val-end", default="",
        help="Validation data ends at this date (YYYY-MM-DD). Default: use .env")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--pred-len", type=int, default=5)
    parser.add_argument("--checkpoint", default="",
        help="Override checkpoint path. Default: from .env")
    args = parser.parse_args()

    cfg = get_settings()
    checkpoint_path = args.checkpoint or cfg.model_checkpoint_path
    train_end = args.train_end or cfg.train_end_date or None
    val_end   = args.val_end   or cfg.val_end_date   or None

    # ── Step 1: Fetch data and compute features ──────────────────────────────
    df = fetch_and_engineer_features(cfg.tickers_list, period=args.period)

    # ── Step 2: Time-based split ─────────────────────────────────────────────
    train_df, val_df, test_df = time_split(df, train_end, val_end)

    if len(train_df) < args.seq_len + args.pred_len:
        logger.error(
            "Training set too small (%d rows). Need at least %d. "
            "Try a longer --period or loosen split dates.",
            len(train_df), args.seq_len + args.pred_len,
        )
        sys.exit(1)

    # ── Step 3: Build datasets ───────────────────────────────────────────────
    num_features = 12  # 9 original + 3 volume features (Phase 5)
    train_ds = TimeSeriesDataset(train_df, seq_len=args.seq_len, pred_len=args.pred_len)
    val_ds   = TimeSeriesDataset(val_df,   seq_len=args.seq_len, pred_len=args.pred_len)

    if len(train_ds) == 0:
        logger.error("Training dataset is empty after time split. Adjust dates.")
        sys.exit(1)

    # If val dataset is empty, reuse training set for validation (fallback)
    if len(val_ds) == 0:
        logger.warning("Validation set empty — using train set for validation (not ideal).")
        val_ds = train_ds

    # ── Step 4: Init MLflow run (Phase 7) ────────────────────────────────────
    # MLflow is optional — if server is unreachable or library not installed,
    # training continues normally without any tracking.
    run_id: Optional[str] = None
    mlflow_run = None

    if _MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
            mlflow.set_experiment(cfg.mlflow_experiment_name)
            mlflow_run = mlflow.start_run(
                run_name=f"tft_{args.period}_{args.epochs}ep"
            )
            run_id = mlflow_run.info.run_id
            logger.info("MLflow Run started — run_id=%s", run_id)

            # Log all hyperparameters upfront
            mlflow.log_params({
                "hidden_size":       cfg.hidden_size,
                "num_features":      num_features,
                "lookback":          args.seq_len,
                "pred_len":          args.pred_len,
                "output_size":       4,
                "dropout":           0.1,
                "lambda_quantile":   cfg.lambda_quantile,
                "lambda_volatility": cfg.lambda_volatility,
                "epochs":            args.epochs,
                "batch_size":        args.batch_size,
                "learning_rate":     args.lr,
                "period":            args.period,
                "train_end_date":    train_end or "auto",
                "val_end_date":      val_end or "auto",
                "tickers":           cfg.tickers,
                "lookback_days":     cfg.lookback_days,
                "device":            cfg.device,
            })
        except Exception as e:
            logger.warning("MLflow init failed (%s) — continuing without tracking.", e)
            mlflow_run = None
            run_id = None

    # ── Step 5: Train ────────────────────────────────────────────────────────
    history = train(
        train_ds=train_ds,
        val_ds=val_ds,
        num_features=num_features,
        hidden_size=cfg.hidden_size,
        checkpoint_path=checkpoint_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_quantile=cfg.lambda_quantile,
        lambda_vol=cfg.lambda_volatility,
        device=cfg.device,
        mlflow_run=mlflow_run,
    )

    # ── Step 6: Validate checkpoint can be loaded ────────────────────────────
    logger.info("Validating checkpoint can be reloaded ...")
    predictor = RiskPredictor(checkpoint_path=checkpoint_path, device=cfg.device)
    if not predictor.checkpoint_loaded:
        logger.error("❌ Checkpoint validation FAILED — file could not be reloaded!")
        if mlflow_run is not None and _MLFLOW_AVAILABLE:
            mlflow.set_tag("status", "checkpoint_invalid")
            mlflow.end_run(status="FAILED")
        sys.exit(1)
    logger.info("✅ Checkpoint validation passed.")

    # ── Step 7: Save metadata JSON (unchanged) ───────────────────────────────
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tickers": cfg.tickers_list,
        "interval": "1m",
        "train_end_date": train_end or "not specified (ratio split)",
        "val_end_date": val_end or "not specified (ratio split)",
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "num_features": num_features,
        "epochs_run": len(history["train"]),
        "lambda_quantile": cfg.lambda_quantile,
        "lambda_volatility": cfg.lambda_volatility,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "checkpoint_path": checkpoint_path,
        "best_val_total_loss": min(
            r["val_total_loss"] for r in history["train"]
        ) if history["train"] else None,
        "model_hyperparameters": {
            "num_features": num_features,
            "hidden_size": cfg.hidden_size,
        },
        "mlflow_run_id": run_id,  # added for traceability
        "warning": (
            "This model was trained on 1-minute bar data. "
            "Do NOT use a daily-trained checkpoint for 1-minute inference "
            "unless features and horizon are explicitly adapted."
        ),
    }

    meta_path = checkpoint_path.replace(".pt", "_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved to %s", meta_path)

    # ── Step 8: MLflow — log model + finalise run ────────────────────────────
    if mlflow_run is not None and _MLFLOW_AVAILABLE:
        try:
            # Log metadata JSON as artifact
            mlflow.log_artifact(meta_path, artifact_path="metadata")

            # Log full PyTorch model to Registry
            mlflow.pytorch.log_model(
                pytorch_model=predictor.model,
                artifact_path="tft_model",
                registered_model_name="PortfolioRiskTFT",
            )

            # Tag run with summary info
            mlflow.set_tags({
                "model_type":   "TFT",
                "framework":    "pytorch",
                "n_features":   str(num_features),
                "data_source":  "yfinance_1min",
                "status":       "trained",  # updated to "backtested" by backtest script
            })

            mlflow.end_run()
            logger.info(
                "MLflow run complete. View at %s/#/runs/%s",
                cfg.mlflow_tracking_uri, run_id,
            )
        except Exception as e:
            logger.warning("MLflow finalisation failed (%s) — run may be incomplete.", e)
            mlflow.end_run(status="FAILED")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TFT TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Metadata   : {meta_path}")
    print(f"  Train rows : {len(train_df)}  (samples: {len(train_ds)})")
    print(f"  Val rows   : {len(val_df)}    (samples: {len(val_ds)})")
    print(f"  Test rows  : {len(test_df)}   (unseen, for backtest)")
    if run_id:
        print(f"  MLflow Run : {cfg.mlflow_tracking_uri}/#/runs/{run_id}")
    if history["train"]:
        best = min(history["train"], key=lambda r: r["val_total_loss"])
        final = history["train"][-1]

        # Export simple training metrics as requested
        metrics = {
            "số_epoch": len(history["train"]),
            "best_val_loss": best["val_total_loss"],
            "final_train_loss": final["train_total_loss"],
            "final_val_loss": final["val_total_loss"],
            "feature_count": num_features,
            "hidden_size": cfg.hidden_size,
            "train_period": train_end or "auto_split",
            "val_period": val_end or "auto_split",
        }
        metrics_path = os.path.join(os.path.dirname(checkpoint_path), "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Training metrics saved to %s", metrics_path)

        print(f"  Best epoch : {best['epoch']}")
        print(f"  Best val   : total={best['val_total_loss']:.5f}  "
              f"q={best['val_quantile_loss']:.5f}  "
              f"vol={best['val_volatility_loss']:.5f}")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run backtest: python scripts/backtest_var.py")
    print("  2. Start the system: docker-compose up")
    print("  3. Dashboard shows TFT predictions (not baseline) automatically")

    return run_id  # returned for Airflow XCom / programmatic usage


if __name__ == "__main__":
    main()
