"""
model/trainer.py — Model Trainer with Early Stopping.

Handles the PyTorch training loop for the TFT model.
Saves:
  - Best model weights  → model/checkpoints/tft_best.pt
  - Loss history + meta → model/checkpoints/training_history.json
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam

from model.tft import TemporalFusionTransformer, tft_loss
from model.dataset import PortfolioRiskDataset
from config.settings import get_settings

logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class ModelTrainer:
    def __init__(
        self,
        dataset: PortfolioRiskDataset,
        batch_size: int = 64,
        val_split: float = 0.2,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = TemporalFusionTransformer(num_features=9, hidden_size=32).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.lr = lr

        # Split dataset
        if len(dataset) == 0:
            raise ValueError("Dataset is empty.")

        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        self.val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        self.early_stopping = EarlyStopping(patience=5)

        # History tracking (for thesis Section 3.4)
        self.train_losses: list[float] = []
        self.val_losses:   list[float] = []
        self.best_epoch:   int = 0

    def train(
        self,
        epochs: int = 20,
        checkpoint_path: str = "model/checkpoints/tft_best.pt",
        history_path: str = "model/checkpoints/training_history.json",
    ) -> dict:
        """
        Run training loop.

        Returns history dict with train/val losses per epoch.
        Saves:
          - Best model weights → checkpoint_path  (.pt)
          - Loss history + metadata → history_path (.json)
        """
        logger.info("Starting training on %s...", self.device)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        start_time = time.time()

        for epoch in range(epochs):
            # ── Train ─────────────────────────────────────────────────────────
            self.model.train()
            train_loss = 0.0
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(x)
                loss  = tft_loss(preds, y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * x.size(0)
            train_loss /= len(self.train_loader.dataset)

            # ── Validation ────────────────────────────────────────────────────
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in self.val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    preds = self.model(x)
                    loss  = tft_loss(preds, y)
                    val_loss += loss.item() * x.size(0)
            if len(self.val_loader.dataset) > 0:
                val_loss /= len(self.val_loader.dataset)

            # ── Record ────────────────────────────────────────────────────────
            self.train_losses.append(round(train_loss, 6))
            self.val_losses.append(round(val_loss, 6))

            logger.info(
                "Epoch %d/%d — Train Loss: %.6f — Val Loss: %.6f",
                epoch + 1, epochs, train_loss, val_loss,
            )

            # ── Save best checkpoint ──────────────────────────────────────────
            improved    = val_loss <= self.early_stopping.best_loss
            should_stop = self.early_stopping(val_loss)

            if improved:
                torch.save(self.model.state_dict(), checkpoint_path)
                self.best_epoch = epoch + 1
                logger.info(
                    "  ✔ Best checkpoint saved (epoch %d, val=%.6f)",
                    self.best_epoch, val_loss,
                )

            if should_stop:
                logger.info("Early stopping triggered at epoch %d", epoch + 1)
                break

        # ── Save history JSON ─────────────────────────────────────────────────
        elapsed = round(time.time() - start_time, 2)
        history = {
            "metadata": {
                "trained_at":            datetime.now().isoformat(),
                "epochs_ran":            len(self.train_losses),
                "best_epoch":            self.best_epoch,
                "best_val_loss":         min(self.val_losses),
                "final_train_loss":      self.train_losses[-1],
                "total_training_seconds": elapsed,
                "device":                str(self.device),
                "batch_size":            self.batch_size,
                "learning_rate":         self.lr,
                "n_train_samples":       len(self.train_loader.dataset),
                "n_val_samples":         len(self.val_loader.dataset),
            },
            "train_losses": self.train_losses,
            "val_losses":   self.val_losses,
        }
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        logger.info("Training history saved → %s", history_path)
        logger.info(
            "Training complete — best_epoch=%d  best_val=%.6f  time=%.1fs",
            self.best_epoch, min(self.val_losses), elapsed,
        )
        return history


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = get_settings()

    try:
        ds = PortfolioRiskDataset("data/raw/training_data.parquet", seq_len=60, pred_len=5)
        trainer = ModelTrainer(ds, batch_size=64, device=cfg.device)
        history = trainer.train(
            epochs=30,
            checkpoint_path="model/checkpoints/tft_best.pt",
            history_path="model/checkpoints/training_history.json",
        )
        print("\n=== Training Summary ===")
        print(f"  Epochs ran    : {history['metadata']['epochs_ran']}")
        print(f"  Best epoch    : {history['metadata']['best_epoch']}")
        print(f"  Best val loss : {history['metadata']['best_val_loss']:.6f}")
        print(f"  Train samples : {history['metadata']['n_train_samples']:,}")
        print(f"  Val samples   : {history['metadata']['n_val_samples']:,}")
        print(f"  Time (s)      : {history['metadata']['total_training_seconds']}")
    except Exception as e:
        logger.error("Training failed: %s", e)
        raise
