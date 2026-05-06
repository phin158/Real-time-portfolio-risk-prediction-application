"""
model/trainer.py — Model Trainer with Early Stopping.

Handles the PyTorch training loop for the TFT model.
"""
from __future__ import annotations

import logging
import os
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from typing import Tuple

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
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model = TemporalFusionTransformer(num_features=9, hidden_size=32).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        
        # Split dataset
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        
        if len(dataset) == 0:
            raise ValueError("Dataset is empty.")
            
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        self.early_stopping = EarlyStopping(patience=5)
        
    def train(self, epochs: int = 20, checkpoint_path: str = "model/checkpoints/tft_best.pt") -> None:
        logger.info("Starting training on %s...", self.device)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                preds = self.model(x)
                loss = tft_loss(preds, y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * x.size(0)
                
            train_loss /= len(self.train_loader.dataset)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in self.val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    preds = self.model(x)
                    loss = tft_loss(preds, y)
                    val_loss += loss.item() * x.size(0)
            
            if len(self.val_loader.dataset) > 0:
                val_loss /= len(self.val_loader.dataset)
            
            logger.info("Epoch %d/%d - Train Loss: %.4f - Val Loss: %.4f", epoch + 1, epochs, train_loss, val_loss)
            
            if self.early_stopping(val_loss):
                logger.info("Early stopping triggered at epoch %d", epoch + 1)
                break
                
            # Save if best
            if self.early_stopping.best_loss == val_loss:
                torch.save(self.model.state_dict(), checkpoint_path)
                logger.debug("Checkpoint saved to %s", checkpoint_path)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = get_settings()
    
    try:
        ds = PortfolioRiskDataset("data/raw/training_data.parquet", seq_len=60, pred_len=5)
        trainer = ModelTrainer(ds, batch_size=64, device=cfg.device)
        trainer.train(epochs=10)
    except Exception as e:
        logger.error("Training failed: %s", e)
