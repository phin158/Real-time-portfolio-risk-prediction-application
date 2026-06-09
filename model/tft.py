"""
model/tft.py — Simplified Temporal Fusion Transformer in PyTorch.

Predicts quantiles for returns (1%, 5%, 50%) to derive VaR, 
and a point forecast for future volatility.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedResidualNetwork(nn.Module):
    """Simplified GRN: Dense -> ELU -> Dense -> GLU -> Add -> LayerNorm"""
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2) # *2 for GLU
        self.dropout = nn.Dropout(dropout)
        
        # Project input to hidden_size if necessary for residual connection
        if input_size != hidden_size:
            self.skip = nn.Linear(input_size, hidden_size)
        else:
            self.skip = nn.Identity()
            
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        
        # GLU
        out = F.glu(out, dim=-1)
        
        # Add and Norm
        return self.norm(residual + out)

class VariableSelectionNetwork(nn.Module):
    """Simplified VSN for continuous features."""
    def __init__(self, num_features: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.num_features = num_features
        # Flattened input -> feature weights
        self.weight_net = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, num_features),
            nn.Softmax(dim=-1)
        )
        # Individual feature transforms
        self.feature_nets = nn.ModuleList([
            GatedResidualNetwork(1, hidden_size, dropout) 
            for _ in range(num_features)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, num_features)
        weights = self.weight_net(x) # (batch, seq_len, num_features)
        
        transformed_features = []
        for i in range(self.num_features):
            feat = x[..., i:i+1] # (batch, seq_len, 1)
            t_feat = self.feature_nets[i](feat) # (batch, seq_len, hidden)
            transformed_features.append(t_feat)
            
        # Stack -> (batch, seq_len, num_features, hidden_size)
        stacked = torch.stack(transformed_features, dim=2)
        
        # Multiply by weights -> (batch, seq_len, num_features, hidden_size)
        weighted = stacked * weights.unsqueeze(-1)
        
        # Sum across features -> (batch, seq_len, hidden_size)
        return weighted.sum(dim=2)

class TemporalFusionTransformer(nn.Module):
    """
    Simplified TFT for portfolio risk prediction.
    Outputs: [Return 1% Q, Return 5% Q, Return 50% Q, Volatility]
    """
    def __init__(
        self,
        num_features: int = 12,
        hidden_size: int = 64,
        lstm_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vsn = VariableSelectionNetwork(num_features, hidden_size, dropout)
        
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.output_layer = nn.Linear(hidden_size, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, num_features)
        
        # Variable Selection
        vsn_out = self.vsn(x) # (batch, seq_len, hidden_size)
        
        # LSTM Encoding
        lstm_out, _ = self.lstm(vsn_out) # (batch, seq_len, hidden_size)
        
        # Self-Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out) # (batch, seq_len, hidden_size)
        
        # Take the last time step for prediction
        last_step = attn_out[:, -1, :] # (batch, hidden_size)
        
        # Output layer
        out = self.output_layer(last_step) # (batch, 4)
        return out

def quantile_loss(preds: torch.Tensor, target: torch.Tensor, quantiles: list[float]) -> torch.Tensor:
    """
    Quantile loss for VaR prediction.
    preds: (batch, 3) for quantiles
    target: (batch,) true returns
    """
    losses = []
    for i, q in enumerate(quantiles):
        err = target - preds[:, i]
        loss = torch.max((q - 1) * err, q * err)
        losses.append(loss)
    return torch.stack(losses, dim=1).mean()

def tft_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    lambda_quantile: float = 1.0,
    lambda_volatility: float = 0.1,
) -> torch.Tensor:
    """
    Combined loss: Quantile loss for returns + MSE for volatility.

    IMPORTANT: Quantile loss on returns and MSE on volatility have DIFFERENT scales.
    Using 1:1 weighting (the old approach) is not justified.
    The default LAMBDA_VOLATILITY=0.1 is a reasonable starting point — tune as needed.

    Formula:
        total_loss = lambda_quantile * quantile_loss + lambda_volatility * vol_loss

    Args:
        preds:             (batch, 4) — [Q0.01, Q0.05, Q0.50, Volatility]
        targets:           (batch, 2) — [target_return, target_volatility]
        lambda_quantile:   Weight for quantile loss component (default 1.0).
        lambda_volatility: Weight for volatility MSE loss (default 0.1).

    Returns:
        Scalar total loss tensor.
    """
    # Returns (VaR quantile loss)
    target_returns = targets[:, 0]
    q_loss = quantile_loss(preds[:, :3], target_returns, quantiles=[0.01, 0.05, 0.50])

    # Volatility (MSE)
    target_vols = targets[:, 1]
    vol_loss = F.mse_loss(preds[:, 3], target_vols)

    # Weighted combination
    total = lambda_quantile * q_loss + lambda_volatility * vol_loss
    return total


def tft_loss_components(
    preds: torch.Tensor,
    targets: torch.Tensor,
    lambda_quantile: float = 1.0,
    lambda_volatility: float = 0.1,
) -> dict:
    """
    Same as tft_loss but returns individual components for logging.

    Returns:
        Dict with: total_loss, quantile_loss, volatility_loss (all tensors).
    """
    target_returns = targets[:, 0]
    q_loss = quantile_loss(preds[:, :3], target_returns, quantiles=[0.01, 0.05, 0.50])

    target_vols = targets[:, 1]
    vol_loss = F.mse_loss(preds[:, 3], target_vols)

    total = lambda_quantile * q_loss + lambda_volatility * vol_loss
    return {
        "total_loss": total,
        "quantile_loss": q_loss,
        "volatility_loss": vol_loss,
    }

if __name__ == "__main__":
    model = TemporalFusionTransformer(num_features=12, hidden_size=64)
    x = torch.randn(16, 60, 12)  # (batch_size, seq_len, num_features)
    out = model(x)
    print(f"Model output shape: {out.shape}")  # Expected: (16, 4)
    print(f"  hidden_size=64, num_features=12 ✅")

    targets = torch.randn(16, 2)  # [return, vol]
    loss = tft_loss(out, targets)
    print(f"Loss: {loss.item():.4f}")
