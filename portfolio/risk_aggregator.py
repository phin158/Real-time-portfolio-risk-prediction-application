"""
portfolio/risk_aggregator.py — Mathematically correct portfolio risk aggregation.

This module replaces the INCORRECT weighted-average VaR approach with proper
covariance-based portfolio risk metrics.

KEY FORMULAS:
============
Portfolio Return:
    r_p = w.T @ r          (dot product of weights and asset returns)

Portfolio Variance:
    σ²_p = w.T @ Σ @ w    (where Σ is the covariance matrix of asset returns)

Portfolio Volatility:
    σ_p = sqrt(σ²_p)

Parametric VaR (Normal distribution assumption):
    VaR_95 = max(0, -(μ_p + z_05 * σ_p))   where z_05 = -1.645
    VaR_99 = max(0, -(μ_p + z_01 * σ_p))   where z_01 = -2.326

Historical VaR (empirical quantile method):
    portfolio_returns = returns_matrix @ weights
    VaR_95 = -quantile(portfolio_returns, 0.05)
    VaR_99 = -quantile(portfolio_returns, 0.01)

CVaR / Expected Shortfall:
    CVaR_95 = -mean(portfolio_returns[portfolio_returns <= quantile_0.05])
    CVaR_99 = -mean(portfolio_returns[portfolio_returns <= quantile_0.01])

IMPORTANT NOTE ON HORIZON:
- 1-minute bar volatility σ_1min is NOT the same as annualized volatility.
- For 1-minute VaR, use σ_p directly (horizon = 1 bar).
- Annualized: σ_annualized = σ_1min * sqrt(252 * 390)
- Both horizon and annualized are returned for transparency.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Standard Normal quantiles for VaR
Z_05: float = -1.6449  # quantile at 5% (for VaR 95%)
Z_01: float = -2.3263  # quantile at 1% (for VaR 99%)

# Annualisation factor for 1-minute bars
# 1 trading day = 390 minutes, 1 trading year = 252 days
ANN_FACTOR_1MIN: float = float(np.sqrt(252 * 390))  # ≈ 313.47

# Minimum observations for reliable estimation
MIN_RELIABLE_OBS: int = 60    # basic minimum
MIN_FULL_OBS: int = 390       # for stable covariance estimation


class PortfolioRiskAggregator:
    """
    Computes portfolio-level risk metrics from aligned asset return series.

    This class accepts a returns matrix (time × assets) and portfolio weights,
    then computes parametric VaR, historical VaR, and CVaR using statistically
    correct formulas.

    Args:
        ann_factor: Annualisation multiplier for 1-minute bars.
                    Default = sqrt(252 * 390) ≈ 313.47.
    """

    def __init__(self, ann_factor: float = ANN_FACTOR_1MIN) -> None:
        self.ann_factor = ann_factor

    # ── Public API ────────────────────────────────────────────────────────────

    def compute_portfolio_risk(
        self,
        returns_matrix: np.ndarray,
        weights: np.ndarray,
        symbols: List[str],
    ) -> Dict:
        """
        Compute full portfolio risk metrics from aligned returns.

        Args:
            returns_matrix: Shape (T, N) — T timesteps, N assets.
                            Each column is aligned log returns for one symbol.
            weights:        Shape (N,) — portfolio weights summing to 1.0.
            symbols:        List of N symbol names (for labelling).

        Returns:
            Dictionary with all portfolio risk metrics, warnings, and metadata.
        """
        T, N = returns_matrix.shape
        assert len(weights) == N, f"Weight count {len(weights)} != asset count {N}"
        assert len(symbols) == N, f"Symbol count {len(symbols)} != asset count {N}"

        # Normalise weights to ensure they sum to 1
        w = np.array(weights, dtype=np.float64)
        w_sum = w.sum()
        if abs(w_sum) < 1e-9:
            return self._empty_result("All weights are zero.")
        w = w / w_sum  # normalise

        # Check data quality
        if T < 2:
            return self._empty_result("Fewer than 2 observations. Cannot compute risk.")

        # Determine reliability
        reliable = T >= MIN_FULL_OBS
        warning = None
        if T < MIN_RELIABLE_OBS:
            warning = (
                f"Insufficient history for model inference: {T}/{MIN_RELIABLE_OBS} bars available."
            )
        elif T < MIN_FULL_OBS:
            warning = (
                f"Insufficient history for stable covariance estimation: "
                f"{T}/{MIN_FULL_OBS} bars available. Results are provisional."
            )

        # ── Covariance matrix ────────────────────────────────────────────────
        cov_matrix = self.compute_covariance_matrix(returns_matrix)
        corr_matrix = self.compute_correlation_matrix(returns_matrix)

        # ── Portfolio returns series ─────────────────────────────────────────
        # Shape: (T,) — weighted sum of asset returns at each timestep
        portfolio_returns = returns_matrix @ w

        # ── Parametric VaR ───────────────────────────────────────────────────
        parametric = self.compute_parametric_var(w, cov_matrix, portfolio_returns)

        # ── Historical VaR & CVaR ────────────────────────────────────────────
        historical = self.compute_historical_var(portfolio_returns)
        cvar = self.compute_cvar(portfolio_returns)

        # ── Latest portfolio return ──────────────────────────────────────────
        latest_return = float(portfolio_returns[-1]) if len(portfolio_returns) > 0 else 0.0

        # ── Format matrices for JSON serialisation ──────────────────────────
        cov_df = pd.DataFrame(cov_matrix, index=symbols, columns=symbols)
        corr_df = pd.DataFrame(corr_matrix, index=symbols, columns=symbols)

        return {
            "weights_used": {sym: float(w[i]) for i, sym in enumerate(symbols)},
            "n_observations": T,
            "portfolio_return_latest": round(latest_return, 8),
            # Volatility for 1-bar horizon (same scale as log returns)
            "portfolio_volatility_horizon": round(parametric["sigma_p"], 8),
            # Annualised volatility
            "portfolio_volatility_annualized": round(
                parametric["sigma_p"] * self.ann_factor, 6
            ),
            # Parametric VaR (Normal distribution)
            "portfolio_var_95_parametric": round(parametric["var_95"], 8),
            "portfolio_var_99_parametric": round(parametric["var_99"], 8),
            # Historical VaR (empirical quantiles)
            "portfolio_var_95_historical": round(historical["var_95"], 8),
            "portfolio_var_99_historical": round(historical["var_99"], 8),
            # CVaR / Expected Shortfall
            "portfolio_cvar_95": round(cvar["cvar_95"], 8),
            "portfolio_cvar_99": round(cvar["cvar_99"], 8),
            # Matrices for transparency
            "correlation_matrix": corr_df.round(6).to_dict(),
            "covariance_matrix": cov_df.round(10).to_dict(),
            # Metadata
            "method_used": "parametric+historical",
            "reliable": reliable,
            "warning": warning,
        }

    def compute_covariance_matrix(self, returns_matrix: np.ndarray) -> np.ndarray:
        """
        Compute sample covariance matrix from returns.

        Args:
            returns_matrix: Shape (T, N).

        Returns:
            Covariance matrix of shape (N, N).
        """
        # np.cov expects (features, observations) → transpose
        if returns_matrix.shape[0] < 2:
            N = returns_matrix.shape[1]
            return np.zeros((N, N))
        cov = np.cov(returns_matrix.T, ddof=1)
        # Ensure it is a 2D matrix even for single asset
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        return cov

    def compute_correlation_matrix(self, returns_matrix: np.ndarray) -> np.ndarray:
        """
        Compute Pearson correlation matrix from returns.

        Returns:
            Correlation matrix of shape (N, N).
        """
        if returns_matrix.shape[0] < 2:
            N = returns_matrix.shape[1]
            return np.eye(N)
        return np.corrcoef(returns_matrix.T)

    def compute_parametric_var(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        portfolio_returns: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute parametric VaR under Normal distribution assumption.

        Formula:
            σ²_p = w.T @ Σ @ w
            σ_p  = sqrt(σ²_p)
            μ_p  = mean(portfolio_returns)
            VaR_95 = max(0, -(μ_p + z_05 * σ_p))   z_05 = -1.645
            VaR_99 = max(0, -(μ_p + z_01 * σ_p))   z_01 = -2.326

        The VaR is expressed as a positive loss figure. A VaR_95 of 0.02
        means we expect to lose at most 2% with 95% confidence.

        Args:
            weights:          Portfolio weight vector, shape (N,).
            cov_matrix:       Asset return covariance matrix, shape (N, N).
            portfolio_returns: Historical portfolio return series, shape (T,).

        Returns:
            Dict with mu_p, sigma_p, var_95, var_99.
        """
        # Portfolio variance: σ²_p = w.T @ Σ @ w
        portfolio_variance = float(weights @ cov_matrix @ weights)

        # Guard against numerical issues
        if portfolio_variance < 0:
            portfolio_variance = 0.0

        sigma_p = float(np.sqrt(portfolio_variance))

        # Expected portfolio return (historical mean)
        mu_p = float(np.mean(portfolio_returns)) if len(portfolio_returns) > 0 else 0.0

        # Parametric VaR
        # VaR = -(mu + z * sigma), expressed as positive loss
        var_95 = float(max(0.0, -(mu_p + Z_05 * sigma_p)))
        var_99 = float(max(0.0, -(mu_p + Z_01 * sigma_p)))

        return {
            "mu_p": mu_p,
            "sigma_p": sigma_p,
            "var_95": var_95,
            "var_99": var_99,
        }

    def compute_historical_var(self, portfolio_returns: np.ndarray) -> Dict[str, float]:
        """
        Compute Historical VaR from empirical return distribution.

        Formula:
            VaR_95 = -quantile(portfolio_returns, 0.05)
            VaR_99 = -quantile(portfolio_returns, 0.01)

        VaR is the LOSS THRESHOLD exceeded with probability p.
        E.g., VaR_95 is exceeded ~5% of the time. It is NOT the maximum loss.

        Args:
            portfolio_returns: Historical portfolio return series, shape (T,).

        Returns:
            Dict with var_95 and var_99.
        """
        if len(portfolio_returns) < 2:
            return {"var_95": 0.0, "var_99": 0.0}

        q05 = float(np.quantile(portfolio_returns, 0.05))
        q01 = float(np.quantile(portfolio_returns, 0.01))

        return {
            "var_95": float(max(0.0, -q05)),
            "var_99": float(max(0.0, -q01)),
        }

    def compute_cvar(self, portfolio_returns: np.ndarray) -> Dict[str, float]:
        """
        Compute CVaR / Expected Shortfall from empirical return distribution.

        CVaR (Conditional VaR) is the EXPECTED LOSS given that the loss
        exceeds the VaR threshold. It is always >= VaR.

        Formula:
            threshold_05 = quantile(portfolio_returns, 0.05)
            CVaR_95 = -mean(portfolio_returns[portfolio_returns <= threshold_05])

            threshold_01 = quantile(portfolio_returns, 0.01)
            CVaR_99 = -mean(portfolio_returns[portfolio_returns <= threshold_01])

        Args:
            portfolio_returns: Historical portfolio return series, shape (T,).

        Returns:
            Dict with cvar_95 and cvar_99.
        """
        if len(portfolio_returns) < 20:
            return {"cvar_95": 0.0, "cvar_99": 0.0}

        q05 = float(np.quantile(portfolio_returns, 0.05))
        q01 = float(np.quantile(portfolio_returns, 0.01))

        tail_95 = portfolio_returns[portfolio_returns <= q05]
        tail_99 = portfolio_returns[portfolio_returns <= q01]

        cvar_95 = float(max(0.0, -np.mean(tail_95))) if len(tail_95) > 0 else 0.0
        cvar_99 = float(max(0.0, -np.mean(tail_99))) if len(tail_99) > 0 else 0.0

        return {
            "cvar_95": cvar_95,
            "cvar_99": cvar_99,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_result(warning: str) -> Dict:
        """Return an empty risk result with a warning message."""
        return {
            "weights_used": {},
            "n_observations": 0,
            "portfolio_return_latest": 0.0,
            "portfolio_volatility_horizon": 0.0,
            "portfolio_volatility_annualized": 0.0,
            "portfolio_var_95_parametric": 0.0,
            "portfolio_var_99_parametric": 0.0,
            "portfolio_var_95_historical": 0.0,
            "portfolio_var_99_historical": 0.0,
            "portfolio_cvar_95": 0.0,
            "portfolio_cvar_99": 0.0,
            "correlation_matrix": {},
            "covariance_matrix": {},
            "method_used": "none",
            "reliable": False,
            "warning": warning,
        }


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("Testing PortfolioRiskAggregator …")
    rng = np.random.default_rng(42)

    # Simulate 500 bars of 1-minute returns for 3 assets
    T, N = 500, 3
    # Correlated returns
    mu = np.array([0.0001, 0.00015, 0.00012])
    cov_true = np.array([
        [0.0001, 0.00005, 0.00003],
        [0.00005, 0.00012, 0.00004],
        [0.00003, 0.00004, 0.00009],
    ])
    returns_matrix = rng.multivariate_normal(mu, cov_true, size=T)

    weights = np.array([0.4, 0.4, 0.2])
    symbols = ["AAPL", "MSFT", "GOOGL"]

    aggregator = PortfolioRiskAggregator()
    result = aggregator.compute_portfolio_risk(returns_matrix, weights, symbols)

    print(f"  ✅ Portfolio Volatility (horizon):   {result['portfolio_volatility_horizon']:.6f}")
    print(f"  ✅ Portfolio Volatility (annualized): {result['portfolio_volatility_annualized']:.4f}")
    print(f"  ✅ VaR 95% Parametric:  {result['portfolio_var_95_parametric']:.6f}")
    print(f"  ✅ VaR 99% Parametric:  {result['portfolio_var_99_parametric']:.6f}")
    print(f"  ✅ VaR 95% Historical:  {result['portfolio_var_95_historical']:.6f}")
    print(f"  ✅ VaR 99% Historical:  {result['portfolio_var_99_historical']:.6f}")
    print(f"  ✅ CVaR 95%:            {result['portfolio_cvar_95']:.6f}")
    print(f"  ✅ CVaR 99%:            {result['portfolio_cvar_99']:.6f}")
    print(f"  ✅ Reliable:            {result['reliable']}")
    print(f"  ✅ Method:              {result['method_used']}")

    # Verify covariance formula manually (use normalised weights — same as aggregator does)
    w_norm = weights / weights.sum()
    cov_computed = aggregator.compute_covariance_matrix(returns_matrix)
    port_var_manual = w_norm @ cov_computed @ w_norm
    port_vol_manual = np.sqrt(port_var_manual)
    assert abs(port_vol_manual - result["portfolio_volatility_horizon"]) < 1e-6, \
        "Portfolio volatility mismatch!"
    print(f"  ✅ Covariance formula verified: σ_p = {port_vol_manual:.8f}")

    # VaR 99% should be >= VaR 95%
    assert result["portfolio_var_99_parametric"] >= result["portfolio_var_95_parametric"], \
        "VaR 99% should be >= VaR 95%"
    assert result["portfolio_cvar_99"] >= result["portfolio_var_99_historical"], \
        "CVaR_99 should be >= VaR_99"
    print("  ✅ VaR ordering verified: VaR99 >= VaR95, CVaR >= VaR")

    print("\n✅ All PortfolioRiskAggregator tests passed.")
    sys.exit(0)
