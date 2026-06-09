"""
tests/test_portfolio_risk.py — Unit tests for portfolio risk aggregation formulas.

Tests verify:
1. Covariance-based portfolio volatility formula: w.T @ Sigma @ w
2. Parametric VaR using z-score formula
3. Historical VaR using empirical quantiles
4. CVaR / Expected Shortfall calculation
5. Insufficient history warning (reliable=false when < 390 bars)
6. Missing checkpoint fallback (verified in Phase 2 — placeholder here)
7. No NaN passed to model (Phase 2 — placeholder here)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.risk_aggregator import PortfolioRiskAggregator, Z_05, Z_01, MIN_FULL_OBS


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def aggregator() -> PortfolioRiskAggregator:
    """Default aggregator with 1-minute annualisation factor."""
    return PortfolioRiskAggregator()


@pytest.fixture
def sample_returns_large(rng=None) -> np.ndarray:
    """500×3 return matrix with known covariance structure."""
    rng = np.random.default_rng(42)
    mu = np.array([0.0001, 0.00015, 0.00012])
    cov_true = np.array([
        [0.0001,  0.00005, 0.00003],
        [0.00005, 0.00012, 0.00004],
        [0.00003, 0.00004, 0.00009],
    ])
    return rng.multivariate_normal(mu, cov_true, size=500)


@pytest.fixture
def equal_weights_3() -> np.ndarray:
    return np.array([1/3, 1/3, 1/3])


@pytest.fixture
def symbols_3() -> list:
    return ["AAPL", "MSFT", "GOOGL"]


# ── Test 1: Covariance Portfolio Volatility ───────────────────────────────────

class TestCovariancePortfolioVolatility:
    """Verify the formula: sigma_p = sqrt(w.T @ Sigma @ w)"""

    def test_formula_matches_aggregator(
        self, aggregator, sample_returns_large, equal_weights_3, symbols_3
    ):
        """Aggregator must return portfolio volatility == sqrt(w.T @ cov @ w)."""
        w = equal_weights_3
        returns = sample_returns_large

        # Compute manually
        cov_manual = np.cov(returns.T, ddof=1)
        port_var_manual = w @ cov_manual @ w
        sigma_manual = float(np.sqrt(port_var_manual))

        # Compute via aggregator
        result = aggregator.compute_portfolio_risk(returns, w, symbols_3)

        assert result["portfolio_volatility_horizon"] == pytest.approx(
            sigma_manual, rel=1e-6
        ), "Portfolio volatility must equal sqrt(w.T @ Sigma @ w)"

    def test_portfolio_vol_less_than_max_asset_vol(
        self, aggregator, sample_returns_large, equal_weights_3, symbols_3
    ):
        """Diversification: portfolio vol should be <= max individual asset vol."""
        returns = sample_returns_large
        result = aggregator.compute_portfolio_risk(returns, equal_weights_3, symbols_3)

        # Individual asset vols
        asset_vols = [float(np.std(returns[:, i], ddof=1)) for i in range(3)]
        max_asset_vol = max(asset_vols)

        assert result["portfolio_volatility_horizon"] <= max_asset_vol + 1e-10, (
            "Portfolio volatility should not exceed maximum individual asset volatility "
            "(diversification benefit)"
        )

    def test_single_asset_portfolio(self, aggregator):
        """Single asset portfolio: sigma_p must equal asset sigma."""
        rng = np.random.default_rng(1)
        returns = rng.normal(0.0001, 0.01, size=(200, 1))
        weights = np.array([1.0])
        symbols = ["AAPL"]

        result = aggregator.compute_portfolio_risk(returns, weights, symbols)
        expected_vol = float(np.std(returns[:, 0], ddof=1))

        assert result["portfolio_volatility_horizon"] == pytest.approx(
            expected_vol, rel=1e-6
        )

    def test_weight_normalisation(
        self, aggregator, sample_returns_large, symbols_3
    ):
        """Weights should be auto-normalised; result should be same as with sum=1 weights."""
        w_unnorm = np.array([30.0, 30.0, 40.0])
        w_norm   = w_unnorm / w_unnorm.sum()

        result_unnorm = aggregator.compute_portfolio_risk(
            sample_returns_large, w_unnorm, symbols_3
        )
        result_norm = aggregator.compute_portfolio_risk(
            sample_returns_large, w_norm, symbols_3
        )

        assert result_unnorm["portfolio_volatility_horizon"] == pytest.approx(
            result_norm["portfolio_volatility_horizon"], rel=1e-6
        ), "Unnormalised and normalised weights must give same volatility"


# ── Test 2: Parametric VaR Formula ────────────────────────────────────────────

class TestParametricVaR:
    """Verify parametric VaR formula: VaR = max(0, -(mu + z * sigma))"""

    def test_var_formula_correctness(self, aggregator):
        """Manually verify parametric VaR formula for known inputs."""
        rng = np.random.default_rng(99)
        returns_1d = rng.normal(0.0001, 0.01, size=300)
        portfolio_returns = returns_1d
        weights = np.array([1.0])
        cov = np.array([[float(np.var(returns_1d, ddof=1))]])

        result = aggregator.compute_parametric_var(weights, cov, portfolio_returns)

        mu    = float(np.mean(portfolio_returns))
        sigma = float(np.sqrt(float(weights @ cov @ weights)))

        expected_var_95 = max(0.0, -(mu + Z_05 * sigma))
        expected_var_99 = max(0.0, -(mu + Z_01 * sigma))

        assert result["var_95"] == pytest.approx(expected_var_95, rel=1e-6)
        assert result["var_99"] == pytest.approx(expected_var_99, rel=1e-6)

    def test_var_99_geq_var_95(
        self, aggregator, sample_returns_large, equal_weights_3, symbols_3
    ):
        """VaR 99% must be >= VaR 95% (higher confidence = larger loss threshold)."""
        result = aggregator.compute_portfolio_risk(
            sample_returns_large, equal_weights_3, symbols_3
        )
        assert result["portfolio_var_99_parametric"] >= result["portfolio_var_95_parametric"], (
            "VaR 99% must be >= VaR 95%"
        )

    def test_var_non_negative(
        self, aggregator, sample_returns_large, equal_weights_3, symbols_3
    ):
        """VaR must always be non-negative (it represents a loss, not a gain)."""
        result = aggregator.compute_portfolio_risk(
            sample_returns_large, equal_weights_3, symbols_3
        )
        assert result["portfolio_var_95_parametric"] >= 0.0
        assert result["portfolio_var_99_parametric"] >= 0.0

    def test_z_scores_correct(self):
        """Verify z-score constants are approximately correct."""
        from scipy import stats
        assert Z_05 == pytest.approx(stats.norm.ppf(0.05), rel=1e-3)
        assert Z_01 == pytest.approx(stats.norm.ppf(0.01), rel=1e-3)


# ── Test 3: Historical VaR ─────────────────────────────────────────────────────

class TestHistoricalVaR:
    """Verify historical VaR = -quantile(portfolio_returns, p)"""

    def test_var_equals_negative_quantile(self, aggregator):
        """Historical VaR must equal -quantile at the correct level."""
        rng = np.random.default_rng(5)
        port_returns = rng.normal(0, 0.01, size=1000)

        result = aggregator.compute_historical_var(port_returns)

        expected_95 = max(0.0, -float(np.quantile(port_returns, 0.05)))
        expected_99 = max(0.0, -float(np.quantile(port_returns, 0.01)))

        assert result["var_95"] == pytest.approx(expected_95, rel=1e-6)
        assert result["var_99"] == pytest.approx(expected_99, rel=1e-6)

    def test_historical_var_99_geq_95(self, aggregator):
        """Historical VaR 99% must be >= VaR 95%."""
        rng = np.random.default_rng(6)
        port_returns = rng.normal(0, 0.01, size=500)
        result = aggregator.compute_historical_var(port_returns)
        assert result["var_99"] >= result["var_95"]

    def test_historical_var_in_full_result(
        self, aggregator, sample_returns_large, equal_weights_3, symbols_3
    ):
        """Full result must include historical VaR keys."""
        result = aggregator.compute_portfolio_risk(
            sample_returns_large, equal_weights_3, symbols_3
        )
        assert "portfolio_var_95_historical" in result
        assert "portfolio_var_99_historical" in result
        assert result["portfolio_var_95_historical"] >= 0.0
        assert result["portfolio_var_99_historical"] >= 0.0

    def test_insufficient_data_returns_zero(self, aggregator):
        """If fewer than 2 observations, return zero VaR (not error)."""
        result = aggregator.compute_historical_var(np.array([0.01]))
        assert result["var_95"] == 0.0
        assert result["var_99"] == 0.0


# ── Test 4: CVaR / Expected Shortfall ─────────────────────────────────────────

class TestCVaR:
    """Verify CVaR = -mean(returns in the tail below VaR threshold)"""

    def test_cvar_geq_var(self, aggregator):
        """CVaR must be >= historical VaR (ES >= VaR always)."""
        rng = np.random.default_rng(7)
        port_returns = rng.normal(0, 0.015, size=1000)

        var_result  = aggregator.compute_historical_var(port_returns)
        cvar_result = aggregator.compute_cvar(port_returns)

        assert cvar_result["cvar_95"] >= var_result["var_95"] - 1e-10, (
            "CVaR_95 must be >= VaR_95"
        )
        assert cvar_result["cvar_99"] >= var_result["var_99"] - 1e-10, (
            "CVaR_99 must be >= VaR_99"
        )

    def test_cvar_formula_correctness(self, aggregator):
        """CVaR must equal mean of returns in the tail."""
        rng = np.random.default_rng(8)
        port_returns = rng.normal(0, 0.01, size=2000)

        result = aggregator.compute_cvar(port_returns)

        q05 = np.quantile(port_returns, 0.05)
        q01 = np.quantile(port_returns, 0.01)
        expected_95 = max(0.0, -float(np.mean(port_returns[port_returns <= q05])))
        expected_99 = max(0.0, -float(np.mean(port_returns[port_returns <= q01])))

        assert result["cvar_95"] == pytest.approx(expected_95, rel=1e-6)
        assert result["cvar_99"] == pytest.approx(expected_99, rel=1e-6)

    def test_cvar_in_full_result(
        self, aggregator, sample_returns_large, equal_weights_3, symbols_3
    ):
        """Full result must include CVaR keys."""
        result = aggregator.compute_portfolio_risk(
            sample_returns_large, equal_weights_3, symbols_3
        )
        assert "portfolio_cvar_95" in result
        assert "portfolio_cvar_99" in result
        assert result["portfolio_cvar_99"] >= result["portfolio_cvar_95"] - 1e-10


# ── Test 5: Insufficient History Warning ─────────────────────────────────────

class TestInsufficientHistory:
    """Verify reliable=false and warning when observations < MIN_FULL_OBS."""

    def test_reliable_false_when_short_history(self, aggregator, symbols_3):
        """With < 390 bars, reliable must be False and warning must be set."""
        rng = np.random.default_rng(9)
        # Only 100 bars — not enough for stable covariance
        short_returns = rng.normal(0, 0.01, size=(100, 3))
        weights = np.array([0.4, 0.4, 0.2])

        result = aggregator.compute_portfolio_risk(short_returns, weights, symbols_3)

        assert result["reliable"] is False, "reliable must be False with only 100 bars"
        assert result["warning"] is not None, "warning must be set when history is short"
        assert "100" in result["warning"] or "390" in result["warning"], (
            "warning message should mention bar counts"
        )

    def test_reliable_true_when_enough_history(
        self, aggregator, sample_returns_large, equal_weights_3, symbols_3
    ):
        """With >= 390 bars, reliable must be True."""
        result = aggregator.compute_portfolio_risk(
            sample_returns_large, equal_weights_3, symbols_3
        )
        assert result["reliable"] is True, (
            f"reliable must be True with {len(sample_returns_large)} bars (>= {MIN_FULL_OBS})"
        )
        assert result["warning"] is None, "No warning should be set with sufficient history"

    def test_empty_input_returns_no_error(self, aggregator):
        """Empty returns matrix should return empty result, not raise."""
        result = aggregator.compute_portfolio_risk(
            np.empty((0, 3)), np.array([0.4, 0.4, 0.2]), ["A", "B", "C"]
        )
        assert result["reliable"] is False
        assert result["warning"] is not None

    def test_all_zero_weights_returns_warning(self, aggregator, sample_returns_large, symbols_3):
        """All-zero weights should return warning, not raise."""
        result = aggregator.compute_portfolio_risk(
            sample_returns_large, np.array([0.0, 0.0, 0.0]), symbols_3
        )
        assert result["reliable"] is False
        assert result["warning"] is not None


# ── Test 6: Response Schema ────────────────────────────────────────────────────

class TestResponseSchema:
    """Verify the full response contains all required keys."""

    REQUIRED_KEYS = [
        "weights_used",
        "n_observations",
        "portfolio_return_latest",
        "portfolio_volatility_horizon",
        "portfolio_volatility_annualized",
        "portfolio_var_95_parametric",
        "portfolio_var_99_parametric",
        "portfolio_var_95_historical",
        "portfolio_var_99_historical",
        "portfolio_cvar_95",
        "portfolio_cvar_99",
        "correlation_matrix",
        "covariance_matrix",
        "method_used",
        "reliable",
        "warning",
    ]

    def test_full_result_has_all_keys(
        self, aggregator, sample_returns_large, equal_weights_3, symbols_3
    ):
        """Full result must contain all required response keys."""
        result = aggregator.compute_portfolio_risk(
            sample_returns_large, equal_weights_3, symbols_3
        )
        for key in self.REQUIRED_KEYS:
            assert key in result, f"Missing key: {key}"

    def test_matrices_have_correct_shape(
        self, aggregator, sample_returns_large, equal_weights_3, symbols_3
    ):
        """Correlation and covariance matrices must be N×N."""
        result = aggregator.compute_portfolio_risk(
            sample_returns_large, equal_weights_3, symbols_3
        )
        corr_df = pd.DataFrame(result["correlation_matrix"])
        cov_df  = pd.DataFrame(result["covariance_matrix"])

        N = len(symbols_3)
        assert corr_df.shape == (N, N), f"Corr matrix shape {corr_df.shape} != ({N}, {N})"
        assert cov_df.shape  == (N, N), f"Cov matrix shape {cov_df.shape} != ({N}, {N})"

    def test_correlation_diagonal_is_one(
        self, aggregator, sample_returns_large, equal_weights_3, symbols_3
    ):
        """Diagonal of correlation matrix must be 1.0."""
        result = aggregator.compute_portfolio_risk(
            sample_returns_large, equal_weights_3, symbols_3
        )
        corr_df = pd.DataFrame(result["correlation_matrix"])
        diag = np.diag(corr_df.values)
        np.testing.assert_allclose(diag, 1.0, atol=1e-6)

    def test_covariance_is_positive_semidefinite(
        self, aggregator, sample_returns_large, symbols_3
    ):
        """Covariance matrix must be positive semidefinite (all eigenvalues >= 0)."""
        w = np.array([0.4, 0.4, 0.2])
        result = aggregator.compute_portfolio_risk(sample_returns_large, w, symbols_3)
        cov = pd.DataFrame(result["covariance_matrix"]).values
        eigenvalues = np.linalg.eigvals(cov)
        assert np.all(eigenvalues >= -1e-9), (
            f"Covariance matrix is not PSD: negative eigenvalues {eigenvalues}"
        )


# ── Test 7: FeatureEngineer Integration ───────────────────────────────────────

class TestFeatureEngineerIntegration:
    """Verify new get_returns_matrix() and get_covariance_matrix() methods."""

    def test_returns_matrix_shape(self):
        """get_returns_matrix must return (T, N) array."""
        from feature_engineering.engineer import FeatureEngineer
        from data_pipeline.schemas import ValidatedTick
        from datetime import datetime, timezone, timedelta

        symbols = ["AAPL", "MSFT", "GOOGL"]
        engineer = FeatureEngineer(symbols=symbols)

        rng = np.random.default_rng(10)
        base_prices = {"AAPL": 180.0, "MSFT": 370.0, "GOOGL": 155.0}
        base_ts = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)

        # Feed 450 ticks to exceed the 390-bar window
        for i in range(450):
            ts = base_ts + timedelta(minutes=i)
            for sym in symbols:
                price = base_prices[sym] * np.exp(rng.normal(0, 0.001))
                base_prices[sym] = price
                tick = ValidatedTick(
                    symbol=sym, timestamp=ts,
                    open=price, high=price * 1.001, low=price * 0.999,
                    close=price, volume=1_000_000.0, is_valid=True,
                )
                engineer.update(tick)

        matrix, included = engineer.get_returns_matrix(symbols=symbols, window=390)
        assert matrix.ndim == 2, "Returns matrix must be 2D"
        assert matrix.shape[1] == len(included), "Column count must equal included symbols"
        assert matrix.shape[0] > 0, "Row count must be > 0"

    def test_covariance_matrix_symmetric(self):
        """get_covariance_matrix must return a symmetric matrix."""
        from feature_engineering.engineer import FeatureEngineer
        from data_pipeline.schemas import ValidatedTick
        from datetime import datetime, timezone, timedelta

        symbols = ["AAPL", "MSFT"]
        engineer = FeatureEngineer(symbols=symbols)

        rng = np.random.default_rng(11)
        base_prices = {"AAPL": 180.0, "MSFT": 370.0}
        base_ts = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)

        for i in range(450):
            ts = base_ts + timedelta(minutes=i)
            for sym in symbols:
                price = base_prices[sym] * np.exp(rng.normal(0, 0.001))
                base_prices[sym] = price
                tick = ValidatedTick(
                    symbol=sym, timestamp=ts,
                    open=price, high=price * 1.001, low=price * 0.999,
                    close=price, volume=500_000.0, is_valid=True,
                )
                engineer.update(tick)

        cov, included = engineer.get_covariance_matrix(symbols=symbols, window=390)
        if len(included) == 2:
            np.testing.assert_allclose(cov, cov.T, atol=1e-10, err_msg="Cov matrix not symmetric")

    def test_history_status_fields(self):
        """get_history_status must return all required fields."""
        from feature_engineering.engineer import FeatureEngineer
        from data_pipeline.schemas import ValidatedTick
        from datetime import datetime, timezone

        engineer = FeatureEngineer(symbols=["AAPL"])
        # Feed only 10 ticks (below MIN_HISTORY_BASIC=60)
        for i in range(10):
            tick = ValidatedTick(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc),
                open=180.0, high=181.0, low=179.0, close=180.0, volume=1000.0,
                is_valid=True,
            )
            engineer.update(tick)

        status = engineer.get_history_status("AAPL")
        assert "symbol" in status
        assert "tick_count" in status
        assert "basic_ready" in status
        assert "full_ready" in status
        assert "reliable" in status
        assert "warning" in status
        assert status["basic_ready"] is False
        assert status["full_ready"] is False
        assert status["warning"] is not None
