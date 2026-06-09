"""
api/routers/portfolio.py — REST endpoints for portfolio-level risk calculation.

IMPORTANT CHANGE:
The old /api/portfolio endpoint used weighted-average VaR:
    portfolio_var = sum(weight_i * var_i)

This was MATHEMATICALLY INCORRECT because it ignores asset correlations.

The new implementation uses:
    portfolio_variance = w.T @ covariance_matrix @ w
    portfolio_volatility = sqrt(portfolio_variance)
    VaR (parametric) = -(mu_p + z * sigma_p)
    VaR (historical) = -quantile(portfolio_returns, 0.05 or 0.01)
    CVaR = expected loss beyond VaR threshold

See portfolio/risk_aggregator.py for the full mathematical details.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


class PortfolioRequest(BaseModel):
    weights: dict[str, float] = Field(
        ...,
        json_schema_extra={
            "example": {"AAPL": 0.3, "MSFT": 0.3, "GOOGL": 0.2, "AMZN": 0.1, "NVDA": 0.1}
        },
    )
    covariance_window: int = Field(
        default=390,
        ge=10,
        le=2000,
        description=(
            "Number of historical 1-minute bars to use for covariance estimation. "
            "At least 390 bars (≈1 trading day) recommended for reliable estimation."
        ),
    )

    @model_validator(mode="after")
    def validate_weights(self) -> "PortfolioRequest":
        """Ensure weights are non-negative and not all zero."""
        for sym, w in self.weights.items():
            if w < 0:
                raise ValueError(f"Weight for {sym} must be non-negative, got {w}")
        if all(w == 0 for w in self.weights.values()):
            raise ValueError("At least one weight must be non-zero")
        return self


@router.get("/metrics")
async def get_metrics():
    """Returns current state of all symbols (predictions + correlation)."""
    # Import locally to avoid circular dependency since risk_service is initialized in main
    from api.main import risk_service

    if risk_service is None:
        return {"status": "starting_up"}

    return risk_service.get_current_metrics()


@router.post("/portfolio")
async def calculate_portfolio(request: PortfolioRequest):
    """
    Compute portfolio-level VaR, CVaR, and volatility using the covariance matrix.

    This endpoint no longer uses the incorrect weighted-average VaR formula.

    The correct approach:
    1. Collect aligned log-return series for all requested symbols.
    2. Compute the rolling covariance matrix Σ from those returns.
    3. Compute portfolio volatility: σ_p = sqrt(w.T @ Σ @ w)
    4. Compute parametric VaR: -(μ_p + z * σ_p) where z = -1.645 (VaR 95%) or -2.326 (VaR 99%)
    5. Compute historical VaR: -quantile(portfolio_returns, 0.05 or 0.01)
    6. Compute CVaR: -mean(returns in the tail beyond VaR threshold) using historical simulation on portfolio returns.

    Note on CVaR calculation:
    CVaR (Conditional Value at Risk / Expected Shortfall) in the API response is calculated
    using historical simulation directly on the weighted portfolio returns distribution. E.g.,
    CVaR 95% is the average of the worst 5% of historical portfolio returns.

    Note on VaR definition:
    VaR is the loss threshold that is exceeded with a small probability over a
    specified horizon. VaR 95% is exceeded approximately 5% of the time, and
    VaR 99% is exceeded approximately 1% of the time. VaR is NOT the maximum loss.
    """
    from api.main import risk_service

    if risk_service is None:
        raise HTTPException(status_code=503, detail="Service starting up")

    try:
        result = risk_service.get_portfolio_risk(
            weights=request.weights,
            covariance_window=request.covariance_window,
        )
    except Exception as e:
        logger.error("Portfolio risk calculation failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Portfolio risk calculation error: {str(e)}",
        )

    return result
