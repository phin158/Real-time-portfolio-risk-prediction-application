"""
portfolio/ — Portfolio-level risk aggregation package.

Provides mathematically correct portfolio risk metrics:
- Covariance-based portfolio volatility
- Parametric VaR using Normal distribution
- Historical VaR using empirical return quantiles
- CVaR / Expected Shortfall
"""
from portfolio.risk_aggregator import PortfolioRiskAggregator

__all__ = ["PortfolioRiskAggregator"]
