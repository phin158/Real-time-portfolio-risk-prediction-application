"""
api/routers/portfolio.py — REST endpoints for portfolio logic.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

class PortfolioRequest(BaseModel):
    weights: dict[str, float] = Field(..., json_schema_extra={"example": {"AAPL": 0.5, "MSFT": 0.5}})

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
    Calculates Portfolio VaR and Volatility based on current model predictions
    and requested weights.
    """
    from api.main import risk_service
    if risk_service is None:
        raise HTTPException(status_code=503, detail="Service starting up")
        
    metrics = risk_service.get_current_metrics()
    preds = metrics.get("predictions", {})
    
    if not preds:
        raise HTTPException(status_code=400, detail="No predictions available yet. Still warming up.")
        
    # Basic Weighted Sum for demonstration. 
    # Real portfolio VaR should account for covariance, but since TFT outputs VaR per asset,
    # we can do an approximation or just return the weighted average.
    port_var99 = 0.0
    port_var95 = 0.0
    port_vol = 0.0
    
    for sym, w in request.weights.items():
        sym = sym.upper()
        if sym in preds:
            port_var99 += w * preds[sym]["var_99"]
            port_var95 += w * preds[sym]["var_95"]
            port_vol += w * preds[sym]["vol_forecast"]
        else:
            logger.warning(f"Symbol {sym} not in current predictions.")
            
    return {
        "portfolio_var_99": port_var99,
        "portfolio_var_95": port_var95,
        "portfolio_vol_forecast": port_vol,
        "weights_used": request.weights
    }
