from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from model8 import analyze_and_forecast_enhanced

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
)

class ForecastRequest(BaseModel):
    country: str
    operator: str
    horizon: int = 8  # Default: 2 years

@app.post("/api/forecast")
async def get_forecast(request: ForecastRequest):
    try:
        # Load data (replace with your actual data loader)
        ts_cleaned, _ = prepare_quarterly_data_enhanced(request.country, request.operator)
        
        # Generate forecast
        results, _, _ = analyze_and_forecast_enhanced(
            country=request.country,
            operator=request.operator,
            variance_method='hybrid',
            zero_method='smart_interpolation'
        )
        
        # Format response
        return {
            "historical": ts_cleaned.to_dict(),
            "forecast": results["Predicted"].tolist(),
            "dates": results.index.astype(str).tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
