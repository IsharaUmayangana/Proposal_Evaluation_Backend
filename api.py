from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict

from predict_utils import (
    predict_investment,
    prepare_features,
    explain_prediction
)

app = FastAPI(title="BOI Investment Predictor")

# --- CORS (Vite dev server) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------- Input Schema --------
class ProjectInput(BaseModel):
    product_description: str = ""
    sub_product: str = ""

    sector: str
    shareholder_type: str
    district: str
    factory_zone: str

    agreement_year: int

    estimated_total_investments_usd_mn: float = 0
    estimated_local_investments_usd_mn: float = 0
    estimated_foreign_investments_usd_mn: float = 0
    est_total_manpower_local: int = 0
    est_total_manpower_foreign: int = 0
    project_duration_months: int = 0
    
    # Debug/verbose flag
    verbose_embeddings: bool = False


# -------- Output Schema (optional but clean) --------
class Explanation(BaseModel):
    feature: str
    impact: float
    semantic: Optional[str] = None


class PredictionResponse(BaseModel):
    log_prediction: float
    predicted_investment_usd_mn: float
    explanations: Optional[List[Explanation]] = None


# -------- Prediction Endpoint --------
@app.post("/predict", response_model=PredictionResponse)
def predict(data: ProjectInput):
    user_input = data.dict()

    # --- Log input data to backend console ---
    print("\n" + "="*60)
    print("📋 RECEIVED INPUT DATA:")
    for key, value in user_input.items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")

    # --- Prediction ---
    log_pred, real_pred = predict_investment(user_input)

    # --- Explanation (SHAP) ---
    X = prepare_features(user_input)
    explanation_df = explain_prediction(X, top_k=6)

    explanations = [
        {"feature": row.feature, "impact": float(row.impact), "semantic": row.semantic}
        for row in explanation_df.itertuples()
    ]

    return {
        "log_prediction": float(log_pred),
        "predicted_investment_usd_mn": float(real_pred),
        "explanations": explanations
    }
