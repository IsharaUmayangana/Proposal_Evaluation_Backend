from fastapi import APIRouter

from constants import TOP_K_EXPLANATIONS
from predict_utils import explain_prediction, predict_investment, prepare_features
from routers.shared import log_request_payload, to_explanations
from schemas import PredictionResponse, ProjectInput

router = APIRouter(tags=["Investment"])


@router.post("/predict", response_model=PredictionResponse)
def predict(data: ProjectInput):
    user_input = data.dict()
    log_request_payload(user_input)

    log_pred, real_pred = predict_investment(user_input)

    features = prepare_features(user_input)
    explanation_df = explain_prediction(features, top_k=TOP_K_EXPLANATIONS)
    explanations = to_explanations(explanation_df, include_semantic=True)

    return {
        "log_prediction": float(log_pred),
        "predicted_investment_usd_mn": float(real_pred),
        "explanations": explanations,
    }
