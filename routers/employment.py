from fastapi import APIRouter

from constants import TOP_K_EXPLANATIONS
from employment_predict_utils import (
    explain_prediction as explain_employment_prediction,
)
from employment_predict_utils import (
    predict_employment,
    prepare_features as prepare_employment_features,
)
from routers.shared import to_explanations
from schemas import EmploymentPredictionResponse, ProjectInput

router = APIRouter(tags=["Employment"])


@router.post("/predict_employment", response_model=EmploymentPredictionResponse)
def predict_employment_endpoint(data: ProjectInput):
    user_input = data.dict()

    log_pred, real_pred = predict_employment(user_input)

    features = prepare_employment_features(user_input)
    explanation_df = explain_employment_prediction(features, top_k=TOP_K_EXPLANATIONS)
    explanations = to_explanations(explanation_df, include_semantic=False)

    return {
        "log_prediction": float(log_pred),
        "predicted_employment": float(real_pred),
        "explanations": explanations,
    }
