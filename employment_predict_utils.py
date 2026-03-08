import numpy as np
import pandas as pd
import re
import joblib
import shap
from pathlib import Path

ARTIFACT_DIR = Path("Model_Tools") / "Employment_Prediction"


def _load_artifact(filename: str):
    return joblib.load(ARTIFACT_DIR / filename)


# ===============================
# Load Employment Artifacts
# ===============================
model = _load_artifact("model_lgb.joblib")

tfidf_desc = _load_artifact("tfidf_desc.joblib")
svd_desc = _load_artifact("svd_desc.joblib")

tfidf_sub = _load_artifact("tfidf_sub.joblib")
svd_sub = _load_artifact("svd_sub.joblib")

district_te = _load_artifact("district_te.joblib")
factory_zone_te = _load_artifact("factory_zone_te.joblib")

numeric_cols = _load_artifact("numeric_cols.joblib")
feature_cols = _load_artifact("feature_cols.joblib")
GLOBAL_MEAN = _load_artifact("global_mean.joblib")

explainer = shap.TreeExplainer(model)


# ===============================
# Text Numeric Extraction
# ===============================
def extract_numeric_from_text(text: str) -> dict:
    text = "" if pd.isna(text) else text.upper()

    mw = re.findall(r'(\d+\.?\d*)\s*MW', text)
    kw = re.findall(r'(\d+\.?\d*)\s*KW', text)
    rooms = re.findall(r'(\d+)\s*ROOM', text)

    return {
        "capacity_mw": max(map(float, mw), default=0.0),
        "capacity_kw": max(map(float, kw), default=0.0),
        "hotel_rooms": max(map(int, rooms), default=0),
        "is_power_project": int(
            any(x in text for x in ["POWER", "HYDRO", "SOLAR", "BIOMASS"])
        ),
        "is_hotel_project": int("HOTEL" in text)
    }


# ===============================
# Feature Preparation
# ===============================
def prepare_features(user_input: dict) -> pd.DataFrame:
    row = {}

    # ---- Log numeric ----
    for col in numeric_cols:
        val = max(user_input.get(col, 0), 0)
        row[f"log_{col}"] = np.log1p(val)

    # ---- Text derived numerics ----
    text_nums = extract_numeric_from_text(
        user_input.get("product_description", "")
    )

    for k, v in text_nums.items():
        if k in ["capacity_mw", "capacity_kw", "hotel_rooms"]:
            row[f"log_{k}"] = np.log1p(v)
        else:
            row[k] = v

    # ---- Target Encoding ----
    row["district_te"] = district_te.get(
        user_input.get("district"), GLOBAL_MEAN
    )

    row["factory_zone_te"] = factory_zone_te.get(
        user_input.get("factory_zone"), GLOBAL_MEAN
    )

    # ---- Categorical ----
    row["sector"] = user_input.get("sector", "Unknown")
    row["shareholder_type"] = user_input.get("shareholder_type", "Unknown")

    # ---- Description embeddings ----
    desc = user_input.get("product_description", "").lower()
    desc_vec = tfidf_desc.transform([desc])
    desc_svd = svd_desc.transform(desc_vec)

    for i in range(desc_svd.shape[1]):
        row[f"desc_svd_{i+1}"] = desc_svd[0, i]

    # ---- Sub-product embeddings ----
    sub = user_input.get("sub_product", "").lower()
    sub_vec = tfidf_sub.transform([sub])
    sub_svd = svd_sub.transform(sub_vec)

    for i in range(sub_svd.shape[1]):
        row[f"sub_svd_{i+1}"] = sub_svd[0, i]

    df = pd.DataFrame([row])

    for col in ["sector", "shareholder_type"]:
        df[col] = df[col].astype("category")

    return df[feature_cols]


# ===============================
# Prediction
# ===============================
def predict_employment(user_input: dict):
    X = prepare_features(user_input)

    log_pred = float(model.predict(X)[0])
    real_pred = float(np.expm1(log_pred))

    return log_pred, real_pred


# ===============================
# SHAP Explanation
# ===============================
def explain_prediction(X: pd.DataFrame, top_k: int = 6) -> pd.DataFrame:
    shap_values = explainer.shap_values(X)

    shap_contrib = pd.DataFrame({
        "feature": X.columns,
        "impact": shap_values[0]
    })

    shap_contrib["abs_impact"] = shap_contrib["impact"].abs()
    top_features = shap_contrib.nlargest(top_k, "abs_impact")

    return top_features[["feature", "impact"]]