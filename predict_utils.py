import numpy as np
import pandas as pd
import re
import joblib
import shap

# ===============================
# Load artifacts
# ===============================
model = joblib.load("Model_Tools/model_lgb.joblib")

tfidf_desc = joblib.load("Model_Tools/tfidf_desc.joblib")
svd_desc = joblib.load("Model_Tools/svd_desc.joblib")

tfidf_sub = joblib.load("Model_Tools/tfidf_sub.joblib")
svd_sub = joblib.load("Model_Tools/svd_sub.joblib")

district_te = joblib.load("Model_Tools/district_te.joblib")
factory_zone_te = joblib.load("Model_Tools/factory_zone_te.joblib")

numeric_cols = joblib.load("Model_Tools/numeric_cols.joblib")
feature_cols = joblib.load("Model_Tools/feature_cols.joblib")

gdp_lookup = joblib.load("Model_Tools/gdp_lookup.joblib")

GLOBAL_MEAN = 0.9399485469451957  # log-scale target mean

# SHAP explainer (Tree-based, fast)
explainer = shap.TreeExplainer(model)


# ===============================
# SVD Interpretation Helpers
# ===============================
def get_svd_component_interpretation(svd_model, tfidf_model, component_idx, top_words=10):
    """
    Decode what an SVD component represents by showing the top TF-IDF terms
    that contribute most to that component.
    
    Args:
        svd_model: The fitted SVD model (TruncatedSVD)
        tfidf_model: The fitted TF-IDF vectorizer
        component_idx: Which SVD component to interpret (0-indexed)
        top_words: How many top terms to show
    
    Returns:
        List of (term, weight) tuples representing the component's meaning
    """
    # Get the component weights from SVD
    component = svd_model.components_[component_idx]
    
    # Get feature names from TF-IDF
    feature_names = np.array(tfidf_model.get_feature_names_out())
    
    # Get indices of top positive and negative weights
    top_indices = np.argsort(np.abs(component))[-top_words:][::-1]
    
    interpretation = []
    for idx in top_indices:
        term = feature_names[idx]
        weight = component[idx]
        interpretation.append((term, float(weight)))
    
    return interpretation


def describe_text_embedding(text, tfidf_model, svd_model, model_type="desc"):
    """
    Show what semantic components a text gets mapped to.
    
    Args:
        text: The input text (product description or sub-product)
        tfidf_model: The fitted TF-IDF vectorizer
        svd_model: The fitted SVD model
        model_type: "desc" or "sub" for display purposes
    
    Returns:
        DataFrame showing which components are active and their interpretation
    """
    try:
        # Transform text through TF-IDF then SVD
        tfidf_vec = tfidf_model.transform([text.lower()])
        svd_vec = svd_model.transform(tfidf_vec)
        
        # Get component interpretations
        results = []
        for i in range(svd_vec.shape[1]):
            component_value = float(svd_vec[0, i])
            # Show all non-negligible components (abs value > 0.01)
            if abs(component_value) > 0.01:
                interpretation = get_svd_component_interpretation(
                    svd_model, tfidf_model, i, top_words=5
                )
                top_terms = ", ".join([f"{term}({w:.2f})" for term, w in interpretation[:3]])
                results.append({
                    "component": f"{model_type.upper()}_SVD_{i+1}",
                    "value": f"{component_value:.4f}",
                    "top_terms": top_terms
                })
        
        return pd.DataFrame(results) if results else pd.DataFrame()
    except Exception as e:
        print(f"❌ Error in describe_text_embedding: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

# ===============================
# Helpers
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

    # ---- Numeric (log) ----
    for col in numeric_cols:
        val = max(user_input.get(col, 0), 0)
        row[f"log_{col}"] = np.log1p(val)

    # ---- Text-derived numerics ----
    text_nums = extract_numeric_from_text(
        user_input.get("product_description", "")
    )

    for k, v in text_nums.items():
        if k in ["capacity_mw", "capacity_kw", "hotel_rooms"]:
            row[f"log_{k}"] = np.log1p(v)
        else:
            row[k] = v  # is_power_project / is_hotel_project

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

    # ---- GDP (derived from agreement year) ----
    gdp_val = gdp_lookup.get(user_input.get("agreement_year"), 0)
    row["log_gdp_growth"] = np.log1p(max(gdp_val, 0))

    # ---- Description embeddings ----
    desc = user_input.get("product_description", "").lower()
    desc_vec = tfidf_desc.transform([desc])
    desc_svd = svd_desc.transform(desc_vec)

    for i in range(desc_svd.shape[1]):
        row[f"desc_svd_{i+1}"] = desc_svd[0, i]
    
    # Optional: Log description embedding interpretation
    verbose = user_input.get("verbose_embeddings", False)
    if desc and verbose:
        print("\n📝 PRODUCT DESCRIPTION EMBEDDING BREAKDOWN:")
        print(f"   Raw SVD values: {desc_svd[0]}")
        desc_breakdown = describe_text_embedding(desc, tfidf_desc, svd_desc, "desc")
        if not desc_breakdown.empty:
            print(desc_breakdown.to_string(index=False))
        else:
            print("   (No significant components detected)")

    # ---- Sub-product embeddings ----
    sub = user_input.get("sub_product", "").lower()
    sub_vec = tfidf_sub.transform([sub])
    sub_svd = svd_sub.transform(sub_vec)

    for i in range(sub_svd.shape[1]):
        row[f"sub_svd_{i+1}"] = sub_svd[0, i]
    
    # Optional: Log sub-product embedding interpretation
    if sub and verbose:
        print("\n🏷️  SUB-PRODUCT EMBEDDING BREAKDOWN:")
        print(f"   Raw SVD values: {sub_svd[0]}")
        sub_breakdown = describe_text_embedding(sub, tfidf_sub, svd_sub, "sub")
        if not sub_breakdown.empty:
            print(sub_breakdown.to_string(index=False))
        else:
            print("   (No significant components detected)")

    df = pd.DataFrame([row])

    # ---- Ensure categorical dtype (RF safe but consistent) ----
    for col in ["sector", "shareholder_type"]:
        df[col] = df[col].astype("category")

    return df[feature_cols]


# ===============================
# Prediction
# ===============================
def predict_investment(user_input: dict):
    X = prepare_features(user_input)

    log_pred = float(model.predict(X)[0])
    real_pred = float(np.expm1(log_pred))

    return log_pred, real_pred


# ===============================
# Explanation (SHAP)
# ===============================
def get_feature_semantic_meaning(feature_name: str) -> str:
    """
    Convert raw feature names to user-friendly descriptions with semantic meaning.
    """
    import re
    
    # User-friendly mappings for common features
    feature_mappings = {
        "log_estimated_total_investments_usd_mn": "Total Investment (USD Mn)",
        "log_estimated_local_investments_usd_mn": "Local Investment (USD Mn)",
        "log_estimated_foreign_investments_usd_mn": "Foreign Investment (USD Mn)",
        "log_est_total_manpower_local": "Local Manpower",
        "log_est_total_manpower_foreign": "Foreign Manpower",
        "log_project_duration_months": "Project Duration (months)",
        "log_capacity_mw": "Power Capacity (MW)",
        "log_capacity_kw": "Power Capacity (KW)",
        "log_hotel_rooms": "Hotel Rooms",
        "is_power_project": "Power Project Indicator",
        "is_hotel_project": "Hotel Project Indicator",
        "district_te": "District",
        "factory_zone_te": "Factory Zone",
        "log_gdp_growth": "GDP Growth",
        "sector": "Industry Sector",
        "shareholder_type": "Shareholder Type",
        "log_land_extend_acres": "Land Extend (Acres)",
    }
    
    # Check if it's in the mappings
    if feature_name in feature_mappings:
        return feature_mappings[feature_name]
    
    # Check if it's a description SVD component
    match = re.match(r'desc_svd_(\d+)', feature_name)
    if match:
        component_idx = int(match.group(1)) - 1
        try:
            interpretation = get_svd_component_interpretation(
                svd_desc, tfidf_desc, component_idx, top_words=3
            )
            top_terms = ", ".join([f"{term}" for term, _ in interpretation[:2]])
            print("top_terms:", top_terms)
            return f"Product Description: {top_terms}"
        except:
            return feature_name
    
    # Check if it's a sub-product SVD component
    match = re.match(r'sub_svd_(\d+)', feature_name)
    if match:
        component_idx = int(match.group(1)) - 1
        try:
            interpretation = get_svd_component_interpretation(
                svd_sub, tfidf_sub, component_idx, top_words=3
            )
            top_terms = ", ".join([f"{term}" for term, _ in interpretation[:2]])
            return f"Sub-Product: {top_terms}"
        except:
            return feature_name
    
    # Fallback: clean up the feature name
    return feature_name.replace("_", " ").title()


def explain_prediction(X: pd.DataFrame, top_k: int = 6) -> pd.DataFrame:
    """
    Get explanations focused on the most semantically relevant SVD features.
    Shows: top 1 desc_svd + top 1 sub_svd + top other features by impact.
    """
    shap_values = explainer.shap_values(X)

    shap_contrib = pd.DataFrame({
        "feature": X.columns,
        "impact": shap_values[0],
        "value": X.iloc[0].values
    })
    
    # Separate SVD types
    desc_svd = shap_contrib[shap_contrib["feature"].str.contains(r'desc_svd', regex=True)].copy()
    sub_svd = shap_contrib[shap_contrib["feature"].str.contains(r'sub_svd', regex=True)].copy()
    other_features = shap_contrib[
        ~shap_contrib["feature"].str.contains(r'(desc_svd|sub_svd)', regex=True)
    ].copy()
    
    result_list = []
    
    # Get top 1 desc_svd by activation strength
    if not desc_svd.empty:
        desc_svd["activation"] = pd.to_numeric(desc_svd["value"], errors='coerce').abs()
        top_desc = desc_svd.nlargest(1, "activation")
        result_list.append(top_desc)
    
    # Get top 1 sub_svd by activation strength
    if not sub_svd.empty:
        sub_svd["activation"] = pd.to_numeric(sub_svd["value"], errors='coerce').abs()
        top_sub = sub_svd.nlargest(1, "activation")
        result_list.append(top_sub)
    
    # Get top other features by SHAP impact
    if not other_features.empty:
        other_features["abs_impact"] = other_features["impact"].abs()
        top_other = other_features.nlargest(top_k - len(result_list), "abs_impact")
        result_list.append(top_other)
    
    # Combine all
    result = pd.concat(result_list, ignore_index=True) if result_list else shap_contrib
    
    # Sort by absolute impact for display
    result = result.assign(abs_impact=lambda d: d["impact"].abs()).sort_values("abs_impact", ascending=False)
    
    # Add semantic meaning
    result["semantic"] = result["feature"].apply(get_feature_semantic_meaning)

    return result[["feature", "impact", "semantic"]]
