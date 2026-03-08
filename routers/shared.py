from typing import List


def to_explanations(explanation_df, include_semantic: bool) -> List[dict]:
    """Convert explanation dataframe rows to API-friendly dictionaries."""
    rows: List[dict] = []
    for row in explanation_df.itertuples():
        item = {
            "feature": row.feature,
            "impact": float(row.impact),
        }
        if include_semantic:
            item["semantic"] = row.semantic
        rows.append(item)
    return rows


def log_request_payload(user_input: dict) -> None:
    """Print request payload in a readable format for local debugging."""
    print("\n" + "=" * 60)
    print("RECEIVED INPUT DATA:")
    for key, value in user_input.items():
        print(f"  {key}: {value}")
    print("=" * 60 + "\n")
