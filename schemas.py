from typing import List, Optional

from pydantic import BaseModel


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

    # Debug flag to print embedding interpretation details
    verbose_embeddings: bool = False


class Explanation(BaseModel):
    feature: str
    impact: float
    semantic: Optional[str] = None


class PredictionResponse(BaseModel):
    log_prediction: float
    predicted_investment_usd_mn: float
    explanations: Optional[List[Explanation]] = None


class EmploymentPredictionResponse(BaseModel):
    log_prediction: float
    predicted_employment: float
    explanations: Optional[List[Explanation]] = None
