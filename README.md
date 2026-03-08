# Proposal Evaluation Backend

FastAPI backend for BOI proposal evaluation with two prediction services:
- Investment prediction
- Employment prediction

## Project Structure

```text
Proposal_Evaluation_Backend/
|-- api.py
|-- constants.py
|-- schemas.py
|-- predict_utils.py
|-- employment_predict_utils.py
|-- routers/
|   |-- __init__.py
|   |-- investment.py
|   |-- employment.py
|   `-- shared.py
|-- requirements.txt
|-- Model_Tools/
|   |-- Investment_Prediction/
|   `-- Employment_Prediction/
`-- README.md
```

## Main Files

- `api.py`: FastAPI app bootstrap (middleware + router registration).
- `constants.py`: App-level constants (title, CORS origins, explanation size).
- `schemas.py`: Pydantic request/response models.
- `routers/investment.py`: Investment prediction endpoint.
- `routers/employment.py`: Employment prediction endpoint.
- `routers/shared.py`: Shared endpoint helpers.
- `predict_utils.py`: Investment feature preparation, prediction, and SHAP explanation helpers.
- `employment_predict_utils.py`: Employment feature preparation, prediction, and SHAP explanation helpers.

## API Endpoints

- `POST /predict`: Predicts investment value and returns top feature explanations.
- `POST /predict_employment`: Predicts employment value and returns top feature explanations.

## Run Locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start server:

```bash
uvicorn api:app --reload
```

3. Open docs:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Notes

- All model artifacts are loaded from `Model_Tools/` at import time.
- Keep artifact file names unchanged unless corresponding loader code is updated.
