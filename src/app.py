from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from prometheus_fastapi_instrumentator import Instrumentator
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("loan-risk-api")

app = FastAPI(
    title="Loan Risk Prediction API",
    description="Predicts whether a loan will be Approved or Rejected",
    version="1.0"
)

Instrumentator().instrument(app).expose(app)

model = joblib.load("models/model.pkl")


class LoanInput(BaseModel):
    no_of_dependents: int
    education: int
    self_employed: int
    income_annum: int
    loan_amount: int
    loan_term: int
    cibil_score: int
    residential_assets_value: int
    commercial_assets_value: int
    luxury_assets_value: int
    bank_asset_value: int


@app.get("/")
def home():
    return {"message": "Loan Risk Prediction API is running"}


@app.post("/predict")
def predict(data: LoanInput):

    input_df = pd.DataFrame([data.dict()])

    logger.info(f"Prediction request received: {data}")

    prediction = model.predict(input_df)[0]

    result = "Approved" if prediction == 0 else "Rejected"

    logger.info(f"Prediction result: {result}")

    return {
        "prediction": result
    }