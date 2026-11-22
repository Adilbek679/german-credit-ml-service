from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="German Credit Risk API",
    description="Logistic Regression trained on the German Credit dataset. It predicts if a loan applicant is a good or a bad risk."
)

# the model is trained and saved by train.py
# it is loaded one time when the app starts, and then used in every /predict request
model = joblib.load("model.joblib")


# the same 20 features as in train.py, the text values are the readable values from code_maps
class CreditApplication(BaseModel):
    checking_status: Literal["< 0 DM", "0-200 DM", ">= 200 DM", "no checking account"]
    duration: int = Field(gt=0, description="Duration of the loan in months")
    credit_history: Literal["no credits / all paid back", "all paid back at this bank",
                            "existing paid back till now", "delay in the past", "critical account"]
    purpose: Literal["new car", "used car", "furniture/equipment", "radio/television", "domestic appliances",
                     "repairs", "education", "retraining", "business", "other"]
    credit_amount: float = Field(gt=0, description="Credit amount in DM")
    savings: Literal["< 100 DM", "100-500 DM", "500-1000 DM", ">= 1000 DM", "unknown / none"]
    employment: Literal["unemployed", "< 1 year", "1-4 years", "4-7 years", ">= 7 years"]
    installment_rate: int = Field(ge=1, le=4, description="Installment rate in % of income (1-4)")
    personal_status: Literal["male divorced/separated", "female divorced/separated/married",
                             "male single", "male married/widowed"]
    other_debtors: Literal["none", "co-applicant", "guarantor"]
    residence_since: int = Field(ge=1, le=4, description="Years at the present residence (1-4)")
    property: Literal["real estate", "savings agreement / life insurance", "car or other", "unknown / no property"]
    age: int = Field(ge=18, description="Age in years")
    other_installment_plans: Literal["bank", "stores", "none"]
    housing: Literal["rent", "own", "for free"]
    existing_credits: int = Field(ge=1, description="Number of existing credits at this bank")
    job: Literal["unemployed / unskilled non-resident", "unskilled resident",
                 "skilled employee", "highly qualified / management"]
    people_liable: int = Field(ge=1, description="Number of people the client has to support")
    telephone: Literal["none", "yes"]
    foreign_worker: Literal["yes", "no"]

    # the first client from the dataset, it is shown in Swagger as an example
    model_config = {"json_schema_extra": {"examples": [{
        "checking_status": "< 0 DM",
        "duration": 6,
        "credit_history": "critical account",
        "purpose": "radio/television",
        "credit_amount": 1169,
        "savings": "unknown / none",
        "employment": ">= 7 years",
        "installment_rate": 4,
        "personal_status": "male single",
        "other_debtors": "none",
        "residence_since": 4,
        "property": "real estate",
        "age": 67,
        "other_installment_plans": "none",
        "housing": "own",
        "existing_credits": 2,
        "job": "skilled employee",
        "people_liable": 1,
        "telephone": "yes",
        "foreign_worker": "yes"
    }]}}


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.post("/predict")
def predict(application: CreditApplication):
    # one row DataFrame with the same column names as in training
    data = pd.DataFrame([application.model_dump()])

    prediction = int(model.predict(data)[0])
    probability_bad = float(model.predict_proba(data)[0, 1])

    return {
        "prediction": prediction,
        "risk": "bad" if prediction == 1 else "good",
        "probability_bad": round(probability_bad, 3)
    }
