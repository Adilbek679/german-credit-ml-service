import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# the file has no header, column names are from the UCI description (german.doc)
columns = ["checking_status", "duration", "credit_history", "purpose", "credit_amount",
           "savings", "employment", "installment_rate", "personal_status", "other_debtors",
           "residence_since", "property", "age", "other_installment_plans", "housing",
           "existing_credits", "job", "people_liable", "telephone", "foreign_worker", "target"]

# codes like A11 are changed to readable values, so the API can accept normal text
# A47 (vacation) and A95 (female single) are in the description, but not in the data
code_maps = {
    "checking_status": {"A11": "< 0 DM", "A12": "0-200 DM", "A13": ">= 200 DM", "A14": "no checking account"},
    "credit_history": {"A30": "no credits / all paid back", "A31": "all paid back at this bank",
                       "A32": "existing paid back till now", "A33": "delay in the past", "A34": "critical account"},
    "purpose": {"A40": "new car", "A41": "used car", "A42": "furniture/equipment", "A43": "radio/television",
                "A44": "domestic appliances", "A45": "repairs", "A46": "education", "A48": "retraining",
                "A49": "business", "A410": "other"},
    "savings": {"A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM", "A64": ">= 1000 DM", "A65": "unknown / none"},
    "employment": {"A71": "unemployed", "A72": "< 1 year", "A73": "1-4 years", "A74": "4-7 years", "A75": ">= 7 years"},
    "personal_status": {"A91": "male divorced/separated", "A92": "female divorced/separated/married",
                        "A93": "male single", "A94": "male married/widowed"},
    "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "property": {"A121": "real estate", "A122": "savings agreement / life insurance", "A123": "car or other",
                 "A124": "unknown / no property"},
    "other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
    "job": {"A171": "unemployed / unskilled non-resident", "A172": "unskilled resident",
            "A173": "skilled employee", "A174": "highly qualified / management"},
    "telephone": {"A191": "none", "A192": "yes"},
    "foreign_worker": {"A201": "yes", "A202": "no"}
}

df = pd.read_csv("german.data", sep=" ", header=None, names=columns)
df["target"] = df["target"].map({1: 0, 2: 1})   # 0 = good risk, 1 = bad risk (like in Practice 3)

for col, mapping in code_maps.items():
    df[col] = df[col].map(mapping)

print("Data:", df.shape)
print("Missing values after mapping:", df.isnull().sum().sum())

X = df.drop(columns="target")
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols)
])

# the whole pipeline is saved, so the API gets raw values and does the same preprocessing
model = Pipeline([("prep", preprocessor), ("model", LogisticRegression(max_iter=1000))])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print()
print(pd.DataFrame(cm, index=["actual good", "actual bad"], columns=["pred good", "pred bad"]))
print()
print("Accuracy: ", round(accuracy_score(y_test, y_pred), 3))
print("Precision:", round(precision_score(y_test, y_pred), 3))
print("Recall:   ", round(recall_score(y_test, y_pred), 3))
print("F1:       ", round(f1_score(y_test, y_pred), 3))

joblib.dump(model, "model.joblib")
print()
print("Model saved to model.joblib")
