import os
import json
import tempfile
import warnings

import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# MLflow prints this hint for every model, our data has no missing values, so it is not a problem
warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer column")

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

# MLflow: runs are saved in mlflow.db, model files in the mlruns folder
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("german-credit-risk")
model_name = "german-credit-risk"

# logreg_C_1 is the model from Practice 3 and Practice 6, the others are for comparison
models = {
    "logreg_C_1": LogisticRegression(max_iter=1000),
    "logreg_C_0.1": LogisticRegression(max_iter=1000, C=0.1),
    "logreg_balanced": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
}

rows = []
pipes = {}

for name, clf in models.items():
    # clone, so every pipeline has its own preprocessor
    pipe = Pipeline([("prep", clone(preprocessor)), ("model", clf)])
    pipe.fit(X_train, y_train)
    pipes[name] = pipe

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

    with mlflow.start_run(run_name=name) as run:
        # parameters: hyperparameters of the model and the split
        mlflow.log_param("model_type", type(clf).__name__)
        mlflow.log_params(clf.get_params())
        mlflow.log_params({"split_test_size": 0.2, "split_random_state": 42, "train_rows": len(X_train)})

        # metrics on the test set (bad risk = positive class)
        mlflow.log_metrics(metrics)

        # artifacts: the same model.joblib file that the API uses, and the model in MLflow format
        with tempfile.TemporaryDirectory() as tmp:
            joblib.dump(pipe, os.path.join(tmp, "model.joblib"))
            mlflow.log_artifact(os.path.join(tmp, "model.joblib"))
        # MLflow saves the model with skops, the tree type of Random Forest has to be marked as trusted
        # (it is safe here, because we trained this model ourselves)
        model_info = mlflow.sklearn.log_model(pipe, name="model", input_example=X_train.head(3),
                                              skops_trusted_types=["sklearn.tree._tree.Tree"])

    rows.append({"run": name, **metrics, "run_id": run.info.run_id, "model_uri": model_info.model_uri})

results_df = pd.DataFrame(rows).set_index("run")
print()
print(results_df[["accuracy", "precision", "recall", "f1", "roc_auc"]].round(3))

# the best run by F1 is registered in the MLflow Model Registry
best = results_df["f1"].idxmax()
version = mlflow.register_model(results_df.loc[best, "model_uri"], model_name)

client = MlflowClient()
client.set_registered_model_alias(model_name, "champion", version.version)
client.update_model_version(model_name, version.version,
                            description=f"{best}, F1 = {results_df.loc[best, 'f1']:.3f} on the test set")

print()
print(f"Best run: {best} (F1 = {results_df.loc[best, 'f1']:.3f})")
print(f"Registered as '{model_name}' version {version.version}, alias 'champion'")

# the champion is also saved for the API, model_info.json tells the API which version it is
joblib.dump(pipes[best], "model.joblib")
with open("model_info.json", "w") as f:
    json.dump({"model_name": model_name, "version": int(version.version), "run": best,
               "f1": round(results_df.loc[best, "f1"], 3)}, f, indent=2)

print("Model saved to model.joblib, info saved to model_info.json")
