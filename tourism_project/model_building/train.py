"""Model training with experiment tracking.

- loads the train/test splits produced by the data-prep job (downloaded as a workflow artifact)
- defines an XGBoost pipeline and a hyperparameter grid, tunes it with GridSearchCV
- logs every tuned parameter combination and the final metrics to MLflow
- evaluates the best model and saves it to tourism_project/deployment/ so the workflow can commit it
"""
import os
import json
import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
import numpy as np
from sklearn.metrics import (classification_report, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)

# MLflow tracking: the GitHub Actions job starts a server on localhost:5000.
# When run elsewhere, MLFLOW_TRACKING_URI can point to any store (e.g. file:./mlruns).
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("tourism-wellness-package")

MODEL_DIR = os.path.join("tourism_project", "deployment")
MODEL_PATH = os.path.join(MODEL_DIR, "best_tourism_model_v1.joblib")

# 1. Load the splits from the workflow artifact
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

# 2. Preprocessing: scale numeric features, one-hot encode categorical ones
numeric_features = Xtrain.select_dtypes("number").columns.tolist()
categorical_features = Xtrain.select_dtypes("object").columns.tolist()
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

# 3. Model: XGBoost, with scale_pos_weight to handle the ~19% / 81% class imbalance
scale_pos_weight = (ytrain == 0).sum() / (ytrain == 1).sum()
xgb_model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="logloss",
                              random_state=42, n_jobs=-1)
model_pipeline = make_pipeline(preprocessor, xgb_model)

# 4. Hyperparameter grid
param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5, 7],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__colsample_bytree": [0.6, 0.8],
    "xgbclassifier__reg_lambda": [1, 5],
}

with mlflow.start_run(run_name="xgb-gridsearch"):
    # 5. Tune with 5-fold stratified CV, optimising F1 of the purchase class
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(model_pipeline, param_grid, cv=cv, scoring="f1", n_jobs=-1)
    grid.fit(Xtrain, ytrain)

    # 6. Log every tuned parameter combination as a nested run
    res = grid.cv_results_
    for i in range(len(res["params"])):
        with mlflow.start_run(run_name=f"combo_{i}", nested=True):
            mlflow.log_params(res["params"][i])
            mlflow.log_metric("mean_cv_f1", res["mean_test_score"][i])
            mlflow.log_metric("std_cv_f1", res["std_test_score"][i])

    # 7. Pick the decision threshold from out-of-fold predictions on the TRAINING set only:
    #    every training row is scored by a model that did not see it, and I keep the cut-off with the best F1.
    oof_proba = cross_val_predict(grid.best_estimator_, Xtrain, ytrain, cv=cv, method="predict_proba")[:, 1]
    candidates = np.round(np.arange(0.10, 0.91, 0.01), 2)
    oof_f1 = [f1_score(ytrain, (oof_proba >= t).astype(int)) for t in candidates]
    CLASSIFICATION_THRESHOLD = float(candidates[int(np.argmax(oof_f1))])
    print(f"Tuned threshold (max out-of-fold F1 on train): {CLASSIFICATION_THRESHOLD:.2f} | OOF F1 = {max(oof_f1):.4f}")
    mlflow.log_metric("oof_f1_at_threshold", max(oof_f1))

    # Log the best parameters
    mlflow.log_params(grid.best_params_)
    mlflow.log_param("scale_pos_weight", round(float(scale_pos_weight), 3))
    mlflow.log_param("classification_threshold", CLASSIFICATION_THRESHOLD)
    print("Combinations tried:", len(res["params"]))
    print("Best params:", grid.best_params_)
    print(f"Best CV F1: {grid.best_score_:.4f}")

    # 8. Evaluate the best model on train and test using the chosen threshold
    best_model = grid.best_estimator_
    metrics = {}
    for split, X_, y_ in [("train", Xtrain, ytrain), ("test", Xtest, ytest)]:
        proba = best_model.predict_proba(X_)[:, 1]
        pred = (proba >= CLASSIFICATION_THRESHOLD).astype(int)
        metrics.update({
            f"{split}_accuracy": accuracy_score(y_, pred),
            f"{split}_precision": precision_score(y_, pred),
            f"{split}_recall": recall_score(y_, pred),
            f"{split}_f1": f1_score(y_, pred),
            f"{split}_roc_auc": roc_auc_score(y_, proba),
        })
        print(f"\n--- {split.upper()} classification report (threshold={CLASSIFICATION_THRESHOLD}) ---")
        print(classification_report(y_, pred, digits=3))
    mlflow.log_metrics(metrics)

    # 9. Save the best model so the pipeline can commit it to the repository
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    with open(os.path.join(MODEL_DIR, "model_metrics.json"), "w") as f:
        json.dump({"best_params": grid.best_params_, "threshold": CLASSIFICATION_THRESHOLD,
                   **{k: round(v, 4) for k, v in metrics.items()}}, f, indent=2)
    mlflow.log_artifact(MODEL_PATH, artifact_path="model")
    print("\nSaved best model to", MODEL_PATH)
