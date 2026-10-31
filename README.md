# Visit with Us – Wellness Tourism Package Purchase Prediction (MLOps)

End-to-end MLOps pipeline that predicts whether a customer will buy the Wellness Tourism Package (`ProdTaken`).

## Structure
```
.github/workflows/pipeline.yml        # CI/CD: register -> prep -> train -> commit model
tourism_project/
  data/tourism.csv                    # versioned dataset
  model_building/data_register.py     # schema validation + summary
  model_building/prep.py              # cleaning + stratified train/test split
  model_building/train.py             # XGBoost + GridSearchCV, MLflow tracking, saves model
  deployment/app.py                   # Streamlit app (loads committed model)
  deployment/requirements.txt         # app dependencies (Streamlit Cloud)
  requirements.txt                    # pipeline dependencies (GitHub Actions)
```

## Results (test set, threshold 0.45)
Tuned XGBoost — Recall 0.819, Precision 0.858, F1 0.838, Accuracy 0.939, ROC-AUC 0.960.

## Run locally
```bash
pip install -r tourism_project/requirements.txt
python tourism_project/model_building/data_register.py
python tourism_project/model_building/prep.py
MLFLOW_TRACKING_URI=file:./mlruns python tourism_project/model_building/train.py
pip install -r tourism_project/deployment/requirements.txt
streamlit run tourism_project/deployment/app.py
```
