"""Data preparation: load raw data from the repo, clean it, split it and save the splits locally.

Outputs (written to the working directory so the GitHub Actions job can upload them as an artifact):
    Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join("tourism_project", "data", "tourism.csv")
TARGET = "ProdTaken"

# 1. Load the dataset directly from the repository data folder
df = pd.read_csv(DATA_PATH)
print("Raw shape:", df.shape)

# 2. Remove unnecessary columns
#    - the unnamed index column left over from an earlier export
#    - CustomerID is just an identifier and carries no predictive information
unnamed = [c for c in df.columns if c.startswith("Unnamed")]
df = df.drop(columns=unnamed + ["CustomerID"])
print("Dropped columns:", unnamed + ["CustomerID"])

# 3. Clean categorical values
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})      # fix typo
df["Occupation"] = df["Occupation"].replace({"Free Lancer": "Freelancer"})  # consistent spelling
for col in df.select_dtypes("object").columns:
    df[col] = df[col].str.strip()

# 4. Remove exact duplicate customer profiles (prevents train/test leakage)
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Removed {before - len(df)} duplicate rows")

# 5. Handle missing values defensively (current file has none, but new data may)
num_cols = df.select_dtypes("number").columns.drop(TARGET)
cat_cols = df.select_dtypes("object").columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
print("Missing values after cleaning:", int(df.isna().sum().sum()))

# 6. Split into features / target and into train / test (stratified to keep the ~19% purchase rate)
X = df.drop(columns=[TARGET])
y = df[TARGET]
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 7. Save the splits locally - the workflow passes these to the training job as an artifact
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)
print(f"Train: {Xtrain.shape} | Test: {Xtest.shape}")
print(f"Purchase rate - train: {ytrain.mean():.2%}, test: {ytest.mean():.2%}")
print("Saved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")
