"""Data registration: validate the raw dataset stored in the repo and print a summary.

The dataset is versioned inside the GitHub repository itself (tourism_project/data/tourism.csv),
so 'registering' it means confirming it exists and has the schema the rest of the pipeline expects.
The script exits with a non-zero code if validation fails, which stops the GitHub Actions workflow.
"""
import os
import sys
import pandas as pd

DATA_PATH = os.path.join("tourism_project", "data", "tourism.csv")

# Columns the downstream scripts rely on (from the data dictionary)
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "DurationOfPitch",
    "Occupation", "Gender", "NumberOfPersonVisiting", "NumberOfFollowups", "ProductPitched",
    "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]

def main():
    # 1. The file must exist in the repo
    if not os.path.exists(DATA_PATH):
        sys.exit(f"[ERROR] Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # 2. Schema check - every expected column must be present
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing:
        sys.exit(f"[ERROR] Missing expected columns: {missing}")

    # 3. Basic sanity checks on the target
    if not set(df["ProdTaken"].unique()) <= {0, 1}:
        sys.exit("[ERROR] Target 'ProdTaken' must be binary (0/1)")

    # 4. Summary
    print("=" * 60)
    print("DATASET REGISTERED:", DATA_PATH)
    print("=" * 60)
    print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    print("All expected columns present:", len(EXPECTED_COLUMNS))
    print("Extra columns (will be dropped in prep):", extra if extra else "none")
    print("Missing values (total):", int(df.isna().sum().sum()))
    print("Duplicate CustomerIDs:", int(df["CustomerID"].duplicated().sum()))
    print("Target distribution (ProdTaken):")
    print(df["ProdTaken"].value_counts().rename({0: "Not purchased (0)", 1: "Purchased (1)"}).to_string())
    print(f"Purchase rate: {df['ProdTaken'].mean():.2%}")
    print("Validation passed.")

if __name__ == "__main__":
    main()
