"""Train the startup-success model used by the Flask app.

This project stores the trained artifacts in the model/ folder using the exact
names expected by app.py:
    startup_success_model.pkl
    imputer.pkl
    feature_columns.pkl
"""

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

DATA_PATH = "data/prepared_startup_data.csv"
MODEL_DIR = "model"
FEATURES = [
    "funding_total_usd",
    "funding_rounds",
    "milestones",
    "relationships",
    "avg_participants",
]
TARGET_COLUMN = "success"


def main():
    if not os.path.exists(DATA_PATH):
        raise SystemExit(
            f"Could not find {DATA_PATH}. Run prepare_data.py first to generate "
            "the prepared CSV."
        )

    df = pd.read_csv(DATA_PATH)
    missing = [column for column in FEATURES + [TARGET_COLUMN] if column not in df.columns]
    if missing:
        raise SystemExit(
            "The CSV is missing expected columns: "
            f"{missing}. Check the dataset schema."
        )

    df = df[FEATURES + [TARGET_COLUMN]].copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    X = df[FEATURES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train_imputed, y_train)

    accuracy = model.score(X_test_imputed, y_test)
    print(f"Validation accuracy: {accuracy:.3f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "startup_success_model.pkl"))
    joblib.dump(imputer, os.path.join(MODEL_DIR, "imputer.pkl"))
    joblib.dump(FEATURES, os.path.join(MODEL_DIR, "feature_columns.pkl"))

    print(f"Saved model artifacts to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
