from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "german_credit_data.csv"
MODEL_PATH = PROJECT_DIR / "credit_risk_pipeline.joblib"
NUMERIC_FEATURES = ["Age", "Job", "Credit amount", "Duration"]
CATEGORICAL_FEATURES = [
    "Sex",
    "Housing",
    "Saving accounts",
    "Checking account",
    "Purpose",
]
FEATURE_COLUMNS = [
    "Age",
    "Sex",
    "Job",
    "Housing",
    "Saving accounts",
    "Checking account",
    "Credit amount",
    "Duration",
    "Purpose",
]
TARGET_COLUMN = "Risk"


def create_model() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="Unknown"),
            ),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    classifier = ExtraTreesClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", classifier),
        ]
    )


def main() -> None:
    dataset = pd.read_csv(DATASET_PATH)
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = sorted(required_columns - set(dataset.columns))
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(missing_columns)}."
        )

    features = dataset.loc[:, FEATURE_COLUMNS].copy()
    raw_target = dataset[TARGET_COLUMN]
    if raw_target.isna().any():
        raise ValueError("The Risk column must not contain missing values.")
    target = raw_target.astype(str).str.strip().str.lower()
    if target.nunique() < 2:
        raise ValueError("The Risk column must contain at least two non-missing classes.")
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].map(
            lambda value: str(value).strip() if pd.notna(value) else float("nan")
        )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    evaluation_model = create_model()
    evaluation_model.fit(x_train, y_train)
    predictions = evaluation_model.predict(x_test)

    print(f"Rows used: {len(dataset)}")
    print(f"Training rows: {len(x_train)} | Test rows: {len(x_test)}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, predictions):.3f}")
    print("Held-out classification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    final_model = create_model()
    final_model.fit(features, target)
    joblib.dump(final_model, MODEL_PATH)
    print(f"Saved full-data model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
