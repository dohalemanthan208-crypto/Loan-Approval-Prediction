
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# --------------------------------------------------
# 1. PROJECT PATHS AND COLUMN NAMES
# --------------------------------------------------

APP_DIR = Path(__file__).parent
DATA_PATH = APP_DIR / "data" / "train.csv"
MODEL_PATH = APP_DIR / "models" / "loan_model.joblib"

FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]

TARGET = "Loan_Status"

NUMERIC_FEATURES = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]


# --------------------------------------------------
# 2. CREATE A DEMO DATASET IF NO CSV IS AVAILABLE
# --------------------------------------------------

def make_demo_data(n=900, seed=42):
    """Generate synthetic data for educational demonstration only."""

    rng = np.random.default_rng(seed)

    gender = rng.choice(
        ["Male", "Female"], n, p=[0.78, 0.22]
    )
    married = rng.choice(
        ["Yes", "No"], n, p=[0.65, 0.35]
    )
    dependents = rng.choice(
        ["0", "1", "2", "3+"],
        n,
        p=[0.55, 0.18, 0.16, 0.11],
    )
    education = rng.choice(
        ["Graduate", "Not Graduate"],
        n,
        p=[0.78, 0.22],
    )
    self_employed = rng.choice(
        ["Yes", "No"], n, p=[0.14, 0.86]
    )

    applicant_income = np.clip(
        rng.lognormal(10.7, 0.55, n),
        1500,
        250000,
    )

    coapplicant_income = np.where(
        married == "Yes",
        np.clip(
            rng.lognormal(9.7, 0.75, n),
            0,
            150000,
        ),
        0,
    )

    loan_amount = np.clip(
        (applicant_income + coapplicant_income) / 850
        + rng.normal(0, 75, n),
        20,
        700,
    )

    loan_term = rng.choice(
        [120, 180, 240, 300, 360],
        n,
        p=[0.05, 0.08, 0.12, 0.12, 0.63],
    )

    credit_history = rng.choice(
        [0, 1], n, p=[0.18, 0.82]
    )

    property_area = rng.choice(
        ["Urban", "Semiurban", "Rural"],
        n,
        p=[0.32, 0.38, 0.30],
    )

    # Synthetic target rule for demonstration.
    score = (
        -1.0
        + 2.4 * credit_history
        + 0.000012
        * (applicant_income + coapplicant_income)
        + 0.55 * (education == "Graduate")
        + 0.35 * (property_area == "Semiurban")
        - 0.35 * (self_employed == "Yes")
        - 0.0009 * loan_amount
        + rng.normal(0, 0.9, n)
    )

    approved = (
        score > np.quantile(score, 0.36)
    ).astype(int)

    df = pd.DataFrame({
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income.round(0),
        "CoapplicantIncome": coapplicant_income.round(0),
        "LoanAmount": loan_amount.round(0),
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
        "Loan_Status": np.where(
            approved == 1, "Y", "N"
        ),
    })

    # Add missing values to demonstrate imputation.
    for column in [
        "LoanAmount",
        "Credit_History",
        "Self_Employed",
    ]:
        indices = rng.choice(
            df.index,
            size=max(1, int(0.025 * n)),
            replace=False,
        )
        df.loc[indices, column] = np.nan

    return df


# --------------------------------------------------
# 3. LOAD AND CLEAN THE DATASET
# --------------------------------------------------

def load_data():

    if not DATA_PATH.exists():
        return (
            make_demo_data(),
            "Synthetic demo dataset (not real-world data)",
        )

    df = pd.read_csv(DATA_PATH)

    # Remove ID column if present.
    df = df.drop(columns=["Loan_ID"], errors="ignore")

    # Remove spaces around CSV column names.
    df.columns = df.columns.str.strip()

    missing_columns = [
        col
        for col in FEATURES + [TARGET]
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Your CSV is missing these required columns: "
            + ", ".join(missing_columns)
            + ". Check data/train.csv."
        )

    df = df[FEATURES + [TARGET]].copy()

    # Standardize target labels before converting to integers.
    status = (
        df[TARGET]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    status_mapping = {
        "y": 1,
        "n": 0,
        "yes": 1,
        "no": 0,
        "approved": 1,
        "rejected": 0,
        "1": 1,
        "0": 0,
        "1.0": 1,
        "0.0": 0,
        "true": 1,
        "false": 0,
    }

    df[TARGET] = status.map(status_mapping)

    # Drop missing or unrecognized target values.
    df = df.dropna(subset=[TARGET]).copy()

    if df.empty:
        raise ValueError(
            "No valid Loan_Status values were found. "
            "Expected Y/N, Approved/Rejected, or 1/0."
        )

    # Convert target to numeric integer labels.
    df[TARGET] = pd.to_numeric(
        df[TARGET], errors="coerce"
    )

    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)

    # Convert numeric columns safely.
    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    # Clean categorical text fields.
    for column in CATEGORICAL_FEATURES:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
        )

    # Normalize common categorical values.
    for column in [
        "Gender",
        "Married",
        "Education",
        "Self_Employed",
        "Property_Area",
    ]:
        df[column] = df[column].str.title()

    # Normalize dependent values.
    df["Dependents"] = df["Dependents"].replace({
        "3+": "3+",
        "3.0": "3",
        "0.0": "0",
        "1.0": "1",
        "2.0": "2",
    })

    # Ensure there are examples of both classes.
    if df[TARGET].nunique() < 2:
        raise ValueError(
            "Loan_Status must contain both approved "
            "and rejected examples."
        )

    return df, "Public CSV dataset (data/train.csv)"


# --------------------------------------------------
# 4. PREPROCESSING PIPELINE
# --------------------------------------------------

def build_preprocessor():

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES,
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES,
        ),
    ])

    return preprocessor


# --------------------------------------------------
# 5. TRAIN AND COMPARE CLASSIFICATION MODELS
# --------------------------------------------------

def ensure_model(model_path=None):

    if model_path is None:
        model_path = MODEL_PATH

    model_path = Path(model_path)

    df, data_source = load_data()

    X = df[FEATURES].copy()

    y = pd.to_numeric(
        df[TARGET].astype("string").str.strip().str.upper().map({
            "Y": 1,
            "N": 0,
            "YES": 1,
            "NO": 0,
            "APPROVED": 1,
            "REJECTED": 0,
            "1": 1,
            "0": 0,
            "1.0": 1,
            "0.0": 0,
        }),
        errors="coerce",
    )

    valid_rows = y.notna()
    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].astype(int)

    if y.nunique() < 2:
        raise ValueError(
            "Training requires both target classes: "
            "approved and rejected."
        )
        raise ValueError(
            "Training requires both target classes: "
            "approved and rejected."
        )

    # Split the dataset into training and testing sets.
    class_counts = y.value_counts()

    stratify_target = (
        y
        if len(class_counts) == 2
        and class_counts.min() >= 2
        else None
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify_target,
    )

    # Classification algorithms.
    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    metrics = []
    fitted_models = {}
    predictions = {}

    for name, classifier in candidates.items():

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ])

        # Fit using training data only.
        pipeline.fit(X_train, y_train)

        # Evaluate on unseen test data.
        y_pred = pipeline.predict(X_test)

        fitted_models[name] = pipeline
        predictions[name] = y_pred

        metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(
                y_test, y_pred
            ),
            "Precision": precision_score(
                y_test, y_pred, zero_division=0
            ),
            "Recall": recall_score(
                y_test, y_pred, zero_division=0
            ),
            "F1-score": f1_score(
                y_test, y_pred, zero_division=0
            ),
        })

    metrics_df = pd.DataFrame(metrics)

    # Select the model with the highest test F1-score.
    metrics_df = metrics_df.sort_values(
        "F1-score",
        ascending=False,
    ).reset_index(drop=True)

    best_model_name = str(
        metrics_df.iloc[0]["Model"]
    )

    best_model = fitted_models[best_model_name]
    best_predictions = predictions[best_model_name]

    # Save the trained model.
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump({
        "model": best_model,
        "model_name": best_model_name,
        "source": data_source,
    }, model_path)

    print("\nDataset:", data_source)
    print("Selected model:", best_model_name)
    print("\nModel comparison:")
    print(metrics_df.to_string(index=False))

    # Return values required by app.py.
    return {
        "best_model": best_model,
        "best_model_name": best_model_name,
        "metrics": metrics_df,
        "data_source": data_source,
        "feature_names": FEATURES,
        "test_y": y_test,
        "test_pred": best_predictions,
    }


# --------------------------------------------------
# 6. ALLOW THIS FILE TO BE RUN DIRECTLY
# --------------------------------------------------

if __name__ == "__main__":

    results = ensure_model()

    print("\nModel training completed successfully!")
    print("Dataset mode:", results["data_source"])
    print("Best model:", results["best_model_name"])
