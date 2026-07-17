"""
train_model.py
---------------
STEP 2 of the project: train the risk-prediction model.

What this script does, in order:
  1. Loads train.csv / validation.csv / test.csv
  2. Engineers features (via preprocessing.py)
  3. Trains two candidate models: Random Forest and XGBoost
  4. Picks whichever one does better on the validation set
  5. Reports final, honest performance on the test set (only looked at once)
  6. Saves the winning model + preprocessor + label encoder to disk,
     so the FastAPI backend (Step 5) can load and use them later
  7. Saves a SHAP summary plot + a feature-importance table — useful
     directly in the Round 1 slide deck as "Technical Feasibility" proof

Run this from the backend/model/ folder:
    python train_model.py
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")  # so this works on a server with no display
import matplotlib.pyplot as plt

from preprocessing import prepare_features, build_preprocessor, TARGET_COL
from evaluate_report import generate_full_report

# ---- Paths --------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_DIR = Path(__file__).parent / "saved_model"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{name}.csv")


def main():
    # ---- 1. Load data ----------------------------------------------------
    train_df = load_split("train")
    val_df = load_split("validation")
    test_df = load_split("test")
    print(f"Loaded: train={len(train_df)}, val={len(val_df)}, test={len(test_df)} rows")

    # ---- 2. Feature engineering ------------------------------------------
    X_train, y_train_raw, numeric_cols = prepare_features(train_df)
    X_val, y_val_raw, _ = prepare_features(val_df)
    X_test, y_test_raw, _ = prepare_features(test_df)

    # Risk label (Low/Medium/High) -> numbers (0/1/2), so the model can use it
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_val = label_encoder.transform(y_val_raw)
    y_test = label_encoder.transform(y_test_raw)
    print("Classes:", list(label_encoder.classes_))

    preprocessor = build_preprocessor(numeric_cols)

    # Because "Medium" risk is the hardest class (fewest & fuzziest examples),
    # we give the model higher sample weights for under-represented classes
    # so it doesn't just ignore them to maximise plain accuracy.
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    # ---- 3. Candidate model A: Random Forest ------------------------------
    rf_pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=400, max_depth=8, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )),
    ])
    rf_pipeline.fit(X_train, y_train)
    rf_val_pred = rf_pipeline.predict(X_val)
    rf_val_f1 = f1_score(y_val, rf_val_pred, average="macro")
    print(f"\n[Random Forest]   validation macro-F1 = {rf_val_f1:.3f}")

    # ---- 3b. Candidate model B: XGBoost ------------------------------------
    xgb_pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", xgb.XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=42, base_score=0.5,
        )),
    ])
    xgb_pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)
    xgb_val_pred = xgb_pipeline.predict(X_val)
    xgb_val_f1 = f1_score(y_val, xgb_val_pred, average="macro")
    print(f"[XGBoost]         validation macro-F1 = {xgb_val_f1:.3f}")

    # ---- 4. Pick the winner -------------------------------------------------
    if xgb_val_f1 >= rf_val_f1:
        best_name, best_pipeline = "xgboost", xgb_pipeline
    else:
        best_name, best_pipeline = "random_forest", rf_pipeline
    print(f"\n>>> Winner: {best_name}")

    # ---- 5. Final, one-time check on the test set ----------------------------
    test_pred = best_pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average="macro")
    print(f"\nTest accuracy: {test_acc:.3f}")
    print(f"Test macro-F1: {test_f1:.3f}")
    print("\nFull report:\n", classification_report(
        y_test, test_pred, target_names=label_encoder.classes_
    ))

    cm = confusion_matrix(y_test, test_pred)
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_))

    # ---- 6. Save everything the backend will need later -----------------------
    joblib.dump(best_pipeline, OUTPUT_DIR / "model_pipeline.pkl")
    joblib.dump(label_encoder, OUTPUT_DIR / "label_encoder.pkl")
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump({
            "model": best_name,
            "test_accuracy": round(test_acc, 4),
            "test_macro_f1": round(test_f1, 4),
        }, f, indent=2)
    print(f"\nSaved model_pipeline.pkl, label_encoder.pkl, metrics.json to {OUTPUT_DIR}/")

    # ---- 7. SHAP values (needed for several of the charts below) ---------------
    # SHAP needs the data AFTER preprocessing, so we transform it manually here.
    X_test_transformed = best_pipeline.named_steps["preprocess"].transform(X_test)
    feature_names = best_pipeline.named_steps["preprocess"].get_feature_names_out()

    explainer = shap.TreeExplainer(best_pipeline.named_steps["classifier"])
    shap_values = explainer.shap_values(X_test_transformed)

    # Probabilities (not just the final label) — needed for the ROC curves
    test_proba = best_pipeline.predict_proba(X_test)

    # ---- 8. Generate EVERY relevant score + chart, and one combined HTML report --
    generate_full_report(
        output_dir=OUTPUT_DIR,
        classes=list(label_encoder.classes_),
        y_train_raw=y_train_raw.values, y_val_raw=y_val_raw.values, y_test_raw=y_test_raw.values,
        rf_val_f1=rf_val_f1, xgb_val_f1=xgb_val_f1, best_name=best_name,
        y_test=y_test, test_pred=test_pred, test_proba=test_proba,
        trained_classifier=best_pipeline.named_steps["classifier"], feature_names=feature_names,
        shap_values=shap_values, X_test_transformed=X_test_transformed,
        test_accuracy=test_acc, test_macro_f1=test_f1,
    )


if __name__ == "__main__":
    main()