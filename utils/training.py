# utils/training.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_and_save_model(labeled_csv: str, model_path: str) -> str:
    if not os.path.exists(labeled_csv) or os.path.getsize(labeled_csv) == 0:
        raise FileNotFoundError("Labeled dataset not found or empty. Capture Good/Bad sessions first.")

    df = pd.read_csv(labeled_csv)
    if "label" not in df.columns:
        raise ValueError("Missing 'label' column in labeled dataset.")
    feature_cols = [c for c in df.columns if c.startswith(("x_", "y_", "z_", "v_"))]
    if not feature_cols:
        raise ValueError("No feature columns found (x_/y_/z_/v_).")

    X = df[feature_cols].fillna(0.0).values
    y = df["label"].astype(str).values
    if len(set(y)) < 2:
        raise ValueError("Need at least two classes in labeled data (good and bad).")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, solver="lbfgs"))
    ])

    pipe.fit(Xtr, ytr)
    ypred = pipe.predict(Xte)
    acc = accuracy_score(yte, ypred)
    report = classification_report(yte, ypred)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipe, model_path)

    return f"Validation accuracy: {acc:.4f}\n\n{report}\nSaved: {model_path}"
