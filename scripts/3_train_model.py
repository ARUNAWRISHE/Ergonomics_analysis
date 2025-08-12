# scripts/3_train_model.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data/pose_data_labeled.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "posture_model.pkl")

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(DATA_PATH) or os.path.getsize(DATA_PATH) == 0:
        raise FileNotFoundError("data/pose_data_labeled.csv is missing or empty. Label your data first.")

    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns if c.startswith(("x_","y_","z_","v_"))]
    if "label" not in df.columns:
        raise ValueError("Missing 'label' column in data/pose_data_labeled.csv")

    X = df[feature_cols].fillna(0.0).values
    y = df["label"].astype(str).values

    if len(set(y)) < 2:
        raise ValueError("Need at least two classes ('good' and 'bad') to train.")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, solver="lbfgs"))
    ])

    pipe.fit(Xtr, ytr)
    ypred = pipe.predict(Xte)
    print(f"Validation accuracy: {accuracy_score(yte, ypred):.4f}")
    print(classification_report(yte, ypred))

    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved {MODEL_PATH}")

if __name__ == "__main__":
    main()
