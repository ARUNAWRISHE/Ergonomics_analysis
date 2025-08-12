# scripts/label_by_ranges.py
import os
import pandas as pd

IN_PATH = os.path.join("data", "pose_data.csv")
OUT_PATH = os.path.join("data", "pose_data_labeled.csv")

# Fill with your actual session_id and (start_ms, end_ms) ranges
GOOD_RANGES = {}
BAD_RANGES = {}

def in_ranges(ts: int, ranges):
    for a, b in ranges:
        if a <= ts <= b:
            return True
    return False

def main():
    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(f"Input CSV not found: {IN_PATH}")

    df = pd.read_csv(IN_PATH)

    # Basic schema checks
    required = {"session_id", "timestamp_ms"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {IN_PATH}: {missing}")

    # Coerce to numeric
    df["session_id"] = pd.to_numeric(df["session_id"], errors="coerce")
    df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
    if df["session_id"].isna().any() or df["timestamp_ms"].isna().any():
        raise ValueError("Found non-numeric session_id or timestamp_ms; please fix the input CSV.")

    # MUST return a single scalar string per row
    def lab(row) -> str:
        sid = int(row["session_id"])
        ts = int(row["timestamp_ms"])
        if sid in GOOD_RANGES and in_ranges(ts, GOOD_RANGES[sid]):
            return "good"
        if sid in BAD_RANGES and in_ranges(ts, BAD_RANGES[sid]):
            return "bad"
        return ""  # unlabeled

    labels = df.apply(lab, axis=1)

    # Guard: ensure labels is a 1D Series
    if not isinstance(labels, pd.Series):
        # Convert if it somehow became a DataFrame
        labels = pd.Series(labels.squeeze() if hasattr(labels, "squeeze") else labels)

    # Extra guard: if labels is a DataFrame, squeeze to 1D
    if isinstance(labels, pd.DataFrame):
        if labels.shape[1] != 1:
            raise ValueError(f"Labels is not 1D (shape={labels.shape}). Check lab() to return a single string.")
        labels = labels.iloc[:, 0]

    df = df.copy()
    df["label"] = labels.astype(str)

    out = df[df["label"].isin(["good", "bad"])].copy()
    if out.empty:
        raise ValueError(
            "No rows labeled. Fill GOOD_RANGES/BAD_RANGES with correct session_id keys and (start_ms, end_ms) ranges."
        )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} with {len(out)} labeled rows.")

if __name__ == "__main__":
    main()
