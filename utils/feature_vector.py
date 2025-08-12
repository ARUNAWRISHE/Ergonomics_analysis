# utils/feature_vector.py
import numpy as np

NUM_LANDMARKS = 33

def vectorize_landmarks_with_fallback(landmarks) -> np.ndarray:
    # 33*4 vector (x,y,z,visibility); zeros if missing
    feat = []
    for i in range(NUM_LANDMARKS):
        lm = landmarks[i]
        x = float(getattr(lm, "x", 0.0))
        y = float(getattr(lm, "y", 0.0))
        z = float(getattr(lm, "z", 0.0))
        v = float(getattr(lm, "visibility", 0.0))
        feat.extend([x, y, z, v])
    return np.asarray(feat, dtype=np.float32).reshape(1, -1)

def build_columns():
    cols = ["session_id", "timestamp_ms"]
    for i in range(NUM_LANDMARKS):
        cols += [f"x_{i}", f"y_{i}", f"z_{i}", f"v_{i}"]
    return cols
