# scripts/1_live_data_collection.py
import os
import time
import cv2
import pandas as pd
import mediapipe as mp
from collections import deque

# Resolve project root (script directory -> project root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Always write inside the project-local data folder
OUT_DIR = os.path.join(PROJECT_ROOT, "data")
CSV_PATH = os.path.join(OUT_DIR, "pose_data.csv")

# Camera config
CAM_INDEX = 0
USE_AVFOUNDATION = True  # macOS recommended
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_TARGET = 30

# I/O buffering
FLUSH_EVERY_N_ROWS = 50

# MediaPipe config
MODEL_COMPLEXITY = 1
SMOOTH_LANDMARKS = True
ENABLE_SEGMENTATION = False

NUM_LANDMARKS = 33

def build_columns():
    cols = ["session_id", "timestamp_ms"]
    for i in range(NUM_LANDMARKS):
        cols += [f"x_{i}", f"y_{i}", f"z_{i}", f"v_{i}"]
    return cols

def ensure_csv_header(path, columns):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Create header if file doesn't exist or is empty
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        pd.DataFrame(columns=columns).to_csv(path, index=False)

def extract_row(session_id, ts_ms, landmarks):
    row = [session_id, ts_ms]
    for i in range(NUM_LANDMARKS):
        lm = landmarks[i]
        row.extend([float(lm.x), float(lm.y), float(lm.z), float(getattr(lm, "visibility", 1.0))])
    return row

def open_camera(index: int, use_avfoundation: bool = True):
    # Prefer AVFoundation on macOS for reliability
    if use_avfoundation:
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(index)
    return cap

def main():
    cols = build_columns()
    ensure_csv_header(CSV_PATH, cols)

    cap = open_camera(CAM_INDEX, USE_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam. Check CAM_INDEX, permissions, or Camera privacy settings.")

    # Warm-up reads (some macOS setups need a few grabs before stable frames)
    for _ in range(10):
        ok, _ = cap.read()
        if ok:
            break

    # Optionally set properties AFTER capture is proven working; comment out if unstable
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    # cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)

    session_id = int(time.time())
    buffer = deque()

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    last_flush = time.time()
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=MODEL_COMPLEXITY,
        smooth_landmarks=SMOOTH_LANDMARKS,
        enable_segmentation=ENABLE_SEGMENTATION
    ) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed; stopping.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            status = "No pose"
            color = (0, 255, 255)

            if res.pose_landmarks:
                mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                ts = int(time.time() * 1000)
                row = extract_row(session_id, ts, res.pose_landmarks.landmark)
                buffer.append(row)
                status = "Recording pose"
                color = (0, 200, 0)

            cv2.putText(frame, "Press q to stop", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
            cv2.putText(frame, status, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("Collect Pose Data", frame)

            # Periodic flush
            if len(buffer) >= FLUSH_EVERY_N_ROWS or (time.time() - last_flush) > 3.0:
                if buffer:
                    batch = list(buffer)
                    buffer.clear()
                    pd.DataFrame(batch, columns=cols).to_csv(CSV_PATH, mode="a", header=False, index=False)
                    last_flush = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Final flush
    if buffer:
        pd.DataFrame(list(buffer), columns=cols).to_csv(CSV_PATH, mode="a", header=False, index=False)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
