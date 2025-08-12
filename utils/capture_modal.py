# utils/capture_modal.py
import os
import time
from collections import deque
import pandas as pd
import cv2

from utils.io_paths import Paths
from utils.camera import open_capture
from utils.feature_vector import vectorize_landmarks_with_fallback, NUM_LANDMARKS

def _build_columns():
    cols = ["session_id", "timestamp_ms"]
    for i in range(NUM_LANDMARKS):
        cols += [f"x_{i}", f"y_{i}", f"z_{i}", f"v_{i}"]
    return cols

def _append_rows(path: str, rows, columns):
    df = pd.DataFrame(list(rows), columns=columns)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode="a", header=False, index=False)

def run_modal_capture_session(sessions_dir: str, label: str, seconds: int | None = None) -> str:
    """
    Modal OpenCV capture on the main thread. Press 'q' to stop.
    Uses pose lite model and 640x360 inference for speed.
    """
    os.makedirs(sessions_dir, exist_ok=True)
    name = Paths.timestamp_name(f"{label}_pose")
    out_csv = os.path.join(sessions_dir, name)

    cap = open_capture(index=0, use_avfoundation=True)

    mp = __import__("mediapipe").solutions
    mp_pose = mp.pose
    mp_draw = __import__("mediapipe").solutions.drawing_utils

    cols = _build_columns()
    session_id = int(time.time())
    buffer = deque()
    last_flush = time.time()
    start = time.time()
    frame_idx = 0
    draw_every = 3  # throttle landmark drawing

    try:
        with mp_pose.Pose(static_image_mode=False, model_complexity=0, smooth_landmarks=True, enable_segmentation=False) as pose:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("Frame grab failed; stopping.")
                    break

                # Downscale for inference
                small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)

                if res.pose_landmarks:
                    if frame_idx % draw_every == 0:
                        mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    ts = int(time.time() * 1000)
                    feat = vectorize_landmarks_with_fallback(res.pose_landmarks.landmark).reshape(-1)
                    row = [session_id, ts] + feat.tolist()
                    buffer.append(row)

                cv2.putText(frame, f"Recording {label} - press 'q' to stop", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
                cv2.imshow(f"Capture - {label}", frame)
                frame_idx += 1

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

                if seconds is not None and (time.time() - start) >= seconds:
                    break

                if (time.time() - last_flush) > 2.0 and buffer:
                    _append_rows(out_csv, buffer, cols)
                    buffer.clear()
                    last_flush = time.time()

        if buffer:
            _append_rows(out_csv, buffer, cols)

    finally:
        cap.release()
        try:
            cv2.destroyWindow(f"Capture - {label}")
        except Exception:
            cv2.destroyAllWindows()

    return out_csv
