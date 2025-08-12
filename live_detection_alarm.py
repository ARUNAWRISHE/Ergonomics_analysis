# live_detection_alarm.py
import os
import time
from collections import deque
import numpy as np
import cv2
import joblib

from utils.io_paths import Paths
from utils.camera import open_capture
from utils.feature_vector import vectorize_landmarks_with_fallback
from utils.logging_xlsx import append_bad_event
from utils.sound import beep
from utils.visualization import draw_panel

GOOD_LABEL = "good"
BAD_LABEL = "bad"
PRED_WINDOW = 8
SMOOTH_ALPHA = 0.6  # smoothed good probability

def main():
    paths = Paths()
    if not os.path.exists(paths.model_path):
        raise FileNotFoundError("Trained model not found. Run training from admin panel first.")

    pipe = joblib.load(paths.model_path)
    has_proba = hasattr(pipe, "predict_proba")

    os.makedirs(paths.logs_dir, exist_ok=True)
    cap = open_capture(index=0, use_avfoundation=True)

    mp = __import__("mediapipe").solutions
    mp_pose = mp.pose
    mp_draw = __import__("mediapipe").solutions.drawing_utils

    label_hist = deque(maxlen=PRED_WINDOW)
    smoothed_good = 0.5
    fps_clock = deque(maxlen=30)
    last_ts = time.time()

    with mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True, enable_segmentation=False) as pose:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Frame read failed.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            display_label = "No pose"
            color = (180, 180, 0)
            prob_good = None

            if res.pose_landmarks:
                mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                X = vectorize_landmarks_with_fallback(res.pose_landmarks.landmark)
                pred = pipe.predict(X)[0]
                if has_proba:
                    classes = list(pipe.classes_)
                    proba = pipe.predict_proba(X)[0]
                    if GOOD_LABEL in classes:
                        prob_good = float(proba[classes.index(GOOD_LABEL)])
                        smoothed_good = SMOOTH_ALPHA * smoothed_good + (1 - SMOOTH_ALPHA) * prob_good

                label_hist.append(str(pred).lower())
                vals, counts = np.unique(label_hist, return_counts=True)
                voted = vals[np.argmax(counts)]

                if voted == GOOD_LABEL:
                    display_label = "Good posture"
                    color = (0, 200, 0)
                elif voted == BAD_LABEL:
                    display_label = "Bad posture"
                    color = (0, 0, 255)
                    # Trigger alarm and log event
                    beep()  # or beep("assets/beep.wav") if you add a wav file
                    append_bad_event(paths.bad_posture_xlsx, {
                        "timestamp": int(time.time() * 1000),
                        "label": voted,
                        "prob_good": smoothed_good if prob_good is not None else None
                    })
                else:
                    display_label = voted

            # FPS
            now = time.time()
            fps_clock.append(now - last_ts)
            last_ts = now
            fps = 1.0 / (np.mean(fps_clock) if fps_clock else 1e-6)

            lines = [display_label, f"FPS: {fps:.1f}", "Press q to quit"]
            if prob_good is not None:
                lines.insert(1, f"Good prob (smoothed): {smoothed_good:.2f}")
            draw_panel(frame, lines, x=10, y=10)

            cv2.putText(frame, display_label, (10, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            cv2.imshow("Live Detection with Alarm", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
