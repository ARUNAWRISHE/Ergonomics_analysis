# scripts/4_live_detection.py
import os
import time
import cv2
import numpy as np
import joblib
import mediapipe as mp
from collections import deque

MODEL_PATH = os.path.join("models", "posture_model.pkl")

CAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_TARGET = 30

MODEL_COMPLEXITY = 1
SMOOTH_LANDMARKS = True
ENABLE_SEGMENTATION = False

FONT = cv2.FONT_HERSHEY_SIMPLEX
GOOD_COLOR = (0, 200, 0)
BAD_COLOR = (0, 0, 255)
NEUTRAL_COLOR = (180, 180, 0)
TEXT_COLOR = (235, 235, 235)
BG_COLOR = (25, 25, 25)

PRED_WINDOW = 8
PROB_SMOOTH = 0.6
NUM_LANDMARKS = 33

def features_from_landmarks(landmarks):
    feat = []
    for i in range(NUM_LANDMARKS):
        lm = landmarks[i]
        feat.extend([float(lm.x), float(lm.y), float(lm.z), float(getattr(lm, "visibility", 1.0))])
    return np.asarray(feat, dtype=np.float32).reshape(1, -1)

def draw_panel(frame, lines, x=10, y=10, pad=8, line_h=24):
    w = max(cv2.getTextSize(l, FONT, 0.6, 2)[0][0] for l in lines) + 2 * pad
    h = line_h * len(lines) + 2 * pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), BG_COLOR, thickness=-1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    for i, l in enumerate(lines):
        cy = y + pad + (i + 1) * line_h - 6
        cv2.putText(frame, l, (x + pad, cy), FONT, 0.6, TEXT_COLOR, 2, cv2.LINE_AA)

def majority_vote(labels):
    if not labels:
        return None
    vals, counts = np.unique(labels, return_counts=True)
    return vals[np.argmax(counts)]

def main():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        raise FileNotFoundError("Missing model at models/posture_model.pkl. Train it with scripts/3_train_model.py")

    pipe = joblib.load(MODEL_PATH)
    has_proba = hasattr(pipe, "predict_proba")

    cap = cv2.VideoCapture(CAM_INDEX)
    if FRAME_WIDTH: cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    if FRAME_HEIGHT: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if FPS_TARGET: cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam. Adjust CAM_INDEX or permissions.")

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    label_hist = deque(maxlen=PRED_WINDOW)
    smoothed_good_prob = 0.5
    fps_clock = deque(maxlen=30)
    last_time = time.time()

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=MODEL_COMPLEXITY,
        smooth_landmarks=SMOOTH_LANDMARKS,
        enable_segmentation=ENABLE_SEGMENTATION
    ) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame read failed.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            display_label = "No pose"
            display_color = NEUTRAL_COLOR
            prob_good = None

            if res.pose_landmarks:
                mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                X = features_from_landmarks(res.pose_landmarks.landmark)

                pred_label = pipe.predict(X)[0]
                if has_proba:
                    classes = list(pipe.classes_)
                    proba = pipe.predict_proba(X)[0]
                    if "good" in classes:
                        prob_good = float(proba[classes.index("good")])

                label_hist.append(str(pred_label).lower())
                voted = majority_vote(label_hist) or str(pred_label).lower()

                if voted == "good":
                    display_label = "Good posture"
                    display_color = GOOD_COLOR
                elif voted == "bad":
                    display_label = "Bad posture"
                    display_color = BAD_COLOR
                else:
                    display_label = voted

                if prob_good is not None:
                    smoothed_good_prob = PROB_SMOOTH * smoothed_good_prob + (1 - PROB_SMOOTH) * prob_good

            now = time.time()
            fps_clock.append(now - last_time)
            last_time = now
            fps = 1.0 / (np.mean(fps_clock) if fps_clock else 1e-6)

            lines = [f"{display_label}", f"FPS: {fps:.1f}", "Press q to quit"]
            if res.pose_landmarks and prob_good is not None:
                lines.insert(1, f"Good prob (smoothed): {smoothed_good_prob:.2f}")
            draw_panel(frame, lines, x=10, y=10)

            cv2.putText(frame, display_label, (10, FRAME_HEIGHT - 20), FONT, 0.9, display_color, 2, cv2.LINE_AA)

            cv2.imshow("Live Ergonomics Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
