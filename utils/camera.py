# utils/camera.py
import cv2

def open_capture(index: int = 0, use_avfoundation: bool = True):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION) if use_avfoundation else cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera. Ensure permissions are granted and no other app is using it.")
    for _ in range(10):
        ok, _ = cap.read()
        if ok:
            break
    return cap
