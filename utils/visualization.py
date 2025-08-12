# utils/visualization.py
import cv2

def draw_panel(frame, lines, x=10, y=10, pad=8, line_h=24, throttle=False):
    if throttle:
        return
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    w = max(cv2.getTextSize(l, FONT, 0.6, 2)[0][0] for l in lines) + 2 * pad
    h = line_h * len(lines) + 2 * pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 20), thickness=-1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    for i, l in enumerate(lines):
        cy = y + pad + (i + 1) * line_h - 6
        cv2.putText(frame, l, (x + pad, cy), FONT, 0.6, (235, 235, 235), 2, cv2.LINE_AA)
