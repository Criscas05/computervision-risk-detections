# risk_detection/utils/visualization.py
import cv2
import numpy as np

def draw_polygon(frame, poly_np, active=False):
    color = (0, 255, 0) if not active else (0, 0, 255)
    cv2.polylines(frame, [poly_np.astype(np.int32)], True, color, 2)

def draw_line(frame, p1, p2, active=False):
    color = (0, 255, 0) if not active else (0, 0, 255)
    cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 3)

def put_text(frame, text, org=(20, 40), color=(255,255,255), scale=0.8, thick=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

def draw_hud(frame, fps=None, lines=[]):
    y = 30
    if fps is not None:
        put_text(frame, f"FPS: {fps:.1f}", (20, y), (0,255,0), 0.9, 2); y += 30
    for s in lines:
        put_text(frame, s, (20, y), (0,0,255), 0.8, 2); y += 28
