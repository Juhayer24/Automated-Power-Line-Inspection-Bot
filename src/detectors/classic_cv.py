# src/detectors/classic_cv.py
import cv2
import numpy as np

def detect_edges_contours(frame, min_area=800, canny1=50, canny2=150):
    """
    Input: BGR frame
    Output: list of dicts: {'bbox': (x1,y1,x2,y2), 'score': None, 'method':'classic'}
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, canny1, canny2)
    kernel = np.ones((5,5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x,y,w,h = cv2.boundingRect(cnt)
        # basic filter: ignore extremely tall/narrow or tiny shapes
        if w < 10 or h < 10:
            continue
        boxes.append({'bbox': (x, y, x+w, y+h), 'score': None, 'method': 'classic'})
    return boxes
