"""Example detector: finds strongly red objects in the frame and reports them as "box".

Use it to test the whole camera -> obstacle pipeline before your trained model is ready:
    DETECTOR_PROTO=python:color_blob_detector:ColorBlobDetector

Your own detector only needs the same shape: a class with detect(frame_bgr) returning
[{"cls": str, "conf": float, "box": [x1, y1, x2, y2]}, ...]  (pixels).
"""
import cv2
import numpy as np


class ColorBlobDetector:
    min_interval = 0.0      # seconds between two detections (raise it for slow models)
    min_area = 0.01         # ignore blobs smaller than 1% of the image

    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 120, 80), (10, 255, 255)) | cv2.inRange(hsv, (170, 120, 80), (180, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = frame.shape[:2]
        found = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            share = bw * bh / (w * h)
            if share >= self.min_area:
                found.append({"cls": "box", "conf": min(0.99, 0.6 + share), "box": [x, y, x + bw, y + bh]})
        return found
