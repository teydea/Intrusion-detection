import cv2
import json
import time
import numpy as np
from pathlib import Path

def point_in_polygon(point, polygon):
    poly = np.array(polygon, dtype=np.int32).reshape((-1, 2))
    return cv2.pointPolygonTest(poly, point, False) >= 0

def setup_restricted_zones(video_path, zones_file):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Cannot read frame")
    points = []
    def click_event(event, x, y, *args):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            if len(points) > 1:
                cv2.line(frame, tuple(points[-2]), tuple(points[-1]), (0, 255, 0), 2)
            cv2.imshow('Zone Setup', frame)
    cv2.namedWindow('Zone Setup')
    cv2.setMouseCallback('Zone Setup', click_event)
    frame_copy = frame.copy()
    while True:
        disp = frame_copy.copy()
        for i, pt in enumerate(points):
            cv2.circle(disp, tuple(pt), 5, (0, 0, 255), -1)
            if i > 0:
                cv2.line(disp, tuple(points[i-1]), tuple(pt), (0, 255, 0), 2)
        cv2.imshow('Zone Setup', disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and len(points) >= 3:
            if points[0] != points[-1]:
                points.append(points[0])
            Path(zones_file).parent.mkdir(exist_ok=True)
            with open(zones_file, 'w') as f:
                json.dump({"zones": [points]}, f, indent=2)
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            exit(1)
    cv2.destroyAllWindows()

def load_zones(zones_file):
    with open(zones_file, 'r') as f:
        data = json.load(f)
    zones = []
    for zone in data["zones"]:
        pts = np.array(zone, dtype=np.int32)
        if len(pts) > 1 and np.array_equal(pts[0], pts[-1]):
            pts = pts[:-1]
        zones.append(pts)
    return zones