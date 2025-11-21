import cv2
import json
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO

VIDEO_PATH = "data/test.mp4"
ZONES_FILE = "data/restricted_zones.json"
YOLO_MODEL = "yolov8n.pt"
CONF_THRESHOLD = 0.5
ALARM_TIMEOUT_SEC = 3.0

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

def main():
    if not Path(ZONES_FILE).exists():
        setup_restricted_zones(VIDEO_PATH, ZONES_FILE)
    zones = load_zones(ZONES_FILE)
    device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    model = YOLO(YOLO_MODEL).to(device)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delay = int(1000 / fps)
    alarm_active = False
    alarm_off_time = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, classes=[0], conf=CONF_THRESHOLD, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                detections.append((x1, y1, x2, y2))
        intrusion = False
        for x1, y1, x2, y2 in detections:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            for zone in zones:
                if point_in_polygon((cx, cy), zone):
                    intrusion = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "INTRUDER", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    break
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
        if intrusion:
            alarm_active = True
            alarm_off_time = None
        else:
            if alarm_active and alarm_off_time is None:
                alarm_off_time = time.time()
            if alarm_off_time and (time.time() - alarm_off_time) >= ALARM_TIMEOUT_SEC:
                alarm_active = False
                alarm_off_time = None
        for zone in zones:
            cv2.polylines(frame, [zone], isClosed=True, color=(0, 0, 255), thickness=2)
        if alarm_active:
            overlay = frame.copy()
            h, w = frame.shape[:2]
            cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            cv2.putText(frame, "ALARM!", (50, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
        cv2.imshow("Intrusion Detection", frame)
        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()