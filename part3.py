import cv2 as cv
import numpy as np
from ultralytics import YOLO
import time
from collections import defaultdict, deque
import math
import pickle

MIN_HISTORY = 15
BEHAVIOR_WINDOW = 120
FPS = 25

ANOMALY_THRESHOLD = -0.05   
ANOMALY_CONFIRM_COUNT = 5   
SMOOTHING_FACTOR = 0.6      

history = defaultdict(lambda: deque(maxlen=120))
feature_history = defaultdict(lambda: deque(maxlen=120))
anomaly_counter = defaultdict(int)

model_if = pickle.load(open("iforest.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

def getScore(feature):
    feature_scaled = scaler.transform([feature])
    return model_if.decision_function(feature_scaled)[0]


def update_history(track_id, centroid, inside, box):
    now = time.time()
    history[track_id].append((centroid[0], centroid[1], now, inside, box))


def extract_features(track_id):
    data = history[track_id]
    if len(data) < MIN_HISTORY:
        return None

    speeds = []
    path_length = 0
    inside_time = 0

    for i in range(1, len(data)):
        dt = 1 / FPS

        dx = data[i][0] - data[i-1][0]
        dy = data[i][1] - data[i-1][1]
        dist = math.sqrt(dx*dx + dy*dy)

        speed = dist / dt
        speeds.append(speed)
        path_length += dist

        if data[i][3]:
            inside_time += dt

    if len(speeds) < 2:
        return None

    # smoothing speeds (IMPORTANT for accuracy)
    speeds = np.array(speeds)
    speeds = SMOOTHING_FACTOR * speeds + (1 - SMOOTHING_FACTOR) * np.mean(speeds)

    dx = data[-1][0] - data[0][0]
    dy = data[-1][1] - data[0][1]
    net_displacement = math.sqrt(dx*dx + dy*dy)

    total_time = len(data) * (1/FPS)
    if total_time <= 0:
        return None

    feature = [
        np.mean(speeds),
        np.std(speeds),
        path_length,
        net_displacement,
        inside_time / total_time
    ]

    return np.array(feature, dtype=np.float32)


def build_behavior_vector(track_id):
    h = feature_history[track_id]
    if len(h) < BEHAVIOR_WINDOW:
        return None

    h = np.array(h)

    behavior = [
        np.mean(h[:,0]),
        np.mean(h[:,1]),
        np.mean(h[:,2]),
        np.mean(h[:,3]),
        np.mean(h[:,4])
    ]

    return np.array(behavior, dtype=np.float32)


def Load_model():
    return YOLO("yolov8n.pt")


def detect_and_track(model, frame):
    return model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml")


def restricted_zone(frame):
    h, w = frame.shape[:2]
    zone = np.array([[0,0],[w//5,0],[w//5,h//5],[0,h//5]], dtype=np.int32)
    return zone.reshape((-1,1,2))


def Centroid(box):
    x1, y1, x2, y2 = map(int, box)
    return ((x1+x2)//2, (y1+y2)//2)


def is_inside_zone(point, zone):
    return cv.pointPolygonTest(zone, (int(point[0]), int(point[1])), False) >= 0


# ================= MAIN =================
def main():
    model = Load_model()
    cap = cv.VideoCapture("test.mp4")

    ret, frame = cap.read()
    if not ret:
        return

    zone = restricted_zone(frame)

    stTime = time.time()
    outliers = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv.polylines(frame, [zone], True, (0, 0, 255), 2)

        results = detect_and_track(model, frame)

        if results[0].boxes.id is not None:
            for box, track_id in zip(results[0].boxes.xyxy, results[0].boxes.id):
                track_id = int(track_id)
                centroid = Centroid(box)
                inside = is_inside_zone(centroid, zone)

                update_history(track_id, centroid, inside, box)

                features = extract_features(track_id)
                if features is not None:
                    feature_history[track_id].append(features)

        # ====== ANOMALY CHECK ======
        if time.time() - stTime >= 2:
            outliers.clear()

            for track_id in list(feature_history.keys()):
                behavior = build_behavior_vector(track_id)
                if behavior is None:
                    continue

                score = getScore(behavior)

                # More stable decision
                if score < ANOMALY_THRESHOLD:
                    anomaly_counter[track_id] += 1
                else:
                    anomaly_counter[track_id] = max(0, anomaly_counter[track_id] - 1)

                if anomaly_counter[track_id] >= ANOMALY_CONFIRM_COUNT:
                    outliers.append(track_id)

            stTime = time.time()

        # ====== DRAW ======
        global_anomaly = len(outliers) > 0

        if results[0].boxes.id is not None:
            for box, track_id in zip(results[0].boxes.xyxy, results[0].boxes.id):
                track_id = int(track_id)
                x1, y1, x2, y2 = map(int, box)

                if track_id in outliers:
                    cv.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 3)
                    cv.putText(frame, "ANOMALY", (x1,y2+15),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                else:
                    cv.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        # ====== HEADING ALERT ======
        if global_anomaly:
            cv.putText(frame, "ANOMALY DETECTED",
                       (50, 50),
                       cv.FONT_HERSHEY_SIMPLEX,
                       1.2,
                       (0, 0, 255),
                       3)

        cv.imshow("Frame", frame)

        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()