import cv2 as cv
import numpy as np
from ultralytics import YOLO
import os
import math
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

VIDEO_FOLDER = "videos"
MODEL_PATH = "yolov8m.pt"
OUTPUT_DATA = "dataset.npy"
SCALER_PATH = "scaler_new.pkl"
MODEL_SAVE_PATH = "isolation_forest.pkl"

PERSON_CLASS = 0
OBJECT_CLASSES = [24, 26, 28, 63, 67, 73, 74]

model = YOLO(MODEL_PATH)

def centroid(box):
    x1, y1, x2, y2 = map(int, box)
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def process_video(video_path):
    cap = cv.VideoCapture(video_path)
    object_tracks = {}
    velocities = {}
    stationary_frames = {}
    features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, tracker="bytetrack.yaml")

        if results[0].boxes.id is None:
            continue

        boxes = results[0].boxes.xyxy
        ids = list(map(int, results[0].boxes.id))
        classes = list(map(int, results[0].boxes.cls))

        current_objects = {}

        for box, tid, cls in zip(boxes, ids, classes):
            if cls in OBJECT_CLASSES:
                c = centroid(box)
                current_objects[tid] = c

        for oid, current_pos in current_objects.items():
            prev_pos = object_tracks.get(oid)

            if prev_pos is not None:
                vel = distance(prev_pos, current_pos)
            else:
                vel = 0

            velocities[oid] = vel

            if vel < 2:
                stationary_frames[oid] = stationary_frames.get(oid, 0) + 1
            else:
                stationary_frames[oid] = 0

            object_tracks[oid] = current_pos

            feature_vector = [
                vel,
                stationary_frames.get(oid, 0),
                current_pos[0],
                current_pos[1]
            ]

            features.append(feature_vector)

    cap.release()
    return features

def build_dataset():
    all_features = []

    for file in os.listdir(VIDEO_FOLDER):
        if file.endswith(".mp4") or file.endswith(".avi"):
            path = os.path.join(VIDEO_FOLDER, file)
            feats = process_video(path)
            all_features.extend(feats)

    data = np.array(all_features)
    np.save(OUTPUT_DATA, data)
    return data

def train_model(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42
    )

    model.fit(data_scaled)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(model, MODEL_SAVE_PATH)

if __name__ == "__main__":
    dataset = build_dataset()
    train_model(dataset)