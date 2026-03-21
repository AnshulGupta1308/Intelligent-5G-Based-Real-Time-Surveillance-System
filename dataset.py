import numpy as np
from collections import defaultdict, deque
from part3 import *
from main import downscale,open_rtsp
import cv2 as cv    

dataset = []

model = Load_model()
cap = open_rtsp("rtsp://admin:admin123@192.168.128.10:554/avstream/channel=1/stream=0.sdp")

ret, frame = cap.read()
frame = downscale(frame,1400,1300)
zone = restricted_zone(frame)

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    frame = downscale(frame,1400,1300)
    results = detect_and_track(model,frame)

    if results[0].boxes.id is None:
        continue

    for box, track_id in zip(results[0].boxes.xyxy, results[0].boxes.id):

        track_id=int(track_id)

        centroid=Centroid(box)
        inside=is_inside_zone(centroid,zone)

        update_history(track_id, centroid, inside, box)

        features = extract_features(track_id)

        if features is not None:
            dataset.append(features)
    cv.imshow("Frame",frame)
    if cv.waitKey(1) & 0xFF == 27:
        break
cap.release()

dataset = np.array(dataset, dtype=np.float32)
np.save("normal_features2.npy", dataset)

print("dataset shape:", dataset.shape)