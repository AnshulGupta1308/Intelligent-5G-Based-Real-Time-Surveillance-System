import cv2 as cv
import numpy as np
from ultralytics import YOLO
import math
from collections import deque
import time
import os

model = YOLO("yolov8n.pt")

PERSON_CLASS = 0
HISTORY = 10
FLOW_HISTORY = 100
SMOOTHING_FRAMES = 3

tracks = {}
velocities = {}
flow_history = deque(maxlen=FLOW_HISTORY)

panic_count = 0
counter_count = 0
congestion_count = 0

os.makedirs("screenshots", exist_ok=True)


def centroid(box):
    x1,y1,x2,y2 = map(int, box)
    return ((x1+x2)//2, (y1+y2)//2)

def compute_velocity(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    speed = math.hypot(dx, dy)
    angle = math.atan2(dy, dx)
    return speed, angle

def main():
    global panic_count, counter_count, congestion_count
    panic_ss=0
    congestion_ss=0
    flow_ss=0
    cap = cv.VideoCapture("test2.mp4")
    frame_count = 0
    while cap.isOpened():
        frame_count += 1
        if(frame_count % 3 != 0):   
                continue
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.35, iou=0.5)

        if results[0].boxes.id is None:
            cv.imshow("Crowd Flow", frame)
            if cv.waitKey(1) & 0xFF == 27:
                break
            continue

        boxes = results[0].boxes.xyxy
        ids = list(map(int, results[0].boxes.id))
        classes = results[0].boxes.cls

        current_flow = []

        for box, tid, cls in zip(boxes, ids, classes):
            if int(cls) != PERSON_CLASS:
                continue

            c = centroid(box)

            if tid not in tracks:
                tracks[tid] = deque(maxlen=HISTORY)

            tracks[tid].append(c)

            if len(tracks[tid]) >= 2:
                p1 = tracks[tid][-2]
                p2 = tracks[tid][-1]
                speed, angle = compute_velocity(p1, p2)
                velocities[tid] = (speed, angle)
                current_flow.append((speed, angle))

            x1,y1,x2,y2 = map(int, box)
            cv.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        if len(current_flow) > 5:
            speeds = np.array([v[0] for v in current_flow])
            angles = np.array([v[1] for v in current_flow])

            avg_speed = np.mean(speeds)
            avg_angle = np.mean(angles)
            speed_var = np.var(speeds)

            flow_history.append((avg_speed, avg_angle, speed_var))

            if len(flow_history) > 10:
                hist_speeds = np.array([f[0] for f in flow_history])
                hist_angles = np.array([f[1] for f in flow_history])
                hist_vars   = np.array([f[2] for f in flow_history])

                base_speed = np.mean(hist_speeds)
                base_std   = np.std(hist_speeds)
                base_angle = np.mean(hist_angles)

                var_mean = np.mean(hist_vars)
                var_std  = np.std(hist_vars)

                cos_sim = math.cos(avg_angle - base_angle)

                panic_flag = speed_var > var_mean +  2*var_std
                counter_flag = cos_sim < 0.25 and abs(avg_speed - base_speed) < base_std
                print(f"Speed: {avg_speed:.2f}, Var: {speed_var:.2f}, CosSim: {cos_sim:.2f}")
                congestion_flag = avg_speed < base_speed - 0.2 * base_std and not panic_flag

                panic_count = panic_count + 1 if panic_flag else 0
                counter_count = counter_count + 1 if counter_flag else 0
                congestion_count = congestion_count + 1 if congestion_flag else 0

                if panic_count >= SMOOTHING_FRAMES:
                    cv.putText(frame,"PANIC DISPERSION",(50,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                    if panic_ss==0:
                        panic_ss=1
                        current_time = time.time()
                        filename = f"screenshots/panic_{tid}_{int(current_time)}.jpg"
                        cv.imwrite(filename, frame)
    
                elif counter_count >= SMOOTHING_FRAMES:
                    cv.putText(frame,"COUNTER FLOW",(50,100),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                    if flow_ss==0:
                        flow_ss=1
                        current_time = time.time()
                        filename = f"screenshots/flow_{tid}_{int(current_time)}.jpg"
                        cv.imwrite(filename, frame)
                elif congestion_count >= SMOOTHING_FRAMES:
                    cv.putText(frame,"CONGESTION",(50,150),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                    if congestion_ss==0:
                        congestion_ss=1
                        current_time = time.time()
                        filename = f"screenshots/congestion_{tid}_{int(current_time)}.jpg"
                        cv.imwrite(filename, frame)
        cv.imshow("Crowd Flow", frame)

        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()