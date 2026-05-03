import cv2 as cv
import numpy as np
from ultralytics import YOLO
import time
import math
import os

model = YOLO("yolov8m.pt")

PERSON_CLASS = 0
OBJECT_CLASSES = [24, 26, 28, 63, 67, 73, 74]

DIST_THRESHOLD = 100
TIME_THRESHOLD = 10
STATIONARY_FRAMES = 30

object_tracks = {}
person_tracks = {}
associations = {}
stationary_counter = {}
abandoned = {}
saved = set()

os.makedirs("abandoned", exist_ok=True)

def centroid(box):
    x1,y1,x2,y2 = map(int, box)
    return ((x1+x2)//2, (y1+y2)//2)

def distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def main():
    cap = cv.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, tracker="bytetrack.yaml")

        if results[0].boxes.id is None:
            cv.imshow("Abandonment Detection", frame)
            if cv.waitKey(1) & 0xFF == 27:
                break
            continue

        boxes = results[0].boxes.xyxy
        ids = results[0].boxes.id
        classes = results[0].boxes.cls

        ids_list = list(map(int, ids))

        current_time = time.time()

        current_persons = {}
        current_objects = {}

        for box, tid, cls in zip(boxes, ids_list, classes):
            cls = int(cls)
            c = centroid(box)

            if cls == PERSON_CLASS:
                current_persons[tid] = c
                person_tracks[tid] = c
            elif cls in OBJECT_CLASSES:
                current_objects[tid] = c

        for oid, oc in current_objects.items():
            if oid not in associations:
                min_dist = float("inf")
                owner = None
                for pid, pc in current_persons.items():
                    d = distance(oc, pc)
                    if d < min_dist:
                        min_dist = d
                        owner = pid
                if owner is not None:
                    associations[oid] = owner

        for oid, oc in current_objects.items():
            prev = object_tracks.get(oid)
            if prev is not None:
                if distance(prev, oc) < 5:
                    stationary_counter[oid] = stationary_counter.get(oid, 0) + 1
                else:
                    stationary_counter[oid] = 0
            object_tracks[oid] = oc

        for oid, oc in current_objects.items():
            if oid in associations:
                owner = associations[oid]

                if owner in current_persons:
                    d = distance(oc, current_persons[owner])

                    if d > DIST_THRESHOLD and stationary_counter.get(oid,0) > STATIONARY_FRAMES:
                        if oid not in abandoned:
                            abandoned[oid] = current_time

                    if oid in abandoned and current_time - abandoned[oid] > TIME_THRESHOLD:
                        if oid not in saved:
                            saved.add(oid)

                            obj_index = ids_list.index(oid)
                            obj_box = boxes[obj_index]

                            owner_box = None
                            if owner in ids_list:
                                owner_index = ids_list.index(owner)
                                owner_box = boxes[owner_index]

                            frame_copy = frame.copy()

                            x1,y1,x2,y2 = map(int, obj_box)
                            cv.rectangle(frame_copy,(x1,y1),(x2,y2),(0,0,255),2)
                            cv.putText(frame_copy,"ABANDONED",(x1,y1-10),cv.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

                            if owner_box is not None:
                                x1,y1,x2,y2 = map(int, owner_box)
                                cv.rectangle(frame_copy,(x1,y1),(x2,y2),(255,0,0),2)
                                cv.putText(frame_copy,"OWNER",(x1,y1-10),cv.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)

                            filename = f"abandoned/abandoned_{oid}_{int(current_time)}.jpg"
                            cv.imwrite(filename, frame_copy)

                else:
                    if oid not in abandoned:
                        abandoned[oid] = current_time

        for box, tid, cls in zip(boxes, ids_list, classes):
            cls = int(cls)
            x1,y1,x2,y2 = map(int, box)

            if cls == PERSON_CLASS:
                cv.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            elif cls in OBJECT_CLASSES:
                color = (255,255,0)
                if tid in abandoned and time.time() - abandoned[tid] > TIME_THRESHOLD:
                    color = (0,0,255)
                cv.rectangle(frame,(x1,y1),(x2,y2),color,2)

        cv.imshow("Abandonment Detection", frame)

        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()