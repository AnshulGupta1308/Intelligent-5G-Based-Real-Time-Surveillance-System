import cv2 as cv
import numpy as np
from ultralytics import YOLO
import pygame
import time
from main import downscale, open_rtsp

LOITER_THRESHOLD = 7 

def init_sound():
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    return pygame.mixer.Sound("sound.mp3")

def Load_model():
    return YOLO("yolov8n.pt")

def detect_and_track(model, frame):
    return model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml")

def restricted_zone(frame):
    h, w = frame.shape[:2]
    zone = np.array([
        [w//4, h//4],
        [3*w // 4, h//4],
        [3*w // 4, 3*h // 4],
        [w // 4, 3*h // 4]
    ], dtype=np.int32)
    return zone.reshape((-1, 1, 2))

def Centroid(box):
    x1, y1, x2, y2 = map(int, box)
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def is_inside_zone(point, zone):
    point = (int(point[0]), int(point[1]))
    return cv.pointPolygonTest(zone, point, False) >= 0

loiter_start_time = {}
loitering_ids = set()

def check_loitering(track_id, inside, frame, box, alarm):
    now = time.time()

    if inside and track_id not in loiter_start_time:
        loiter_start_time[track_id] = now

    elif inside:
        duration = now - loiter_start_time[track_id]
        x1, y1, x2, y2 = map(int, box)

        cv.putText(frame, f"{duration:.1f}s",
                   (x1, y2 + 20),
                   cv.FONT_HERSHEY_SIMPLEX,
                   0.6, (0,255,255), 2)

        if duration > LOITER_THRESHOLD and track_id not in loitering_ids:
            loitering_ids.add(track_id)
            alarm.play()

    elif not inside and track_id in loiter_start_time:
        del loiter_start_time[track_id]
        loitering_ids.discard(track_id)

def draw_visuals(frame, box, centroid, track_id, inside, zone):
    x1, y1, x2, y2 = map(int, box)

    if track_id in loitering_ids:
        color = (0,0,255)
        label = "LOITERING DETECTED"
    elif inside:
        color = (0,255,255)
        label = f"ID:{track_id}"
    else:
        color = (0,255,0)
        label = f"ID:{track_id}"

    cv.rectangle(frame, (x1,y1),(x2,y2),color,2)
    cv.circle(frame, centroid, 4, (255,0,0), -1)

    cv.putText(frame,label,(x1,y1-10),
               cv.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    cv.polylines(frame,[zone],True,(0,0,255),2)

def main():

    model = Load_model()

    cap = cv.VideoCapture(0)
    alarm = init_sound()

    ret, frame = cap.read()
    if not ret:
        print("No frames")
        return

    frame = downscale(frame,1400,1300)
    zone = restricted_zone(frame)

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break
        frame = downscale(frame, 1400, 1300)
        cv.polylines(frame, [zone], True, (0, 0, 255), 2) 
        

        results = detect_and_track(model,frame)

        if results[0].boxes.id is not None:
            for box, track_id in zip(results[0].boxes.xyxy,
                                     results[0].boxes.id):

                track_id=int(track_id)

                centroid=Centroid(box)
                inside=is_inside_zone(centroid,zone)

                check_loitering(track_id,inside,frame,box,alarm)
                draw_visuals(frame,box,centroid,track_id,inside,zone)

        cv.imshow("Loitering Detection",frame)

        if cv.waitKey(1)&0xFF==27:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__=="__main__":
    main()