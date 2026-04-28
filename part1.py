import cv2 as cv
import numpy as np
from ultralytics import YOLO
import pygame
from main import downscale

def init_sound():
    pygame.mixer.init()
    alert_sound = pygame.mixer.Sound("sound.mp3")
    return alert_sound


def Load_model():
    model = YOLO("yolov8n.pt")
    return model

def Capture_video(source):
    cap = cv.VideoCapture(source)
    if not cap.isOpened():
        raise Exception("Video not Found") 
    return cap
def detect_and_track(model, frame):
    results = model.track(
        frame,
        persist=True,
        classes=[0],            # person only
        tracker="bytetrack.yaml"
    )
    return results
def restricted_zone(frame):
    h, w = frame.shape[:2]

    zone = np.array([
    [w // 4, h // 4],
    [3 * w // 4, h // 4],
    [3 * w // 4, 3 * h // 4],
    [w // 4, 3 * h // 4]
], dtype=np.int32)

    return zone.reshape((-1, 1, 2))

def Centroid(box):
    x1,y1,x2,y2=map(int,box)
    cx=(x1+x2)//2
    cy=(y1+y2)//2
    return (cx,cy)

def is_inside_zone(point, zone):
    point = (int(point[0]), int(point[1]))
    return cv.pointPolygonTest(zone, point, False) >= 0


def check_intrusion(track_id, inside_now, object_state,alert_channel): 
    was_inside = object_state.get(track_id, False)
    if inside_now and not was_inside:
        alert_channel.play()
        print(f"INTRUSION DETECTED Object ID: {track_id}")
    object_state[track_id] = inside_now

def draw_visuals(frame, box, centroid, track_id, inside, zone):
    x1, y1, x2, y2 = map(int, box)
    color = (0, 0, 255) if inside else (0, 255, 0)

    cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv.circle(frame, centroid, 4, (255, 0, 0), -1)
    cv.putText(frame, f"ID:{track_id}",
                (x1, y1 - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2)

    cv.polylines(frame, [zone], True, (0, 0, 255), 2)


def main():
    model = Load_model()
    source = 0
    cap = Capture_video(source)
    alert_channel=init_sound()
    object_state = {}   # track_id → inside/outside
    ret2, frame2 = cap.read()
    if not ret2 or frame2 is None:
     return

    frame2 = downscale(frame2, 1400, 1300)
    zone = restricted_zone(frame2)   # CREATE ONCE

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = downscale(frame, 1400, 1300)
        results = detect_and_track(model, frame)

        cv.polylines(frame, [zone], True, (0, 0, 255), 2) 

        if results[0].boxes.id is not None:
            for box, track_id in zip(
                    results[0].boxes.xyxy,
                    results[0].boxes.id):

                track_id = int(track_id)
                centroid = Centroid(box)
                inside = is_inside_zone(centroid, zone)

                check_intrusion(track_id, inside, object_state,alert_channel)
                draw_visuals(frame, box, centroid, track_id, inside, zone)

        cv.imshow("Intrusion Detection", frame)
        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
