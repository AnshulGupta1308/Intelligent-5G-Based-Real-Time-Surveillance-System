import cv2 as cv

def downscale(frame, max_width=800, max_height=600):
    h, w = frame.shape[:2]
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv.resize(frame, (new_w, new_h), interpolation=cv.INTER_AREA)

def open_rtsp(rtsp_url):
    cap = cv.VideoCapture(rtsp_url)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise Exception(f"Failed to open RTSP stream: {rtsp_url}")
    return cap