# app.py — Minimal MJPEG stream from Picamera2
import time
import cv2
from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
import numpy as np
import subprocess
import uuid
import threading

last_frame_gray = None
motion_counter = 0
motion_threshold = 3
motion_detected = False
last_prediction = "No species yet"
prediction_running = False
last_frame_gray = None
motion_detected = False
capture_lock = threading.Lock()
MOTION_HOLD_SECONDS = 20
last_motion_ts = 0.0  # epoch seconds

# --- Camera setup ---
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (1280, 720), "format": "RGB888"},
    buffer_count=4
)
picam2.configure(config)

# If the image is upside down, uncomment one or both:
# picam2.set_controls({"VFlip": True})
# picam2.set_controls({"HFlip": True})

picam2.start()
time.sleep(0.5)  # small warm-up

app = Flask(__name__)

PAGE = """
<!doctype html>
<title>Bird Stream</title>
<style>body{margin:0;background:#111;display:flex;justify-content:center;align-items:center;height:100vh}</style>
<img src="/stream" alt="Bird Stream" />
"""

def run_prediction():
    global last_prediction, prediction_running
    try:
        time.sleep(2)  # wait before capture
        # take a fresh frame AFTER the delay, safely
        with capture_lock:
            fresh = picam2.capture_array()
        img_path = f"/tmp/{uuid.uuid4().hex}.jpg"
        cv2.imwrite(img_path, fresh)
        result = subprocess.check_output(["python3", "predict_species.py", img_path])
        last_prediction = result.decode().strip()
        print("Predicted species:", last_prediction)
    except Exception as e:
        print("Prediction failed:", e)
        last_prediction = "Prediction error"
    finally:
        prediction_running = False


def mjpeg_generator(jpeg_quality=80, fps=15, min_area=1000):
    global last_frame_gray, motion_counter, motion_detected
    global last_prediction, prediction_running, last_motion_ts

    delay = 1.0 / fps

    while True:
        with capture_lock:
            frame = picam2.capture_array()  # RGB

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if last_frame_gray is None:
            last_frame_gray = gray.copy().astype("float")
            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
            time.sleep(delay)
            continue

        # Background model and motion mask
        alpha = 0.05
        cv2.accumulateWeighted(gray, last_frame_gray, alpha)
        frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(last_frame_gray))
        thresh = cv2.threshold(frame_delta, 40, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_this_frame = any(cv2.contourArea(c) >= min_area for c in contours)

        if motion_this_frame:
            motion_counter += 1
        else:
            motion_counter = 0

        # Trigger prediction and stamp the last-motion time
        if motion_counter >= motion_threshold and not prediction_running:
            motion_detected = True
            last_motion_ts = time.time()  # <-- keep motion "alive" from now
            print("Motion detected, starting prediction thread...")
            prediction_running = True
            threading.Thread(target=run_prediction, daemon=True).start()
        elif motion_this_frame:
            # refresh the hold if we keep seeing motion
            last_motion_ts = time.time()

        # Compute whether we should DISPLAY "Motion Detected"
        now = time.time()
        display_motion = (now - last_motion_ts) < MOTION_HOLD_SECONDS
        motion_detected = display_motion  # keep your existing flag consistent

        # === Draw overlays with background (do this EVERY frame) ===
        species_text = f"Species: {last_prediction}"
        motion_text = "Motion Detected" if display_motion else "No Motion"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.8
        thickness = 2
        text_color = (255, 255, 255)  # white

        (w1, h1), _ = cv2.getTextSize(species_text, font, scale, thickness)
        (w2, h2), _ = cv2.getTextSize(motion_text, font, scale, thickness)

        pad = 10
        spacing = 8
        box_width = max(w1, w2) + 2 * pad
        box_height = h1 + h2 + spacing + 3 * pad

        frame_height, frame_width = frame.shape[:2]
        x1, y1 = 5, frame_height - 5 - box_height
        x2, y2 = x1 + box_width, frame_height - 5

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 170, 86), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.6, 0, frame)

        cv2.putText(frame, species_text, (x1 + pad, frame_height - box_height + pad + h1),
                    font, scale, text_color, thickness)
        cv2.putText(frame, motion_text, (x1 + pad, frame_height - box_height + pad + h1 + spacing + h2),
                    font, scale, text_color, thickness)

        # Stream the frame
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")

        time.sleep(delay)

@app.route("/")
def index():
    return render_template_string(PAGE)

@app.route("/stream")
def stream():
    return Response(mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    # 0.0.0.0 makes it reachable from your LAN; port 8080 is arbitrary
    app.run(host="0.0.0.0", port=8080, threaded=True)
