# app.py — Minimal MJPEG stream from Picamera2
import time
import cv2
from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
import numpy as np

last_frame_gray = None
motion_detected = False

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

def mjpeg_generator(jpeg_quality=80, fps=15, min_area=1200): # play around between 1000 - 2000
    global last_frame_gray, motion_detected
    delay = 1.0 / fps
    while True:
        frame = picam2.capture_array()  # RGB
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if last_frame_gray is None:
            last_frame_gray = gray
            # Encode first frame to show something
            ok, jpg = cv2.imencode(".jpg", frame,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                       jpg.tobytes() + b"\r\n")
            time.sleep(delay)
            continue

        # Compute absolute difference between current frame and previous
        frame_delta = cv2.absdiff(last_frame_gray, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        motion_detected = False
        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue
            motion_detected = True
            break

        last_frame_gray = gray

        # Display motion status on frame
        label = "Motion Detected" if motion_detected else "No Motion"
        cv2.putText(
            frame, label, (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )

        # Encode to JPEG for MJPEG streaming
        ok, jpg = cv2.imencode(".jpg", frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if not ok:
            continue

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               jpg.tobytes() + b"\r\n")
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
