# app.py — Minimal MJPEG stream from Picamera2
import time
import cv2
from flask import Flask, Response, render_template_string
from picamera2 import Picamera2

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

def mjpeg_generator(jpeg_quality=80, fps=15):
    delay = 1.0 / fps
    while True:
        frame = picam2.capture_array()  # RGB
        # Encode to JPEG for MJPEG streaming
        ok, jpg = cv2.imencode(".jpg", frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])

        if not ok:
            continue
        buf = jpg.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf + b"\r\n")
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
