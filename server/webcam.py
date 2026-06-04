from dotenv import load_dotenv
import os
import threading
import time
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from ultralytics import YOLO
import cv2

load_dotenv()

model_path = f"{os.getenv('RUNS_DIR')}/detect/car-counting/train/weights/best.onnx"
model = YOLO(model_path)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError('Cannot open camera device 0')

latest_frame = None
frame_lock = threading.Lock()
frame_available = threading.Event()

app = FastAPI()

@app.get('/', response_class=HTMLResponse)
def index():
    return '''
    <html>
      <head>
        <title>YOLO ONNX Stream</title>
      </head>
      <body>
        <h1>YOLO ONNX Live Stream</h1>
        <img src="/video_feed" width="640" height="480" />
        <p>Tekan Ctrl+C di terminal untuk menghentikan server.</p>
      </body>
    </html>
    '''

@app.get('/video_feed')
def video_feed():
    return StreamingResponse(frame_generator(), media_type='multipart/x-mixed-replace; boundary=frame')


def frame_generator():
    while True:
        frame_available.wait()
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)


def capture_loop():
    global latest_frame
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('No frame received from camera, retrying...')
                time.sleep(0.1)
                continue

            results = model(frame, stream=True)
            for r in results:
                annotated_frame = r.plot()
                detections = len(r.boxes) if hasattr(r, 'boxes') else 0
                print(f'Frame {frame_idx}: detected {detections} object(s)')

                success, buffer = cv2.imencode('.jpg', annotated_frame)
                if not success:
                    print('Failed to encode frame')
                    continue

                with frame_lock:
                    latest_frame = buffer.tobytes()
                    frame_available.set()

                frame_idx += 1
                break

            time.sleep(0.01)
    except Exception as exc:
        print(f'Capture loop stopped: {exc}')
    finally:
        cap.release()
        print('Camera released')


if __name__ == '__main__':
    thread = threading.Thread(target=capture_loop, daemon=True)
    thread.start()

    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
