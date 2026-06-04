from dotenv import load_dotenv
import os
from ultralytics import YOLO
import cv2

load_dotenv()

model_path = f"{os.getenv('RUNS_DIR')}/detect/car-counting/train/weights/best.onnx"
model = YOLO(model_path)

video_path = 'test.mp4'
cap = cv2.VideoCapture(video_path)
output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(output_dir, exist_ok=True)
frame_idx = 0

# if not cap.isOpened():
#     raise RuntimeError('Cannot open camera device 0')

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print('No frame received from camera, exiting.')
            break

        results = model(frame, stream=True)
        
        for r in results:
            annotated_frame = r.plot()
            detections = len(r.boxes) if hasattr(r, 'boxes') else 0
            print(f'Frame {frame_idx}: detected {detections} object(s)')
            output_path = os.path.join(output_dir, f'frame_{frame_idx:06d}.jpg')
            cv2.imwrite(output_path, annotated_frame)
            print(f'  saved annotated frame to: {output_path}')
            frame_idx += 1
except KeyboardInterrupt:
    print('Interrupted by user')
finally:
    cap.release()
    print('Camera released')