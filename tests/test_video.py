from dotenv import load_dotenv
import os
import time
import mlflow
import cv2
import matplotlib.pyplot as plt
from matplotlib import rcParams

load_dotenv()

model_name = os.getenv('MLFLOW_MODEL_NAME')
alias = os.getenv('MLFLOW_MODEL_ALIAS')
model = mlflow.pyfunc.load_model(f'models:/{model_name}@{alias}')

video_path = os.path.join(os.path.dirname(__file__), 'test.mp4')
cap = cv2.VideoCapture(video_path)

# Fallback to matplotlib interactive display
fig = None
img_display = None

ret, frame = cap.read()
if not ret:
    raise RuntimeError(f"Cannot read first frame from {video_path}")

rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
rcParams['toolbar'] = 'None'
plt.ion()
fig, ax = plt.subplots()
ax.axis('off')
img_display = ax.imshow(rgb_frame)
plt.show(block=False)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Video ended, exiting.')
            break

        params = {'imgsz': 224, 'conf': 0.25, 'stream': True}
        results = model.predict(frame, params)

        annotated_frame = frame
        for r in results:
            annotated_frame = r.plot()

        if annotated_frame is None:
            annotated_frame = frame
        
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img_display.set_data(rgb_frame)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        if not plt.fignum_exists(fig.number):
            print('Window closed, exiting.')
            break
        time.sleep(0.01)

except KeyboardInterrupt:
    print('Interrupted by user')
finally:
    cap.release()
    plt.close(fig)
    print('Video display finished.')