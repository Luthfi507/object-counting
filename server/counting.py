from collections import defaultdict
from dotenv import load_dotenv

import cv2
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import mlflow
from loguru import logger

load_dotenv()

LINE = (700, 120, 1300, 120)

model_name = os.getenv('MLFLOW_MODEL_NAME')
alias = os.getenv('MLFLOW_MODEL_ALIAS')
params = {
    'mode': 'track',
    'persist': True,
    'verbose': False,
    'tracker': 'bytetrack.yaml',
    'augment': False,
    'conf': 0.4
}

model = mlflow.pyfunc.load_model(f'models:/{model_name}@{alias}', model_config={'params': params, 'predict_fn': 'track'})

video_path = os.path.join(
    os.path.dirname(__file__),
    "test.mp4"
)

track_history = defaultdict(list)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

rcParams["toolbar"] = "None"

plt.ion()

fig, ax = plt.subplots(figsize=(12, 7))
ax.axis("off")

img_display = ax.imshow(rgb_frame)

plt.show(block=False)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("Video has done")
            break

        results = model.predict(frame)
        annotated_frame = frame.copy()

        for r in results:
            annotated_frame = r.plot()
            boxes = r.boxes

            if boxes.id is None:
                continue

            track_ids = boxes.id.cpu().numpy().astype(int)
            boxes = boxes.xyxy.cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                track_history[track_id].append(box)

        cv2.putText(
            annotated_frame,
            f"Count: {len(list(track_history.keys()))}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        rgb_frame = cv2.cvtColor(
            annotated_frame,
            cv2.COLOR_BGR2RGB
        )

        img_display.set_data(rgb_frame)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        if not plt.fignum_exists(fig.number):
            logger.info("Window closed")
            break

        cv2.waitKey(1)

except KeyboardInterrupt:
    logger.info("User stopped the process")
finally:
    cap.release()
    plt.close()