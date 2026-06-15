from collections import defaultdict
from dotenv import load_dotenv

import cv2
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import mlflow
from loguru import logger

load_dotenv()

LINE = (700, 120, 1280, 120)

model_name = os.getenv('MLFLOW_MODEL_NAME')
alias = os.getenv('MLFLOW_MODEL_ALIAS')
model = mlflow.pyfunc.load_model(f'models:/{model_name}@{alias}', model_config={'predict_fn': 'track'})

video_path = os.path.join(
    os.path.dirname(__file__),
    "test.mp4"
)

track_history = defaultdict(list)
counted_ids = set()

count_in = 0
count_out = 0

def get_center(box):
    x1, y1, x2, y2 = box
    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )

def side_of_line(point, line):
    x, y = point
    x1, y1, x2, y2 = line

    return (
        (x - x1) * (y2 - y1)
        - (y - y1) * (x2 - x1)
    )

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
            print("Video has done.")
            break

        results = model.predict(frame)
        annotated_frame = frame.copy()

        for r in results:
            annotated_frame = r.plot()

            if r.boxes.id is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            track_ids = r.boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):

                track_history[track_id].append(box)

                if len(track_history[track_id]) < 2:
                    continue

                prev_box = track_history[track_id][-2]
                curr_box = track_history[track_id][-1]

                prev_center = get_center(prev_box)
                curr_center = get_center(curr_box)

                prev_side = side_of_line(prev_center, LINE)
                curr_side = side_of_line(curr_center, LINE)

                crossed = (
                    prev_side * curr_side < 0
                )

                if crossed and track_id not in counted_ids:

                    if curr_center[1] > prev_center[1]:
                        count_in += 1

                    # Bergerak naik
                    else:
                        count_out += 1

                    counted_ids.add(track_id)

        cv2.line(
            annotated_frame,
            (LINE[0], LINE[1]),
            (LINE[2], LINE[3]),
            (0, 255, 0),
            3
        )

        cv2.putText(
            annotated_frame,
            f"IN : {count_in}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            f"OUT: {count_out}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
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
            print("Window closed.")
            break

        cv2.waitKey(1)

except KeyboardInterrupt:
    print("User stopped the counting.")

finally:
    cap.release()
    plt.close()