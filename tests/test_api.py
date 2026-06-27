import numpy as np
import json
from collections import defaultdict

import cv2
import base64
import matplotlib.pyplot as plt
from matplotlib import rcParams

import websockets
from loguru import logger
import asyncio

fig = None
img_display = None

def setup(idx):
    global fig
    global img_display

    cap = cv2.VideoCapture(idx)
    _, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    rcParams["toolbar"] = "None"
    plt.ion()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")

    img_display = ax.imshow(rgb_frame)

    plt.show(block=False)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

async def stream(idx):
    setup(idx)
    track_history = defaultdict(list)

    async with websockets.connect("ws://localhost:8001/track") as ws:
        cap = cv2.VideoCapture(idx)

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("Video has done")
                break

            annotated_frame = frame.copy()

            _, buffer = cv2.imencode(".jpg", frame)
            await ws.send(buffer.tobytes())

            result = await ws.recv()
            res = json.loads(result)
            detections = res['detection']
            annotations = res['annotated_frame']

            for det in detections:
                track_id = det['track_id']
                bbox = det['bbox']
                track_history[track_id].append(bbox)

            frame_bytes = base64.b64decode(annotations)
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            annotated_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

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

video_path = 'test.mp4'
asyncio.run(stream(video_path))