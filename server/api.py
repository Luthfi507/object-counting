import os
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import cv2
from time import time
from loguru import logger
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect

from utils import ModelVersion

mv = ModelVersion(
    os.getenv('MLFLOW_MODEL_NAME'),
    os.getenv('MLFLOW_MODEL_ALIAS'),
)

params = {
    'mode': 'track',
    'persist': True,
    'verbose': False,
    'tracker': 'bytetrack.yaml',
    'augment': False,
    'conf': 0.4
}
mv_track = ModelVersion(
    os.getenv('MLFLOW_MODEL_NAME'),
    os.getenv('MLFLOW_MODEL_ALIAS'),
    {'params': params, 'predict_fn': 'track'}
)

app = FastAPI()

@app.get("/")
async def home():
    return "Hay my mine gwehh"

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    start = time()
    contents = await image.read()
    image_data = Image.open(BytesIO(contents)).convert("RGB")
    image_array = np.array(image_data)

    try:
        result = mv.predict(image_array)[0]

        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "class_name": result.names[int(box.cls)],
                "confidence": round(float(box.conf), 4),
                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                }
            })

        total_time = time() - start
        return {
            "prediction": detections,
            "total_time": round(total_time, 4)
        }
    except Exception as e:
        logger.exception(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# websocket for real time track prediction
@app.websocket("/track")
async def track(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_bytes()
            detections = []
            
            try:
                np_arr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                results = mv_track.predict(frame)

                for r in results:
                    boxes = r.boxes
                    annotated_frame = r.plot()

                    if boxes.id is None:
                        continue

                    track_ids = boxes.id.cpu().numpy().astype(int)
                    bboxes = boxes.xyxy.cpu().numpy()
                    cls_ids = boxes.cls.cpu().numpy().astype(int)

                    for box, track_id, cls_id in zip(bboxes, track_ids, cls_ids):
                        x1, y1, x2, y2 = box.tolist()
                        detections.append({
                            "track_id": int(track_id),
                            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                            "name": r.names[int(cls_id)]
                        })

                    _, encoded_img = cv2.imencode('.jpg', annotated_frame)
                    annotated_b64 = base64.b64encode(encoded_img.tobytes()).decode('utf-8')

                    await websocket.send_json({
                        "detection": detections,
                        "annotated_frame": annotated_b64
                    })

            except Exception as e:
                logger.exception(f"Error prediction: {e}")
        
    except WebSocketDisconnect:
        logger.info("Websocket disconnected")
    except Exception as e:
        logger.exception(f"Error when tracking: {e}")
        await websocket.close()