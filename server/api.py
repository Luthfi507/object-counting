import os
import numpy as np
from PIL import Image
from io import BytesIO
from time import time
from loguru import logger
from fastapi import FastAPI, UploadFile, File, HTTPException

from utils import ModelVersion

mv = ModelVersion(
    os.getenv('MLFLOW_MODEL_NAME'),
    os.getenv('MLFLOW_MODEL_ALIAS'),
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
    
# TODO: build websocket for real time track prediction