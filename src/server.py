import os
import json
import base64
import time
from io import BytesIO
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from ultralytics import YOLO
from dotenv import load_dotenv

from .analytics import Analytics, PredictionMetric

# Load environment variables
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# Initialize FastAPI app
app = FastAPI(title="Object Counting API", version="1.0.0")

project_dir = os.path.dirname(__file__)
static_dir = os.path.join(project_dir, 'static')
app.mount('/static', StaticFiles(directory=static_dir), name='static')

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
class ModelManager:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.model_version = None
        self.load_latest_model()
    
    def load_latest_model(self):
        """Load the latest production model from MLflow"""
        try:
            # Get the latest run from the experiment
            experiment = mlflow.get_experiment_by_name("Object-Counting-Experiment")
            if not experiment:
                logger.warning("Experiment not found, will try to load from file")
                self._load_local_model()
                return
            
            # Get all runs sorted by start time (latest first)
            client = mlflow.tracking.MlflowClient()
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if runs:
                latest_run = runs[0]
                run_id = latest_run.info.run_id
                
                # Load model from MLflow
                model_uri = f"runs:/{run_id}/model"
                self.model = mlflow.pyfunc.load_model(model_uri)
                self.model_name = f"MLflow-{run_id[:8]}"
                self.model_version = latest_run.data.params.get("epochs", "unknown")
                logger.info(f"Loaded model from MLflow: {self.model_name}")
            else:
                logger.warning("No runs found in experiment, loading local model")
                self._load_local_model()
        except Exception as e:
            logger.error(f"Error loading model from MLflow: {e}")
            self._load_local_model()
    
    def _load_local_model(self):
        """Fallback to load local model file"""
        try:
            project_dir = os.path.dirname(__file__)
            model_path = os.path.join(project_dir, '..', 'runs', 'detect', 'train', 'weights', 'best.onnx')
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                self.model_name = "Local-YOLO"
                self.model_version = "local"
                logger.info(f"Loaded local model: {self.model_name}")
            else:
                logger.error(f"Local model not found at {model_path}")
        except Exception as e:
            logger.error(f"Error loading local model: {e}")
    
    def predict(self, image_path):
        """Run prediction on image"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            results = self.model.predict(source=image_path, conf=0.25)
            return results[0]
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

# Initialize model manager
model_manager = ModelManager()

# Initialize analytics
analytics = Analytics()


# Helper functions
def image_to_base64(image_array):
    """Convert numpy array to base64 string"""
    img = Image.fromarray(image_array.astype('uint8'))
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode()

def draw_predictions(image_array, results):
    """Draw bounding boxes on image"""
    image = image_array.copy()
    boxes = results.boxes
    
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            label = f"{results.names[cls]} ({conf:.2f})"
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Put text
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
    
    return image

# API Routes

@app.get("/")
async def root():
    index_path = os.path.join(static_dir, 'index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail='Playground not found')

@app.get("/api/model-info")
async def model_info():
    """Get current model information"""
    return {
        "model_name": model_manager.model_name,
        "model_version": model_manager.model_version,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """Predict on uploaded image"""
    start_time = time.time()
    
    try:
        # Read and save uploaded file
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        image_array = np.array(image)
        
        # Save temporarily
        temp_path = f"/tmp/temp_image_{datetime.now().timestamp()}.jpg"
        image.save(temp_path)
        
        # Run prediction
        results = model_manager.predict(temp_path)
        
        # Draw predictions
        annotated_image = draw_predictions(image_array, results)
        
        # Extract detections
        detections = []
        if results.boxes is not None:
            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                class_name = results.names[cls]
                
                detections.append({
                    "id": i,
                    "class": class_name,
                    "confidence": round(conf, 3),
                    "bbox": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2)
                    }
                })
                
                # Track analytics
                analytics.add_detection(class_name)
        
        # Track prediction metric
        inference_time = time.time() - start_time
        metric = PredictionMetric(
            timestamp=datetime.now(),
            inference_time=inference_time,
            object_count=len(detections),
            image_size=image_array.shape[:2],
            model_name=model_manager.model_name
        )
        analytics.add_prediction(metric)
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return {
            "success": True,
            "image": image_to_base64(annotated_image),
            "detections": detections,
            "count": len(detections),
            "inference_time": round(inference_time, 4),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time predictions"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "ping":
                await websocket.send_json({"type": "pong"})
            
            elif data["type"] == "predict":
                # Decode image from base64
                image_data = base64.b64decode(data["image"])
                image = Image.open(BytesIO(image_data))
                image_array = np.array(image)
                
                # Save temporarily
                temp_path = f"/tmp/ws_image_{datetime.now().timestamp()}.jpg"
                image.save(temp_path)
                
                ws_start_time = time.time()
                
                try:
                    # Run prediction
                    results = model_manager.predict(temp_path)
                    
                    # Draw predictions
                    annotated_image = draw_predictions(image_array, results)
                    
                    # Extract detections
                    detections = []
                    if results.boxes is not None:
                        for i, box in enumerate(results.boxes):
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0].cpu().numpy())
                            cls = int(box.cls[0].cpu().numpy())
                            class_name = results.names[cls]
                            
                            detections.append({
                                "id": i,
                                "class": class_name,
                                "confidence": round(conf, 3),
                                "bbox": {
                                    "x1": float(x1),
                                    "y1": float(y1),
                                    "x2": float(x2),
                                    "y2": float(y2)
                                }
                            })
                            
                            # Track analytics
                            analytics.add_detection(class_name)
                    
                    # Track prediction metric
                    ws_inference_time = time.time() - ws_start_time
                    metric = PredictionMetric(
                        timestamp=datetime.now(),
                        inference_time=ws_inference_time,
                        object_count=len(detections),
                        image_size=image_array.shape[:2],
                        model_name=model_manager.model_name
                    )
                    analytics.add_prediction(metric)
                    
                    await websocket.send_json({
                        "type": "prediction",
                        "image": image_to_base64(annotated_image),
                        "detections": detections,
                        "count": len(detections),
                        "inference_time": round(ws_inference_time, 4)
                    })
                
                except Exception as e:
                    logger.error(f"WebSocket prediction error: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
                
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

# Analytics Endpoints

@app.get("/api/analytics/summary")
async def analytics_summary():
    """Get analytics summary"""
    return analytics.get_summary()

@app.get("/api/analytics/classes")
async def analytics_classes():
    """Get class statistics"""
    return analytics.get_class_statistics()

@app.get("/api/analytics/performance")
async def analytics_performance(minutes: int = 5):
    """Get performance statistics for the last N minutes"""
    return analytics.get_performance_stats(minutes)

@app.get("/api/analytics/history")
async def analytics_history(limit: int = 100):
    """Get recent prediction history"""
    return analytics.get_history(limit)

@app.get("/api/analytics/export")
async def analytics_export():
    """Export analytics to JSON"""
    export_path = "/tmp/analytics_export.json"
    analytics.export_json(export_path)
    return FileResponse(export_path, filename="analytics.json")


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Object Counting API Server")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
