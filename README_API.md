# 🎯 Object Counting API & Real-time Playground

Complete REST API and interactive web playground for real-time object detection using YOLO11 and MLflow.

## ✨ Key Features

### 🎮 Interactive Web Playground
- **Upload Mode**: Drag-and-drop image upload with instant predictions
- **Webcam Mode**: Real-time capture and detection from your camera
- **Live Results**: Bounding boxes with confidence scores
- **Analytics Dashboard**: Real-time statistics and metrics
- **Model Info**: Display current model and version

### 🔌 Comprehensive REST API
```
POST   /api/predict                 - Upload image for prediction
GET    /api/model-info              - Get model information
GET    /api/analytics/summary       - Overall statistics
GET    /api/analytics/classes       - Per-class detection counts
GET    /api/analytics/performance   - Performance metrics (5min/60min)
GET    /api/analytics/history       - Recent prediction history
GET    /api/analytics/export        - Export analytics as JSON
WS     /ws/stream                   - WebSocket real-time predictions
```

### 📊 Advanced Analytics
- **Real-time Tracking**: Predictions, object counts, inference time
- **Performance Metrics**: Average/min/max inference time
- **Class Statistics**: Per-object-class detection counts
- **Prediction History**: Recent prediction records with timestamps
- **Export**: Download analytics data as JSON

### 🤖 MLflow Integration
- **Auto-Load Latest Model**: Automatically loads most recent trained model
- **Fallback Support**: Falls back to local model if MLflow unavailable
- **Version Tracking**: Model version stored in analytics

### 🐳 Production Ready
- Docker & Docker Compose support
- Environment configuration via .env
- GPU acceleration support
- CORS middleware enabled
- Comprehensive logging

## 🚀 Quick Start

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run Automated Setup
```bash
python quickstart.py
```

### 3️⃣ Start the Server
```bash
python -m src.server
```

### 4️⃣ Open Playground
```
🌍 http://localhost:8000
📖 API Docs: http://localhost:8000/docs
```

## 📖 Detailed Guides

- **[API Documentation](API_README.md)** - Complete API reference with examples
- **[Deployment Guide](DEPLOYMENT_GUIDE.md)** - Production deployment guide
- **[Quick Start Guide](#quick-start-section)** - Getting started (below)

## 🎯 Usage Examples

### Web Playground
1. Open http://localhost:8000
2. Choose **Upload** or **Webcam** mode
3. Upload image or capture from camera
4. View results and analytics in real-time

### Command Line
```bash
# Test connection
python client.py --test

# Get model info
python client.py --info

# Predict on single image
python client.py --image path/to/image.jpg

# Batch predict on directory
python client.py --batch path/to/images/
```

### REST API
```bash
# Upload prediction
curl -X POST "http://localhost:8000/api/predict" \
  -F "file=@image.jpg"

# Get analytics summary
curl "http://localhost:8000/api/analytics/summary" | jq

# Export analytics
curl "http://localhost:8000/api/analytics/export" -o analytics.json
```

### Python Integration
```python
import requests

# Make prediction
with open('image.jpg', 'rb') as f:
    response = requests.post(
        "http://localhost:8000/api/predict",
        files={'file': f}
    )

result = response.json()
print(f"Detected {result['count']} objects in {result['inference_time']}s")

# Get analytics
analytics = requests.get("http://localhost:8000/api/analytics/summary").json()
print(f"Total predictions: {analytics['total_predictions']}")
```

### WebSocket (Real-time Streaming)
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/stream');

ws.onopen = () => {
  // Send image as base64
  ws.send(JSON.stringify({
    type: 'predict',
    image: 'data:image/jpeg;base64,...'
  }));
};

ws.onmessage = (event) => {
  const result = JSON.parse(event.data);
  console.log(`Detected ${result.count} objects`);
};
```

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│   Web Browser / Client Application  │
└──────────────┬──────────────────────┘
               │
      ┌────────┴────────┬──────────┐
      │                 │          │
      ▼                 ▼          ▼
   HTTP/REST       WebSocket    CLI Client
      │                 │          │
      └────────┬────────┴──────────┘
               │
      ┌────────▼──────────────┐
      │   FastAPI Server      │
      │  (src/server.py)      │
      ├───────────────────────┤
      │ • Model Manager       │
      │ • Analytics Engine    │
      │ • Image Processing    │
      └────────┬──────────────┘
               │
      ┌────────▼──────────────┐
      │  YOLO11 Model         │
      │ (from MLflow/Local)   │
      └───────────────────────┘
```

## 📁 Project Structure

```
object-counting/
├── src/
│   ├── server.py              # Main FastAPI server (540+ lines)
│   ├── modeling.py            # Training script with MLflow
│   ├── wrapper.py             # YOLO model wrapper
│   ├── analytics.py           # Analytics engine
│   └── __init__.py
├── dataset/
│   └── Egg-1/                 # Training dataset
├── API_README.md              # API reference
├── DEPLOYMENT_GUIDE.md        # Deployment guide
├── requirements.txt           # Dependencies
├── Dockerfile                 # Docker image
├── docker-compose.yml         # Docker Compose
├── client.py                  # CLI client (310+ lines)
├── quickstart.py              # Setup script (180+ lines)
└── run_server.{sh,bat}        # Startup scripts
```

## ⚙️ Configuration

### Environment Variables (.env)
```env
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000

# Server Configuration (optional)
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
LOG_LEVEL=info
```

### Model Selection Order
1. Latest MLflow trained model
2. Local `yolo11m.pt` (fallback)

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)
```bash
docker-compose up
```

This starts:
- MLflow UI on http://localhost:5000
- API Server on http://localhost:8000

### Manual Docker Build
```bash
docker build -t object-counting:latest .
docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=http://localhost:5000 object-counting:latest
```

## 📊 Analytics Features

### Real-time Dashboard (in Playground)
- Total predictions counter
- Average inference time
- Total objects detected
- Server uptime
- Per-class breakdown

### Analytics Endpoints
```bash
# Summary statistics
curl http://localhost:8000/api/analytics/summary

# Response:
{
  "total_predictions": 42,
  "avg_inference_time": 0.1234,
  "total_objects_detected": 156,
  "uptime_seconds": 3600,
  "uptime": "1h 0m 0s",
  "predictions_per_minute": 0.7
}
```

## 🔧 Setup Instructions

### Prerequisites
- Python 3.8+
- CUDA 12.4+ (optional, for GPU acceleration)
- 4GB+ RAM

### Step-by-Step Setup

**1. Clone/Navigate to Project**
```bash
cd object-counting
```

**2. Create Virtual Environment** (Recommended)
```bash
conda create -n object-counting python=3.10
conda activate object-counting
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure Environment**
```bash
cp .env.example .env
```

**5. Run MLflow** (In separate terminal)
```bash
mlflow ui
```

**6. Start API Server**
```bash
python -m src.server
```

**7. Access Services**
- Playground: http://localhost:8000
- API Docs: http://localhost:8000/docs
- MLflow: http://localhost:5000

## 🎮 Using the Playground

### Upload Mode
1. Click upload area or drag-drop image
2. Click **Predict** button
3. View results with:
   - Annotated image with bounding boxes
   - Detection list with confidence scores
   - Total object count

### Webcam Mode
1. Click **📹 Webcam** tab
2. Click **Start Camera** (grant permissions if prompted)
3. Click **Capture & Predict** to analyze current frame
4. Results appear immediately

### Analytics Section
- **Model Info**: Current model name and version
- **Count Display**: Large number showing detected objects
- **Detections List**: Detailed list of each detection

## 🧪 Testing & Validation

### Test Connection
```bash
python client.py --test
```

### Test with Sample Image
```bash
python client.py --image dataset/Egg-1/test/images/sample.jpg
```

### Batch Processing
```bash
python client.py --batch dataset/Egg-1/test/images/
```

## 🚨 Troubleshooting

### Server Won't Start
```bash
# Check port availability
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Kill process on port 8000 if needed
kill -9 $(lsof -t -i:8000)  # Linux/Mac
```

### MLflow Not Found
```bash
# Start MLflow in separate terminal
mlflow ui

# Verify connection
curl http://localhost:5000
```

### GPU Not Detected
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Slow Inference
- Ensure GPU is available (`nvidia-smi`)
- Check inference times in analytics
- Consider using smaller input size
- Verify image format (JPG recommended)

## 📚 Additional Resources

- **FastAPI**: https://fastapi.tiangolo.com/
- **Ultralytics YOLO**: https://docs.ultralytics.com/
- **MLflow**: https://mlflow.org/docs/
- **Docker**: https://docs.docker.com/

## 🤝 Integration Examples

### Flask Integration
```python
from src.server import app as fastapi_app

# Mount FastAPI app in Flask
from flask import Flask
app = Flask(__name__)
# Integrate as needed
```

### Celery Task Queue
```python
from celery import shared_task
from client import ObjectCountingClient

@shared_task
def predict_batch(image_paths):
    client = ObjectCountingClient()
    return [client.predict_image(path) for path in image_paths]
```

### Database Logging
```python
# Add to server.py to log predictions to database
@app.post("/api/predict")
async def predict(file: UploadFile):
    # ... prediction code ...
    
    # Log to database
    db.predictions.insert_one({
        'timestamp': datetime.now(),
        'count': len(detections),
        'file_name': file.filename
    })
```

## 📈 Performance Metrics

Typical performance on NVIDIA GPU:
- **Inference Time**: 50-150ms per image
- **Throughput**: 6-20 predictions/second
- **Memory Usage**: 2-4GB GPU VRAM
- **Accuracy**: 90%+ on egg detection (depending on model)

## 📝 License

This project is provided for research and educational purposes.

## 🎉 Getting Started

Ready to begin? Follow these steps:

1. Run `python quickstart.py` for automated setup
2. Open http://localhost:8000 in your browser
3. Upload an image or use webcam for real-time detection
4. Check out API documentation at http://localhost:8000/docs

---

**Questions?** Check the [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed information.

**Need API examples?** See [API_README.md](API_README.md) for comprehensive REST API documentation.

**Want to extend it?** The code is well-organized and documented - create custom features easily!

🚀 **Ready to detect objects in real-time?** Start the server and open http://localhost:8000 now!
