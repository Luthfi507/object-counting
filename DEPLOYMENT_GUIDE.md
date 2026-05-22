# Complete Setup & Deployment Guide

## System Requirements

- Python 3.8+
- 4GB+ RAM
- NVIDIA GPU with CUDA support (recommended for faster inference)
- Docker (optional, for containerization)

## Installation

### Step 1: Clone & Navigate to Project

```bash
cd /path/to/object-counting
```

### Step 2: Create Virtual Environment (Optional but Recommended)

```bash
# Using conda
conda create -n object-counting python=3.10
conda activate object-counting

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment

```bash
cp .env.example .env
# Edit .env if needed
```

## Running the API Server

### Option 1: Direct Python Execution

```bash
# Activate your environment first
conda activate object-counting  # or source venv/bin/activate

# Start MLflow (in a separate terminal)
mlflow ui

# In another terminal, start the API
python -m src.server
```

### Option 2: Using Provided Scripts

**Linux/Mac:**
```bash
chmod +x run_server.sh
./run_server.sh
```

**Windows:**
```bash
run_server.bat
```

### Option 3: Docker Deployment

```bash
# Build the image
docker build -t object-counting:latest .

# Run with docker-compose (recommended)
docker-compose up

# Or run manually
docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=http://localhost:5000 object-counting:latest
```

## Accessing the Application

Once the server is running:

- **🎮 Interactive Playground**: http://localhost:8000
- **📊 API Documentation (Swagger)**: http://localhost:8000/docs
- **📖 API Documentation (ReDoc)**: http://localhost:8000/redoc
- **📈 MLflow UI**: http://localhost:5000

## Using the API

### Command Line Client

The project includes a Python client utility for testing and batch processing:

```bash
# Test connection
python client.py --test

# Get model info
python client.py --info

# Predict on single image
python client.py --image path/to/image.jpg

# Batch predict (all images in directory)
python client.py --batch path/to/images/

# Use custom server URL
python client.py --image image.jpg --url http://api.example.com
```

### REST API Examples

**Upload Image:**
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -F "file=@image.jpg"
```

**Get Model Info:**
```bash
curl "http://localhost:8000/api/model-info"
```

**Get Analytics Summary:**
```bash
curl "http://localhost:8000/api/analytics/summary"
```

**Get Class Statistics:**
```bash
curl "http://localhost:8000/api/analytics/classes"
```

**Get Performance Stats (last 5 minutes):**
```bash
curl "http://localhost:8000/api/analytics/performance?minutes=5"
```

**Get Prediction History:**
```bash
curl "http://localhost:8000/api/analytics/history?limit=50"
```

**Export Analytics:**
```bash
curl "http://localhost:8000/api/analytics/export" -o analytics.json
```

### Python Integration

```python
import requests
from pathlib import Path

# Predict on image
image_path = "path/to/image.jpg"
files = {'file': open(image_path, 'rb')}
response = requests.post("http://localhost:8000/api/predict", files=files)

result = response.json()
print(f"Objects detected: {result['count']}")
print(f"Inference time: {result['inference_time']}s")

for detection in result['detections']:
    print(f"  - {detection['class']} ({detection['confidence']:.1%})")

# Get analytics
analytics = requests.get("http://localhost:8000/api/analytics/summary").json()
print(f"Total predictions: {analytics['total_predictions']}")
```

### WebSocket Real-time Streaming

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws/stream');

ws.onopen = () => {
  // Send image as base64
  const imageBase64 = /* base64 image data */;
  ws.send(JSON.stringify({
    type: 'predict',
    image: imageBase64
  }));
};

ws.onmessage = (event) => {
  const result = JSON.parse(event.data);
  console.log(`Detected: ${result.count} objects`);
  console.log(`Inference time: ${result.inference_time}s`);
};
```

## Training & Model Management

### Training New Model

```bash
python -m src.modeling
```

The training script will:
1. Load training data from `dataset/Egg-1/`
2. Train YOLO11 model
3. Log metrics to MLflow
4. Save model artifacts

### Model Selection

The API automatically:
1. Checks MLflow for the latest trained model
2. Falls back to local `yolo11m.pt` if MLflow unavailable
3. Updates model info on startup

To verify model location:
```bash
curl "http://localhost:8000/api/model-info"
```

## Monitoring & Analytics

### Real-time Analytics

The playground includes a live analytics dashboard showing:
- Total predictions
- Average inference time
- Total objects detected
- Server uptime
- Predictions per minute

### Programmatic Analytics Access

```python
import requests

# Get summary
summary = requests.get("http://localhost:8000/api/analytics/summary").json()
print(f"Avg inference time: {summary['avg_inference_time']}s")

# Get class statistics
classes = requests.get("http://localhost:8000/api/analytics/classes").json()
for class_name, count in classes.items():
    print(f"{class_name}: {count} detected")

# Get performance stats
perf = requests.get("http://localhost:8000/api/analytics/performance?minutes=60").json()
print(f"Predictions/hour: {perf['predictions_per_second'] * 3600}")
```

## Performance Optimization

### GPU Usage

Check GPU availability:
```bash
nvidia-smi
```

Ensure GPU is being used in server logs. If not:
1. Install CUDA-enabled PyTorch
2. Verify CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`

### Inference Speed Tips

1. **Image Size**: Current model expects 224x224 - optimal for speed
2. **Batch Processing**: Send multiple images sequentially
3. **Connection**: Use WebSocket for streaming (lower latency)
4. **Compression**: JPEG compression for reduced bandwidth

### Scaling

For production deployment:

```bash
# Use Gunicorn for production
pip install gunicorn

# Run with multiple workers
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.server:app --bind 0.0.0.0:8000

# Or with Docker (production-ready)
docker run -p 8000:8000 --gpus all object-counting:latest
```

## Troubleshooting

### Port Already in Use

```bash
# Find and kill process on port 8000
lsof -ti:8000 | xargs kill -9  # Linux/Mac
netstat -ano | findstr :8000   # Windows
```

### MLflow Connection Failed

```bash
# Start MLflow in separate terminal
mlflow ui --host 0.0.0.0 --port 5000

# Update MLFLOW_TRACKING_URI if different
export MLFLOW_TRACKING_URI=http://localhost:5000
```

### CUDA/GPU Issues

```bash
# Verify GPU setup
python -c "import torch; print(torch.cuda.is_available())"

# Check NVIDIA drivers
nvidia-smi

# Reinstall PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Memory Issues

If running out of memory:
1. Reduce batch size (built-in, default=8)
2. Use smaller image size (current=224)
3. Run on CPU: set `CUDA_VISIBLE_DEVICES=""`
4. Add swap space on system

### Model Not Loading

Check MLflow artifacts:
```bash
# List all runs
mlflow runs list --experiment-name "Object-Counting-Experiment"

# Check run artifacts
mlflow artifacts list --run-id <run_id>
```

## Project Structure

```
object-counting/
├── src/
│   ├── server.py          # Main FastAPI server
│   ├── modeling.py        # Training script
│   ├── wrapper.py         # YOLO wrapper for MLflow
│   ├── analytics.py       # Analytics tracking
│   └── __init__.py
├── dataset/
│   └── Egg-1/
│       ├── data.yaml
│       ├── train/
│       ├── val/
│       └── test/
├── runs/                  # Training outputs
├── mlruns/                # MLflow artifacts
├── requirements.txt       # Python dependencies
├── .env.example           # Environment template
├── run_server.sh          # Linux/Mac startup script
├── run_server.bat         # Windows startup script
├── Dockerfile             # Docker image
├── docker-compose.yml     # Docker Compose config
├── client.py              # CLI client
└── API_README.md          # API documentation
```

## Next Steps

1. **Training**: Run `python -m src.modeling` to train a model
2. **Testing**: Use the playground to test predictions
3. **Integration**: Integrate API into your application
4. **Deployment**: Deploy using Docker to production environment
5. **Monitoring**: Track analytics and performance metrics

## Support

For more information:
- FastAPI docs: https://fastapi.tiangolo.com/
- YOLO docs: https://docs.ultralytics.com/
- MLflow docs: https://mlflow.org/docs/
- Docker docs: https://docs.docker.com/

---

**Ready to get started?** Follow the Installation section and run the server!
