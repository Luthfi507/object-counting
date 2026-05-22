# Object Counting API & Playground

Real-time object detection and counting API with an interactive web-based playground, powered by YOLO11 and MLflow.

## Features

✨ **Interactive Playground UI**
- 📤 Image upload with drag-and-drop support
- 📹 Real-time webcam capture
- 🎯 Live object detection and visualization
- 📊 Real-time results display

🔌 **REST API**
- `/api/predict` - Upload image for prediction
- `/api/model-info` - Get current model information
- `/ws/stream` - WebSocket for real-time streaming

🤖 **MLflow Integration**
- Automatically loads the latest trained model
- Fallback to local model if MLflow is unavailable
- Model version tracking

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your MLflow tracking URI (if needed)
# Default: http://localhost:5000
```

### 3. Start MLflow (if using local tracking)

```bash
mlflow ui
```

This will start MLflow UI at `http://localhost:5000`

### 4. Run the Server

**On Linux/Mac:**
```bash
chmod +x run_server.sh
./run_server.sh
```

**On Windows:**
```bash
run_server.bat
```

**Or directly:**
```bash
python -m src.server
```

The server will start at `http://localhost:8000`

## Usage

### 🎮 Interactive Playground

Open your browser and go to:
```
http://localhost:8000
```

**Features:**
- **Upload Mode**: Click or drag-drop images for instant predictions
- **Webcam Mode**: Real-time predictions from your webcam
- View detections with confidence scores and bounding boxes
- Real-time count display

### 🔗 REST API Examples

**Upload Image for Prediction:**
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -F "file=@path/to/image.jpg"
```

**Get Model Information:**
```bash
curl "http://localhost:8000/api/model-info"
```

**Response Example:**
```json
{
  "success": true,
  "image": "base64_encoded_image_with_boxes",
  "detections": [
    {
      "id": 0,
      "class": "egg",
      "confidence": 0.95,
      "bbox": {
        "x1": 100,
        "y1": 150,
        "x2": 300,
        "y2": 400
      }
    }
  ],
  "count": 1,
  "timestamp": "2024-05-22T10:30:45.123456"
}
```

### 📡 WebSocket API

Connect to the WebSocket for real-time predictions:

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
  const data = JSON.parse(event.data);
  console.log('Detections:', data.detections);
};
```

### 📚 API Documentation

Interactive Swagger/OpenAPI documentation available at:
```
http://localhost:8000/docs
```

ReDoc documentation available at:
```
http://localhost:8000/redoc
```

## Configuration

### Environment Variables

Edit `.env` file:

```env
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000

# Server Configuration (optional, used for startup script)
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
LOG_LEVEL=info
```

### Model Selection

The server automatically selects the model in this order:

1. **Latest MLflow Model** - Most recent trained model from the experiment
2. **Local Model** - Falls back to `yolo11m.pt` if MLflow is unavailable

## Architecture

```
┌─────────────────────────────────────────┐
│     Interactive Web Playground          │
│  (Upload/Webcam + Real-time Results)    │
└──────────┬──────────────────────────────┘
           │
           │ HTTP/WebSocket
           ▼
┌─────────────────────────────────────────┐
│         FastAPI Server                  │
├─────────────────────────────────────────┤
│  • Model Manager (MLflow + Local)      │
│  • Prediction Pipeline                  │
│  • Image Processing & Visualization     │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│    YOLO11 Model                         │
│   (from MLflow or Local)                │
└─────────────────────────────────────────┘
```

## Performance Tips

1. **GPU Acceleration**: Make sure CUDA is available for faster inference
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Batch Processing**: For multiple images, send them sequentially
   
3. **Image Size**: Optimal input size is 224x224 (as configured in training)

4. **Confidence Threshold**: Default is 0.25, adjust in predictions if needed

## Troubleshooting

### Model not loading from MLflow

Check that MLflow tracking URI is correct:
```bash
# View MLflow UI
mlflow ui
```

If using local MLflow:
1. Ensure MLflow is running on `http://localhost:5000`
2. Check that training has produced MLflow artifacts

### Webcam not working

1. Ensure browser has camera permissions
2. Test with `http://` (not `https://` on localhost)
3. Check if camera is already in use by another application

### Slow predictions

1. Verify GPU is being used: `nvidia-smi`
2. Reduce image size or batch size
3. Check server logs for processing time

### WebSocket connection failed

1. Ensure server is running on correct port
2. Check firewall settings
3. Verify WebSocket support in your network

## Example Files

- `src/modeling.py` - Training script with MLflow integration
- `src/wrapper.py` - YOLO model wrapper for MLflow
- `src/server.py` - FastAPI server with playground
- `.env.example` - Configuration template

## Requirements

- Python 3.8+
- PyTorch with CUDA support (optional but recommended)
- FastAPI
- Ultralytics YOLO11
- MLflow
- See `requirements.txt` for full list

## License

This project is provided as-is for research and educational purposes.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review FastAPI docs at `/docs`
3. Check server logs for error details

---

**Ready to start?** Run `./run_server.sh` or `run_server.bat` and open `http://localhost:8000` in your browser! 🚀
