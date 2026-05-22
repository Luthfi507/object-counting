# 🚀 Quick Reference Guide

## Startup (Choose One)

### 1️⃣ Fastest Way
```bash
python -m src.server
# Then open: http://localhost:8000
```

### 2️⃣ With Setup Wizard
```bash
python quickstart.py
# Follow prompts, then start server
```

### 3️⃣ With Docker
```bash
docker-compose up
# MLflow: http://localhost:5000
# API: http://localhost:8000
```

---

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | Playground UI |
| POST | `/api/predict` | Upload image prediction |
| GET | `/api/model-info` | Model information |
| GET | `/api/analytics/summary` | Statistics summary |
| GET | `/api/analytics/classes` | Per-class counts |
| GET | `/api/analytics/performance?minutes=5` | Performance stats |
| GET | `/api/analytics/history?limit=100` | Prediction history |
| GET | `/api/analytics/export` | Download analytics |
| WS | `/ws/stream` | Real-time streaming |

---

## CLI Commands

```bash
# Test connection
python client.py --test

# Get model info
python client.py --info

# Predict single image
python client.py --image path/to/image.jpg

# Batch predict
python client.py --batch path/to/images/

# Custom server URL
python client.py --image image.jpg --url http://api.example.com
```

---

## cURL Examples

```bash
# Predict
curl -X POST "http://localhost:8000/api/predict" \
  -F "file=@image.jpg" | jq

# Get stats
curl "http://localhost:8000/api/analytics/summary" | jq

# Export analytics
curl "http://localhost:8000/api/analytics/export" -o analytics.json

# Model info
curl "http://localhost:8000/api/model-info" | jq
```

---

## Python Integration

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/api/predict",
    files={'file': open('image.jpg', 'rb')}
)
result = response.json()
print(f"Detected: {result['count']} objects")

# Get analytics
analytics = requests.get(
    "http://localhost:8000/api/analytics/summary"
).json()
print(f"Total predictions: {analytics['total_predictions']}")

# Batch processing
import glob
for image in glob.glob("images/*.jpg"):
    response = requests.post(
        "http://localhost:8000/api/predict",
        files={'file': open(image, 'rb')}
    )
    print(f"{image}: {response.json()['count']} objects")
```

---

## WebSocket Example (JavaScript)

```javascript
// Connect
const ws = new WebSocket('ws://localhost:8000/ws/stream');

// On connect
ws.onopen = () => {
  console.log('Connected');
  
  // Send image
  const imageBase64 = 'data:image/jpeg;base64,...'; // your image
  ws.send(JSON.stringify({
    type: 'predict',
    image: imageBase64
  }));
};

// Handle response
ws.onmessage = (event) => {
  const result = JSON.parse(event.data);
  console.log(`Detected: ${result.count} objects`);
  console.log(`Time: ${result.inference_time}s`);
};

// Handle errors
ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};
```

---

## Web Playground Features

| Feature | How To |
|---------|--------|
| **Upload Image** | Click upload area or drag-drop |
| **Webcam** | Click "📹 Webcam" tab → "Start Camera" |
| **Capture** | Click "Capture & Predict" for current frame |
| **Clear** | Click "Clear" button to reset |
| **View Results** | See predictions on right panel |
| **View Analytics** | Check stats in dashboard |

---

## Files Location

| File | Purpose |
|------|---------|
| `src/server.py` | Main API server |
| `src/analytics.py` | Analytics engine |
| `client.py` | CLI client |
| `quickstart.py` | Setup wizard |
| `Dockerfile` | Docker image |
| `docker-compose.yml` | Docker orchestration |

---

## Configuration

### .env File
```env
MLFLOW_TRACKING_URI=http://localhost:5000
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
LOG_LEVEL=info
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 8000 in use | `lsof -ti:8000 \| xargs kill -9` |
| MLflow not running | Open new terminal: `mlflow ui` |
| GPU not available | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| Import errors | `pip install -r requirements.txt` |
| Slow predictions | Check GPU: `nvidia-smi` |

---

## Documentation

- **Full Guide**: [README_API.md](README_API.md)
- **API Docs**: [API_README.md](API_README.md)
- **Deployment**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **Architecture**: [architecture.json](architecture.json)

---

## Performance Tips

1. **Use GPU** - Check with `nvidia-smi`
2. **Batch Processing** - Send multiple images sequentially
3. **WebSocket** - Lower latency for real-time
4. **Image Format** - Use JPEG for better compression
5. **Size** - 224x224 is optimal for this model

---

## Common Workflows

### 📤 Upload & Predict
1. Open http://localhost:8000
2. Upload image
3. Click Predict
4. View results

### 📹 Real-time Webcam
1. Open http://localhost:8000
2. Click "📹 Webcam"
3. Click "Start Camera"
4. Click "Capture & Predict"

### 🔄 Batch Processing
```bash
python client.py --batch ./dataset/images/
```

### 📊 Get Statistics
```bash
curl http://localhost:8000/api/analytics/summary | jq
```

### 🐳 Docker Deployment
```bash
docker-compose up
```

---

## Support

- 📖 API Docs: http://localhost:8000/docs
- 🔍 ReDoc: http://localhost:8000/redoc
- 📊 MLflow: http://localhost:5000
- 🎮 Playground: http://localhost:8000

---

## Version Info

- **API**: v1.0.0
- **YOLO**: 11 (YOLO11m)
- **FastAPI**: Latest
- **Python**: 3.8+

---

**🎯 Ready to start? Run `python -m src.server` and open http://localhost:8000!**
