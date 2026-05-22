const selectedState = {
    selectedFile: null,
    ws: null,
    webcamActive: false,
    videoStream: null,
    canvas: null,
    ctx: null,
};

window.addEventListener('DOMContentLoaded', () => {
    initializeUploadArea();
    loadModelInfo();
    initWebSocket();
    setupCanvasForWebcam();
    attachActionHandlers();
});

function initializeUploadArea() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');

    uploadArea.addEventListener('click', () => fileInput.click());

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            selectedState.selectedFile = files[0];
            previewFile(selectedState.selectedFile);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            selectedState.selectedFile = e.target.files[0];
            previewFile(selectedState.selectedFile);
        }
    });
}

function attachActionHandlers() {
    document.querySelectorAll('.mode-btn').forEach((btn) => {
        btn.addEventListener('click', () => switchMode(btn.dataset.mode, btn));
    });

    document.getElementById('clearBtn').addEventListener('click', clearUpload);
    document.getElementById('predictBtn').addEventListener('click', predictImage);
    document.getElementById('startCameraBtn').addEventListener('click', startWebcam);
    document.getElementById('stopCameraBtn').addEventListener('click', stopWebcam);
    document.getElementById('captureBtn').addEventListener('click', captureFrame);
    document.getElementById('cameraDebugBtn').addEventListener('click', testCamera);
}

function previewFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        const preview = document.getElementById('preview');
        preview.src = e.target.result;
        preview.classList.add('active');
        document.getElementById('predictBtn').disabled = false;

        updateDebugStatus('Image loaded and ready for prediction.', 'info');
    };
    reader.readAsDataURL(file);
}

function clearUpload() {
    selectedState.selectedFile = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('preview').classList.remove('active');
    document.getElementById('predictBtn').disabled = true;
    hideStatus();
    updateDebugStatus('Upload cleared. Select a new image to continue.', 'info');
}

async function predictImage() {
    if (!selectedState.selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedState.selectedFile);

    showLoading(true);
    hideStatus();

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) throw new Error('Prediction failed');

        const data = await response.json();
        displayResults(data);
        showStatus('Prediction successful!', 'success');
        updateDebugStatus('Image uploaded and prediction returned successfully.', 'info');
    } catch (error) {
        console.error('Error:', error);
        showStatus('Error: ' + error.message, 'error');
        updateDebugStatus('Upload prediction failed. ' + error.message, 'error');
    } finally {
        showLoading(false);
    }
}

function displayResults(data) {
    document.getElementById('preview').src = 'data:image/jpeg;base64,' + data.image;
    document.getElementById('preview').classList.add('active');
    document.getElementById('countDisplay').textContent = data.count;

    const detectionsList = document.getElementById('detectionsList');
    if (data.detections.length === 0) {
        detectionsList.innerHTML = '<p style="color: #999; text-align: center; padding: 20px;">No objects detected</p>';
    } else {
        detectionsList.innerHTML = data.detections
            .map(
                (det) => `
            <div class="detection-item">
                <div class="detection-class">${det.class}</div>
                <div class="detection-confidence">Confidence: ${(det.confidence * 100).toFixed(1)}%</div>
                <div class="detection-bbox">Box: [${Math.round(det.bbox.x1)}, ${Math.round(det.bbox.y1)}, ${Math.round(det.bbox.x2)}, ${Math.round(det.bbox.y2)}]</div>
            </div>`
            )
            .join('');
    }
}

function showLoading(show) {
    document.getElementById('loading').classList.toggle('active', show);
}

function showStatus(message, type) {
    const status = document.getElementById('status');
    status.textContent = message;
    status.className = `status active ${type}`;
}

function hideStatus() {
    const status = document.getElementById('status');
    status.className = 'status';
}

function updateDebugStatus(message, type = 'info') {
    const debug = document.getElementById('cameraDebugStatus');
    debug.textContent = message;
    debug.className = `debug-status active ${type}`;
}

async function loadModelInfo() {
    try {
        const response = await fetch('/api/model-info');
        const data = await response.json();

        document.getElementById('modelName').textContent = data.model_name;
        document.getElementById('modelVersion').textContent = data.model_version;
        document.getElementById('modelStatus').textContent = '✅ Ready';

        updateDebugStatus('Model loaded: ' + data.model_name, 'info');
    } catch (error) {
        console.error('Error loading model info:', error);
        updateDebugStatus('Unable to load model info: ' + error.message, 'error');
    }
}

function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    selectedState.ws = new WebSocket(`${protocol}//${window.location.host}/ws/stream`);

    selectedState.ws.onopen = () => {
        console.log('WebSocket connected');
        updateDebugStatus('WebSocket connected successfully.', 'info');
        setInterval(() => {
            if (selectedState.ws && selectedState.ws.readyState === WebSocket.OPEN) {
                selectedState.ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);
    };

    selectedState.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'prediction') {
            displayResults(data);
            showStatus('Real-time prediction received', 'success');
        }
    };

    selectedState.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateDebugStatus('WebSocket error. Check server connection.', 'error');
    };
}

function switchMode(mode, button) {
    document.querySelectorAll('.mode-btn').forEach((btn) => btn.classList.remove('active'));
    button.classList.add('active');

    const uploadMode = document.getElementById('uploadMode');
    const webcamMode = document.getElementById('webcamMode');

    if (mode === 'upload') {
        uploadMode.style.display = 'block';
        webcamMode.style.display = 'none';
        stopWebcam();
    } else {
        uploadMode.style.display = 'none';
        webcamMode.style.display = 'block';
        updateDebugStatus('Webcam mode active. Test camera to confirm.', 'info');
    }
}

function setupCanvasForWebcam() {
    selectedState.canvas = document.createElement('canvas');
    selectedState.ctx = selectedState.canvas.getContext('2d');
}

async function startWebcam() {
    try {
        selectedState.videoStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' },
        });

        const video = document.getElementById('webcam');
        video.srcObject = selectedState.videoStream;
        document.getElementById('webcamContainer').classList.add('active');
        selectedState.webcamActive = true;

        updateDebugStatus('Camera started successfully.', 'info');
    } catch (error) {
        updateDebugStatus('Camera access failed: ' + error.message, 'error');
    }
}

function stopWebcam() {
    if (selectedState.videoStream) {
        selectedState.videoStream.getTracks().forEach((track) => track.stop());
        selectedState.webcamActive = false;
        document.getElementById('webcamContainer').classList.remove('active');
        updateDebugStatus('Camera stopped.', 'info');
    }
}

function captureFrame() {
    if (!selectedState.webcamActive) {
        updateDebugStatus('Camera is not active. Start the camera first.', 'error');
        return;
    }

    const video = document.getElementById('webcam');
    selectedState.canvas.width = video.videoWidth;
    selectedState.canvas.height = video.videoHeight;
    selectedState.ctx.drawImage(video, 0, 0);

    selectedState.canvas.toBlob((blob) => {
        const formData = new FormData();
        formData.append('file', blob, 'webcam.jpg');

        showLoading(true);
        hideStatus();

        fetch('/api/predict', {
            method: 'POST',
            body: formData,
        })
            .then((response) => {
                if (!response.ok) throw new Error('Prediction failed');
                return response.json();
            })
            .then((data) => {
                displayResults(data);
                showStatus('Capture successful!', 'success');
                updateDebugStatus('Webcam frame captured and predicted.', 'info');
            })
            .catch((error) => {
                showStatus('Error: ' + error.message, 'error');
                updateDebugStatus('Webcam prediction failed: ' + error.message, 'error');
            })
            .finally(() => showLoading(false));
    }, 'image/jpeg');
}

async function testCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        updateDebugStatus('Camera API not supported in this browser.', 'error');
        return;
    }

    updateDebugStatus('Testing camera permission and availability...', 'info');

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        stream.getTracks().forEach((track) => track.stop());
        updateDebugStatus('Camera is available and permission granted.', 'success');
    } catch (error) {
        updateDebugStatus('Camera test failed: ' + error.message, 'error');
    }
}
