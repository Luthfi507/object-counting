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
    """Setup matplotlib figure."""
    global fig, img_display
    
    cap = cv2.VideoCapture(idx)
    ret, frame = cap.read()
    
    if not ret:
        raise RuntimeError(f"Cannot read from video source: {idx}")
    
    cap.release()
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rcParams["toolbar"] = "None"
    plt.ion()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")

    img_display = ax.imshow(rgb_frame)
    plt.show(block=False)

async def stream(idx):
    global img_display
    
    setup(idx)
    track_history = defaultdict(list)
    frame_count = 0
    
    try:
        async with websockets.connect("ws://localhost:8002/track") as ws:
            logger.info("✓ WebSocket connected")
            
            cap = cv2.VideoCapture(idx)
            logger.info(f"✓ Video source opened: {idx}")
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        logger.info(f"✓ Video finished (total frames: {frame_count})")
                        break
                    
                    frame_count += 1
                    
                    try:
                        # ─── Send frame ──────────────────────────────────────────
                        _, buffer = cv2.imencode(".jpg", frame)
                        await ws.send(buffer.tobytes())
                        logger.debug(f"[{frame_count}] Frame sent")
                        
                        # ─── Receive response ────────────────────────────────────
                        try:
                            result = await asyncio.wait_for(ws.recv(), timeout=10.0)
                        except asyncio.TimeoutError:
                            logger.error(f"[{frame_count}] Server timeout (10s)")
                            break
                        
                        # ─── Parse JSON response ────────────────────────────────
                        try:
                            res = json.loads(result)
                        except json.JSONDecodeError as e:
                            logger.error(f"[{frame_count}] JSON decode error: {e}")
                            logger.debug(f"Response: {result[:200]}")
                            break
                        
                        detections = res.get('detection', [])
                        annotated_frame_b64 = res.get('annotated_frame')
                        
                        logger.debug(f"[{frame_count}] Received {len(detections)} detections")
                        
                        if not annotated_frame_b64:
                            logger.warning(f"[{frame_count}] No annotated_frame in response")
                            annotated_frame = frame.copy()
                        else:
                            # ─── Decode frame ───────────────────────────────────
                            try:
                                frame_bytes = base64.b64decode(annotated_frame_b64)
                                np_arr = np.frombuffer(frame_bytes, np.uint8)
                                annotated_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                                
                                if annotated_frame is None:
                                    logger.warning(f"[{frame_count}] Failed to decode frame")
                                    annotated_frame = frame.copy()
                            except Exception as e:
                                logger.error(f"[{frame_count}] Frame decode error: {e}")
                                annotated_frame = frame.copy()
                        
                        # ─── Update track history ───────────────────────────────
                        for det in detections:
                            track_id = det['track_id']
                            bbox = det['bbox']
                            track_history[track_id].append(bbox)
                        
                        # ─── Add count text ──────────────────────────────────────
                        count_text = f"Count: {len(track_history.keys())}"
                        cv2.putText(
                            annotated_frame,
                            count_text,
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )
                        
                        # ─── Display ─────────────────────────────────────────────
                        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        img_display.set_data(rgb_frame)
                        fig.canvas.draw_idle()
                        fig.canvas.flush_events()
                        
                    except Exception as e:
                        logger.exception(f"[{frame_count}] Error in frame processing: {e}")
                        break
            
            finally:
                cap.release()
                logger.info("✓ Video capture released")
    
    except ConnectionRefusedError:
        logger.error("✗ Connection refused - WebSocket server not running")
    except websockets.exceptions.ConnectionClosed as e:
        logger.error(f"✗ WebSocket connection closed: {e}")
    except Exception as e:
        logger.exception(f"✗ Fatal error: {e}")

if __name__ == "__main__":
    video_path = 'test.mp4'
    asyncio.run(stream(video_path))