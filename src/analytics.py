"""
Analytics and monitoring module for Object Counting API

Tracks:
- Prediction statistics
- Performance metrics
- Model accuracy
- System resource usage
"""

import time
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List
import json

@dataclass
class PredictionMetric:
    timestamp: datetime
    inference_time: float  # seconds
    object_count: int
    image_size: tuple  # (width, height)
    model_name: str
    
    def to_dict(self):
        return {
            'timestamp': self.timestamp.isoformat(),
            'inference_time': round(self.inference_time, 4),
            'object_count': self.object_count,
            'image_size': self.image_size,
            'model_name': self.model_name
        }

class Analytics:
    """Tracks analytics for the API"""
    
    def __init__(self, max_history=1000):
        self.max_history = max_history
        self.metrics = deque(maxlen=max_history)
        self.class_counts = defaultdict(int)  # Per-class object counts
        self.start_time = datetime.now()
    
    def add_prediction(self, metric: PredictionMetric):
        """Add a prediction metric"""
        self.metrics.append(metric)
    
    def add_detection(self, class_name: str, count: int = 1):
        """Track detected objects by class"""
        self.class_counts[class_name] += count
    
    def get_summary(self) -> Dict:
        """Get overall analytics summary"""
        if not self.metrics:
            return {
                'total_predictions': 0,
                'avg_inference_time': 0,
                'total_objects_detected': 0,
                'uptime': 0
            }
        
        total_predictions = len(self.metrics)
        avg_inference_time = sum(m.inference_time for m in self.metrics) / total_predictions
        total_objects = sum(m.object_count for m in self.metrics)
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'total_predictions': total_predictions,
            'avg_inference_time': round(avg_inference_time, 4),
            'total_objects_detected': total_objects,
            'uptime_seconds': uptime,
            'uptime': self._format_uptime(uptime),
            'predictions_per_minute': round(total_predictions / (uptime / 60)) if uptime > 0 else 0
        }
    
    def get_class_statistics(self) -> Dict:
        """Get statistics per object class"""
        return dict(self.class_counts)
    
    def get_performance_stats(self, minutes: int = 5) -> Dict:
        """Get performance statistics for the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_metrics = [m for m in self.metrics if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {
                'period_minutes': minutes,
                'predictions': 0,
                'avg_inference_time': 0,
                'min_inference_time': 0,
                'max_inference_time': 0
            }
        
        inference_times = [m.inference_time for m in recent_metrics]
        
        return {
            'period_minutes': minutes,
            'predictions': len(recent_metrics),
            'avg_inference_time': round(sum(inference_times) / len(inference_times), 4),
            'min_inference_time': round(min(inference_times), 4),
            'max_inference_time': round(max(inference_times), 4),
            'predictions_per_second': round(len(recent_metrics) / (minutes * 60), 2)
        }
    
    def get_history(self, limit: int = 100) -> List[Dict]:
        """Get recent prediction history"""
        recent = list(self.metrics)[-limit:]
        return [m.to_dict() for m in reversed(recent)]
    
    @staticmethod
    def _format_uptime(seconds: float) -> str:
        """Format uptime in human-readable format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def export_json(self, filepath: str):
        """Export analytics to JSON file"""
        data = {
            'summary': self.get_summary(),
            'class_statistics': self.get_class_statistics(),
            'performance_stats_5min': self.get_performance_stats(5),
            'performance_stats_1hour': self.get_performance_stats(60),
            'recent_history': self.get_history(100),
            'exported_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def reset(self):
        """Reset analytics"""
        self.metrics.clear()
        self.class_counts.clear()
        self.start_time = datetime.now()

# Global analytics instance
analytics = Analytics()
