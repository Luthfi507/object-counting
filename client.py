#!/usr/bin/env python
"""
Client utility to test the Object Counting API

Usage:
    python client.py --image path/to/image.jpg
    python client.py --webcam
    python client.py --url http://localhost:8000
"""

import argparse
import base64
import json
from pathlib import Path
import requests
from PIL import Image
import io

class ObjectCountingClient:
    def __init__(self, url="http://localhost:8000"):
        self.url = url.rstrip('/')
        
    def predict_image(self, image_path):
        """Predict on a single image"""
        print(f"🔍 Predicting on {image_path}...")
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{self.url}/api/predict", files=files)
        
        if response.status_code == 200:
            result = response.json()
            self._display_results(result)
            return result
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return None
    
    def get_model_info(self):
        """Get current model information"""
        print("📊 Fetching model information...")
        
        response = requests.get(f"{self.url}/api/model-info")
        
        if response.status_code == 200:
            info = response.json()
            print("\n" + "="*50)
            print("MODEL INFORMATION")
            print("="*50)
            for key, value in info.items():
                print(f"  {key}: {value}")
            print("="*50 + "\n")
            return info
        else:
            print(f"❌ Error: {response.status_code}")
            return None
    
    def batch_predict(self, image_dir):
        """Predict on all images in a directory"""
        image_dir = Path(image_dir)
        results = []
        
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        print(f"🔍 Found {len(image_files)} images to process")
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing {image_path.name}...")
            result = self.predict_image(str(image_path))
            if result:
                results.append({
                    'image': image_path.name,
                    'count': result.get('count', 0),
                    'detections': result.get('detections', [])
                })
        
        return results
    
    def _display_results(self, result):
        """Display prediction results nicely"""
        print("\n" + "="*50)
        print("PREDICTION RESULTS")
        print("="*50)
        print(f"✅ Objects detected: {result.get('count', 0)}")
        
        if result.get('detections'):
            print("\nDetections:")
            for det in result['detections']:
                print(f"  • {det['class']} (confidence: {det['confidence']:.1%})")
                bbox = det['bbox']
                print(f"    Box: [{bbox['x1']:.0f}, {bbox['y1']:.0f}, {bbox['x2']:.0f}, {bbox['y2']:.0f}]")
        else:
            print("No objects detected")
        
        print(f"Timestamp: {result.get('timestamp', 'N/A')}")
        print("="*50 + "\n")
    
    def test_connection(self):
        """Test connection to the API"""
        print(f"🔗 Testing connection to {self.url}...")
        
        try:
            response = requests.get(f"{self.url}/api/model-info", timeout=5)
            if response.status_code == 200:
                print("✅ Connection successful!")
                return True
            else:
                print(f"❌ Server returned status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"❌ Connection failed. Is the server running?")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(
        description="Client utility for Object Counting API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python client.py --test                          # Test connection
  python client.py --info                          # Get model info
  python client.py --image path/to/image.jpg       # Predict on single image
  python client.py --batch path/to/images/         # Batch predict
  python client.py --url http://api.example.com    # Use custom server
        """
    )
    
    parser.add_argument('--url', default='http://localhost:8000', 
                        help='API server URL (default: http://localhost:8000)')
    parser.add_argument('--test', action='store_true', 
                        help='Test connection to API')
    parser.add_argument('--info', action='store_true', 
                        help='Get model information')
    parser.add_argument('--image', type=str, 
                        help='Path to image for prediction')
    parser.add_argument('--batch', type=str, 
                        help='Directory of images for batch prediction')
    
    args = parser.parse_args()
    
    client = ObjectCountingClient(args.url)
    
    if args.test:
        client.test_connection()
    elif args.info:
        client.get_model_info()
    elif args.image:
        if not Path(args.image).exists():
            print(f"❌ Image file not found: {args.image}")
            return
        client.predict_image(args.image)
    elif args.batch:
        if not Path(args.batch).is_dir():
            print(f"❌ Directory not found: {args.batch}")
            return
        results = client.batch_predict(args.batch)
        
        # Summary
        total_count = sum(r['count'] for r in results)
        print("\n" + "="*50)
        print("BATCH SUMMARY")
        print("="*50)
        print(f"Total images processed: {len(results)}")
        print(f"Total objects detected: {total_count}")
        print("="*50)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
