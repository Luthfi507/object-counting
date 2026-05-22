#!/usr/bin/env python
"""
Quick start setup script for Object Counting API

This script will:
1. Check dependencies
2. Set up environment
3. Start MLflow (if available)
4. Start the API server
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, check=True, shell=False):
    """Run a shell command"""
    try:
        result = subprocess.run(cmd if shell else cmd.split(), check=check, shell=shell)
        return result.returncode == 0
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def check_dependencies():
    """Check if all dependencies are installed"""
    print("📋 Checking dependencies...")
    
    required = ['fastapi', 'uvicorn', 'ultralytics', 'mlflow', 'torch', 'torchvision']
    
    try:
        import pkg_resources
        installed = {pkg.key for pkg in pkg_resources.working_set}
        missing = [pkg for pkg in required if pkg.lower() not in installed]
        
        if missing:
            print(f"❌ Missing packages: {', '.join(missing)}")
            print("Run: pip install -r requirements.txt")
            return False
        else:
            print("✅ All dependencies installed")
            return True
    except:
        print("⚠️  Could not check dependencies, proceeding...")
        return True

def setup_environment():
    """Set up environment variables"""
    print("\n🔧 Setting up environment...")
    
    env_file = Path(".env")
    if not env_file.exists():
        example_file = Path(".env.example")
        if example_file.exists():
            with open(example_file) as f:
                content = f.read()
            with open(env_file, 'w') as f:
                f.write(content)
            print("✅ Created .env from .env.example")
        else:
            print("⚠️  No .env file found")
    else:
        print("✅ .env file exists")

def check_mlflow():
    """Check if MLflow is running"""
    print("\n📊 Checking MLflow...")
    
    try:
        import requests
        response = requests.get("http://localhost:5000", timeout=2)
        print("✅ MLflow is running at http://localhost:5000")
        return True
    except:
        print("⚠️  MLflow not running")
        print("   To start MLflow in another terminal, run: mlflow ui")
        return False

def start_mlflow():
    """Ask user if they want to start MLflow"""
    if sys.platform == 'win32':
        print("\n📊 MLflow not detected. To start:")
        print("   1. Open another terminal/command prompt")
        print("   2. Run: mlflow ui")
        print("   3. MLflow will be available at http://localhost:5000")
    else:
        response = input("\n📊 Start MLflow in background? (y/n): ").lower()
        if response == 'y':
            # Start MLflow in background
            subprocess.Popen(['mlflow', 'ui'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("✅ MLflow started in background")
            import time
            time.sleep(2)
            return True
    return False

def check_gpu():
    """Check if GPU is available"""
    print("\n🎮 Checking GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
        else:
            print("⚠️  No GPU detected - will use CPU (slower)")
        return torch.cuda.is_available()
    except:
        print("⚠️  Could not check GPU")
        return False

def main():
    """Main setup function"""
    print("=" * 60)
    print("🚀 Object Counting API - Quick Start Setup")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first:")
        print("   pip install -r requirements.txt")
        return False
    
    # Setup environment
    setup_environment()
    
    # Check GPU
    check_gpu()
    
    # Check/start MLflow
    if not check_mlflow():
        start_mlflow()
    
    # Final instructions
    print("\n" + "=" * 60)
    print("✅ Setup complete!")
    print("=" * 60)
    print("\n🎯 Next steps:")
    print("   1. Start the server:")
    print("      python -m src.server")
    print("\n   2. Open in browser:")
    print("      http://localhost:8000")
    print("\n   3. API Documentation:")
    print("      http://localhost:8000/docs")
    print("\n   4. MLflow UI:")
    print("      http://localhost:5000")
    print("\n" + "=" * 60)
    print("💡 Quick commands:")
    print("   • Test API: python client.py --test")
    print("   • Predict:  python client.py --image image.jpg")
    print("   • Batch:    python client.py --batch ./images/")
    print("=" * 60 + "\n")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
