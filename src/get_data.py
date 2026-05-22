import os
from roboflow import Roboflow
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# Get data from Roboflow
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace("sinica-dz08y").project("egg-4u31w")
version = project.version(1)
dataset = version.download("yolov11")
logger.success(f"Data downloaded successfully in {dataset}")