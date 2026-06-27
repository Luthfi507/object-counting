from roboflow import Roboflow
import os
from dotenv import load_dotenv
load_dotenv()

rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace("roboflow-100").project("vehicles-q0x2v")
version = project.version(2)
dataset = version.download("yolov8")
location = dataset.location
print(f"Dataset downloaded to: {location}")