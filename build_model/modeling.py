from ultralytics import YOLO
import os
import mlflow

project_dir = os.path.dirname(__file__)
data_dir = os.path.join(project_dir, '..', 'dataset', 'Egg-1')
model = YOLO('yolo11m.pt') 

mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("Object-Counting-Experiment")

# Define training parameters
data = os.path.join(data_dir, 'data.yaml')
epochs = 10
imgsz = 640
batch = 8

def train():
    with mlflow.start_run(run_name='egg-counting'):
        model.train(
            data=data,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device='cuda',
            name='egg-counting',
            project=os.path.join(project_dir, '..', 'monitoring', 'runs'),
            exist_ok=True
        )

if __name__ == "__main__":
    train()