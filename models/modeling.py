import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta
import pytz
from time import time
from loguru import logger
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
import torch

import mlflow
from wrapper import YOLOWrapper
load_dotenv()

data_dir = os.getenv("DATASET_DIRECTORY") + '/vehicles-2'
run_name = str(datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Jakarta')).strftime("%d-%m-%y:%H-%M-%S-%f"))

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("object-counting")
SETTINGS.update(runs_dir=os.getenv("RUNS_DIR"), weights_dir=os.getenv("WEIGHTS_DIR"))

def format_time(t: float):
    return timedelta(seconds=t)

class Trainer:
    def __init__(self, data, project_name, model_type='yolo11m'):
        self.data = data
        self.project_name = project_name
        self.model_type = model_type
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")

    def train(self, epochs, imgsz, batch, **kwargs):
        logger.info(f"Starting training with model: {self.model_type}")
        start = time()
        model = YOLO(f"{self.model_type}.pt")
        results = model.train(
            data=self.data,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=self.project_name,
            device=self.device,
            exist_ok=True,
            **kwargs
        )
        model.export(format='onnx', dynamic=True)
        elapsed = time() - start
        logger.success(f"Training completed in {format_time(elapsed)} seconds")
        return results.save_dir

    def log_training_metrics(self, runs_dir: str):
        logger.info("Log training metrics")
        csv_path = os.path.join(runs_dir, 'results.csv')
        if not os.path.exists(csv_path):
            logger.warning(f"results.csv not found in {runs_dir}")
            return

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        metric_cols = [
            'train/box_loss',
            'train/cls_loss',
            'train/dfl_loss',
            'metrics/precision(B)',
            'metrics/recall(B)',
            'metrics/mAP50(B)',
            'metrics/mAP50-95(B)',
            'val/box_loss',
            'val/cls_loss',
            'val/dfl_loss',
        ]

        for _, row in df.iterrows():
            epoch = int(row['epoch'])
            for col in metric_cols:
                if col in df.columns and pd.notna(row[col]):
                    metric_name = col.replace('/', '_').replace('(B)', '')
                    mlflow.log_metric(metric_name, float(row[col]), step=epoch)

        logger.success(f"Logged {len(df)} epochs × {len(metric_cols)} metric to MLflow")

    def eval(self, best_path):
        logger.info("Evaluating model...")
        start = time()
        model = YOLO(best_path)
        results = model.val(
            data=self.data,
            split='test',
            imgsz=224,
            batch=8,
            project=self.project_name,
            device=self.device,
            exist_ok=True
        )
        metrics = results.results_dict
        elapsed = time() - start
        logger.success(f"Evaluation completed in {format_time(elapsed)} seconds")
        return {k.replace('metrics/', '').replace('(B)', ' testing'): v for k, v in metrics.items() if 'metrics' in k}

    def run(self, epochs, imgsz, batch, **kwargs):

        with mlflow.start_run(run_name=run_name):
            runs_dir = self.train(epochs, imgsz, batch, **kwargs)
            self.log_training_metrics(runs_dir)

            best_path = os.path.join(runs_dir, 'weights', 'best.pt')
            eval_results = self.eval(best_path)

            mlflow.log_artifacts(runs_dir)
            mlflow.log_metrics(eval_results)  
            mlflow.log_params({
                'epochs': epochs,
                'imgsz': imgsz,
                'batch': batch,
                **kwargs
            })
            mlflow.pyfunc.log_model(
                artifact_path='model',
                python_model=YOLOWrapper(),
                artifacts={'model_path': best_path}
            )

        return eval_results
    
if __name__ == "__main__":
    # Define training parameters
    data = os.path.join(data_dir, 'data.yaml')
    epochs = 50
    imgsz = 224
    batch = 16

    hyps = {
        'lr0': 0.01,  # initial learning rate
        'lrf': 0.01,  # final learning rate (multiplier)
        'momentum': 0.937,  # SGD momentum
        'weight_decay': 0.0005,  # optimizer weight decay
        'scale': 0.5,  # image scale (default 0.5)
        'fliplr': 0.5,  # horizontal flip probability
        'flipud': 0.5,  # vertical flip probability
        'hsv_h': 0.015,  # image HSV-Hue augmentation (fraction)
        'hsv_s': 0.7,  # image HSV-Saturation augmentation (fraction)
        'hsv_v': 0.4,  # image HSV-Value augmentation (fraction)
        'erasing': 0.2,  # random erasing probability
    }

    trainer = Trainer(data=data, project_name="car-counting")
    eval_results = trainer.run(epochs=epochs, imgsz=imgsz, batch=batch, **hyps)
    print(eval_results)