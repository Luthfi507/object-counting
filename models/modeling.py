import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pytz
from time import time
from loguru import logger
from ultralytics import YOLO
from ultralytics.utils import SETTINGS, downloads
import torch

import mlflow
from wrapper import YOLOWrapper
load_dotenv()

project_dir = os.path.dirname(__file__)

PROJECT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME')
DATA_DIR = os.getenv("DATASET_DIRECTORY") + PROJECT_NAME
RUN_NAME = str(datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Jakarta')).strftime("%d-%m-%y:%H-%M-%S-%f"))
os.environ['MLFLOW_RUN'] = RUN_NAME

SETTINGS.update(
    runs_dir=os.getenv("RUNS_DIR"), 
    weights_dir=os.getenv("WEIGHTS_DIR"),
    mlflow=True
)

def format_time(t: float):
    return timedelta(seconds=t)

def custom_on_train_end(trainer):
    save_dir: Path = trainer.save_dir
    best_onnx = save_dir / 'weights/best.onnx'

    for f in save_dir.glob('*'):
        if f.suffix in {".png", ".jpg", ".csv", ".yaml"}:
            mlflow.log_artifact(str(f))

    wrapper_path = os.path.join(project_dir, 'wrapper.py')
    mlflow.pyfunc.log_model(
        name='model',
        python_model=YOLOWrapper(),
        artifacts={'model_path': str(best_onnx)},
        code_paths=[wrapper_path],
        model_config={'predict_fn': 'predict', 'params': {}}
    )
    keep_run_active = os.environ.get("MLFLOW_KEEP_RUN_ACTIVE", "False").lower() == "true"
    if keep_run_active:
        logger.info(f"mlflow run still alive, remember to close it using mlflow.end_run()")
    else:
        mlflow.end_run()
        logger.debug(f"mlflow run ended")

class Trainer:
    def __init__(self, data, model_type='yolo11m'):
        self.data = data
        self.model_type = model_type
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")

    def train(self, epochs, imgsz, batch, **kwargs):
        logger.info(f"Starting training with model: {self.model_type}")
        start = time()

        if not (Path(SETTINGS['weights_dir']) / f"{self.model_type}.pt").exists():
            logger.warning(f"Model weights for {self.model_type} not found. Attempting to download...")
            downloads.attempt_download_asset(f"{self.model_type}.pt", dir=SETTINGS['weights_dir'])
        self.model = YOLO(f"{self.model_type}.pt")
        self.model.callbacks['on_train_end'] = [custom_on_train_end]

        result = self.model.train(
            data=self.data,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=PROJECT_NAME,
            device=self.device,
            exist_ok=True,
            **kwargs
        )
        self.model.export(format='onnx', dynamic=True)
        elapsed = time() - start
        logger.success(f"Training completed in {format_time(elapsed)} seconds")
        return result.save_dir

    def eval(self, best_path):
        logger.info("Evaluating model...")
        start = time()
        model = YOLO(best_path)
        results = model.val(
            data=self.data,
            split='test',
            imgsz=224,
            batch=8,
            project=PROJECT_NAME,
            device=self.device,
            exist_ok=True
        )
        metrics = results.results_dict
        elapsed = time() - start
        logger.success(f"Evaluation completed in {format_time(elapsed)} seconds")
        return {k.replace('metrics/', '').replace('(B)', ' testing'): v for k, v in metrics.items() if 'metrics' in k}

    def run(self, epochs, imgsz, batch, **kwargs):
        start = time()

        save_dir = self.train(epochs, imgsz, batch, **kwargs)
        best_pt = save_dir / 'weights/best.pt'
        eval_results = self.eval(best_pt)

        total_elapsed = time() - start
        logger.success(f"Total run completed in {format_time(total_elapsed)} seconds")
        return eval_results
    
if __name__ == "__main__":
    # Define training parameters
    data = os.path.join(DATA_DIR, 'data.yaml')
    params = {
        'epochs': 150,
        'imgsz': 224,
        'batch': 32,
        'optimizer': 'SGD',
        'patience': 20,
    }

    hyps = {
        'lr0': 0.01,  # initial learning rate
        'lrf': 0.01,  # final learning rate (multiplier)
        'momentum': 0.937,  # momentum
        'weight_decay': 0.0005,  # optimizer weight decay
        'scale': 0.5,  # image scale (default 0.5)
        'fliplr': 0.5,  # horizontal flip probability
        'flipud': 0.5,  # vertical flip probability
        'hsv_h': 0.015,  # image HSV-Hue augmentation (fraction)
        'hsv_s': 0.7,  # image HSV-Saturation augmentation (fraction)
        'hsv_v': 0.4,  # image HSV-Value augmentation (fraction)
        'erasing': 0.2,  # random erasing probability
    }

    trainer = Trainer(data, 'yolo26m')
    eval_results = trainer.run(**params, **hyps)
    print(eval_results)