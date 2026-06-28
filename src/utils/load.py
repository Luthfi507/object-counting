import mlflow
from loguru import logger
import threading
import time

from config import MODEL_DIR

class ModelVersion:
    def __init__(self, name: str, alias: str, model_config: dict = {}, poll_interval=60):
        self.name = name
        self.alias = alias
        self.model_config = model_config
        self.poll_interval = poll_interval
        self.current_version = None
        self.model = None

        self.load_model()
        self.start_watcher()

    # Check production model version
    def get_version(self):
        client = mlflow.MlflowClient()
        versions = client.get_model_version_by_alias(name=self.name, alias=self.alias)
        return versions.version
    
    # Load model from mlflow
    def load_model(self):
        latest_version = self.get_version()
        dst_path = MODEL_DIR / self.model_config.get('predict_fn', 'predict')
        dst_path.mkdir(exist_ok=True)

        if latest_version != self.current_version:
            logger.warning(f"Model {self.name} version changed: {self.current_version} -> {latest_version}")
            model_uri = f"models:/{self.name}@{self.alias}"
            self.model = mlflow.pyfunc.load_model(model_uri, dst_path=str(dst_path), model_config=self.model_config)
            self.current_version = latest_version
            logger.debug(f"Current version: {self.current_version}")
        else:
            logger.debug(f"Version not changed for {self.name}")

    def start_watcher(self):
        def watch():
            while True:
                time.sleep(self.poll_interval)
                try:
                    self.load_model()
                except Exception as e:
                    logger.exception(f"Error checking: {e}")

        t = threading.Thread(target=watch, daemon=True)
        t.start()

    def predict(self, data):
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        return self.model.predict(data)