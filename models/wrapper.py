from ultralytics import YOLO
import mlflow

class YOLOWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        model_path = context.artifacts['model_path']
        self.model = YOLO(model_path, task='detect')
        print(f"Loaded model from {model_path}")

    def predict(self, context, model_input):
        predict_fn = context.model_config.get('predict_fn', 'predict') if context.model_config else "predict"
        
        if predict_fn == 'predict': 
            print("Predict model")       
            preds = self.model.predict(model_input)
        else:
            print("Track model")
            preds = self.model.track(model_input)

        return preds