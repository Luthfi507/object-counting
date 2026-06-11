from ultralytics import YOLO
import mlflow

class YOLOWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        model_path = context.artifacts['model_path']
        self.model = YOLO(model_path, task='detect')
        print(f"Loaded model from {model_path}")

    def predict(self, model_input, params=None):
        params = params or {}
        preds = self.model.predict(model_input, **params)
        return preds
    
if __name__ == "__main__":
    class FakeContext:
        def __init__(self, artifacts):
            self.artifacts = artifacts

    wrapper = YOLOWrapper()
    context = FakeContext({'model_path': '/media/duke/Data/deployment/runs/detect/car-counting/train/weights/best.onnx'})
    wrapper.load_context(context)

    img_path = '/media/duke/Data/datasets/Car-1/test/images/IMG_0602_jpeg.rf.fdd986ef02f7e3c0086b989c11adc0bd.jpg'
    params = {'imgsz': 224, 'conf': 0.25}
    pred = wrapper.predict(model_input=img_path, params=params)
    print(pred)