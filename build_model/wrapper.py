from ultralytics import YOLO
import mlflow
import pandas as pd

class YOLOWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = YOLO(context.artifacts['model_path'])

    def predict(self, context, model_input):
        preds = self.model(model_input)
        pred_xyxy = pd.DataFrame(preds[0].boxes.xyxy.cpu().numpy(), columns=['xmin', 'ymin', 'xmax', 'ymax'])
        pred_class = pd.DataFrame(preds[0].boxes.cls.cpu().numpy(), columns=['class']).astype(int)
        pred_conf = pd.DataFrame(preds[0].boxes.conf.cpu().numpy(), columns=['confidence'])
        pred_class['name'] = pred_class['class'].replace(preds[0].names)
        return pd.concat([pred_xyxy, pred_conf, pred_class], axis=1)
    
if __name__ == "__main__":
    class FakeContext:
        def __init__(self, artifacts):
            self.artifacts = artifacts

    wrapper = YOLOWrapper()
    context = FakeContext({'model_path': '../monitoring/runs/weights/best.onnx'})
    wrapper.load_context(context)

    img_path = '../dataset/Egg-1/test/images/not_damaged_9_jpg.rf.bf5fa2b72bc716e9e2431a0756e7923a.jpg'
    pred = wrapper.predict(context=None, model_input=img_path)
    print(pred)