from ultralytics import YOLO
model=YOLO("best.onnx")
model.predict('tcp://127.0.0.1:8888', imgsz=320, stream=True)