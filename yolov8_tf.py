from ultralytics import YOLO

model = YOLO('best.onnx')
results = model('tcp://127.0.0.1:8888', stream=True)


for result in results:
    print(result.boxes, result.probs)
        
        
# libcamera-vid -n -t 0 --width 1280 --height 960 --framerate 1 --inline --listen -o tcp://127.0.0.1:8888