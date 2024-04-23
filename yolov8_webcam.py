# from ultralytics import YOLO
# 
# model = YOLO('best.pt')
# results = model(source=0, show=True)
# 
# 
# for result in results:
#     print(result.boxes, result.probs)


from ultralytics import YOLO
model=YOLO("best.onnx")
results=model.predict(source=0, show=True, stream=True)

# results=model(boxes)