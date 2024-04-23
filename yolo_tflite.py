from ultralytics import YOLO
import picamera
import picamera.array

# Load the YOLOv8 TFLite model
model = YOLO('best_float32.tflite')

# Initialize the Raspberry Pi camera
camera = picamera.PiCamera()
camera.resolution = (640, 480)

# Capture an image
with picamera.array.PiRGBArray(camera) as output:
    camera.capture(output, 'rgb')
    image = output.array

# Perform object detection
results = model(image)

# Process the results
for result in results:
    boxes = result.boxes
    labels = result.get_labels()
    scores = result.get_scores()

    # Do something with the detected objects

# Clean up
camera.close()
