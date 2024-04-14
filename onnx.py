import onnx
import tensorflow as tf

# Load ONNX model
onnx_model = onnx.load("best.onnx")

# Convert ONNX model to TensorFlow graph
tf_rep = tf.experimental.tensorrt.Converter.from_onnx(onnx_model)

# Convert TensorFlow graph to TFLite model
converter = tf.lite.TFLiteConverter.from_concrete_functions([tf_rep.build_concrete_function()])
tflite_model = converter.convert()

# Save TFLite model to a file
with open("converted_model.tflite", "wb") as f:
    f.write(tflite_model)
