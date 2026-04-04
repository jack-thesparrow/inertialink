import tensorflow as tf
import tf2onnx
import onnx
import os

# Paths
KERAS_MODEL_PATH = "data/smart_pen_bilstm.keras"
ONNX_MODEL_PATH = "data/smart_pen.onnx"

print(f"Loading Keras model from {KERAS_MODEL_PATH}...")
keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH)

# We must explicitly tell ONNX what shape the C++ arrays will be.
# Shape: [Batch Size (None = dynamic), 100 frames, 3 features]
input_signature = [tf.TensorSpec([None, 100, 3], tf.float32, name="input_layer")]

print("Converting to ONNX format (this may take a moment)...")
# opset=13 is highly compatible with the C++ ONNX Runtime
onnx_model, _ = tf2onnx.convert.from_keras(keras_model, input_signature, opset=13)

onnx.save(onnx_model, ONNX_MODEL_PATH)
print(f"\n[SUCCESS] Model successfully exported to {ONNX_MODEL_PATH}")
print(f"File size: {os.path.getsize(ONNX_MODEL_PATH) / 1024 / 1024:.2f} MB")
