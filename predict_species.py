import sys
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter  # or from tensorflow.lite import Interpreter

# Load TFLite model and allocate tensors
interpreter = Interpreter(model_path="birds5_tf2_14.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Inspect expected shape & dtype
in_info = input_details[0]
in_shape = in_info["shape"]          # e.g. [1, 256, 256, 3]
in_dtype = in_info["dtype"]          # e.g. np.float32 or np.uint8

# Your classes
CLASS_NAMES = ['Cape Sparrow', 'Cape White-eye', 'Masked Weaver', 'Squirrel', 'Swee Waxbill']

def preprocess_image(img_path):
    img = cv2.imread(img_path)  # BGR
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")

    # Assume NHWC; TFLite almost always uses NHWC
    if len(in_shape) != 4:
        raise ValueError(f"Unexpected input rank: {in_shape}")

    height, width = int(in_shape[1]), int(in_shape[2])

    # Convert BGR->RGB (most Keras models are RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to what the model expects
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    # Type handling
    if in_dtype == np.float32:
        img = img.astype(np.float32) / 255.0
    elif in_dtype == np.uint8:
        img = img.astype(np.uint8)  # no normalization for quantized models
    else:
        # Fallback: cast to expected dtype
        img = img.astype(in_dtype)

    # Add batch dim
    img = np.expand_dims(img, axis=0)  # (1, H, W, C)
    return img

def predict(img_path):
    x = preprocess_image(img_path)
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    class_idx = int(np.argmax(preds))
    prob = float(preds[class_idx])
    return CLASS_NAMES[class_idx], prob


if __name__ == "__main__":
    img_path = sys.argv[1]
    specie, prob = predict(img_path)
    print(f"{specie} ({prob * 100:.1f}%)")  # one decimal place

