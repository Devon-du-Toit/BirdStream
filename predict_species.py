import sys
import cv2
import numpy as np
import os
from datetime import datetime
from tflite_runtime.interpreter import Interpreter  # or from tensorflow.lite import Interpreter

# Load TFLite model and allocate tensors
interpreter = Interpreter(model_path="birds6_tf2_14.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Inspect expected shape & dtype
in_info = input_details[0]
in_shape = in_info["shape"]          # e.g. [1, 256, 256, 3]
in_dtype = in_info["dtype"]          # e.g. np.float32 or np.uint8

# Your classes
CLASS_NAMES = ['Cape Sparrow', 'Cape White-eye', 'Masked Weaver', 'Squirrel', 'Swee Waxbill']

BASE_SAVE_DIR = os.path.expanduser("~/birdstream/predictions")

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
    img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    # Type handling
    if in_dtype == np.float32:
        x = img_resized.astype(np.float32) / 255.0
    elif in_dtype == np.uint8:
        x = img_resized.astype(np.uint8)  # no normalization for quantized models
    else:
        x = img_resized.astype(in_dtype)

    # Add batch dim
    x = np.expand_dims(x, axis=0)  # (1, H, W, C)
    return img, x  # return original (for saving) and preprocessed (for model)

def predict_and_save(img_path):
    orig_img, x = preprocess_image(img_path)

    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    class_idx = int(np.argmax(preds))
    prob = float(preds[class_idx])
    specie = CLASS_NAMES[class_idx]

    # Build save path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(BASE_SAVE_DIR, specie)
    os.makedirs(save_dir, exist_ok=True)

    save_filename = f"{specie}_{prob*100:.1f}_{timestamp}.jpg"
    save_path = os.path.join(save_dir, save_filename)

    # Save original image (in RGB → back to BGR for cv2.imwrite)
    cv2.imwrite(save_path, cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR))

    return specie, prob


if __name__ == "__main__":
    img_path = sys.argv[1]
    specie, prob = predict_and_save(img_path)
    print(f"{specie} ({prob * 100:.1f}%)")
