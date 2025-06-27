import numpy as np
import cv2
from tensorflow.keras.models import load_model
import json
import sys

# Constants
IMAGE_SIZE = (128, 128)
TYPE_LIST = ['G', 'R', 'W', 'L', 'P', 'M', 'D', 'F', 'Y', 'C', 'N']
STAGE_MAP_REV = {0: 'Basic', 1: '1', 2: '2'}

# === Load model ===
model = load_model("pokemon_model.keras")

# === Load and preprocess image ===
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# === Decode prediction vector into JSON ===
def decode_output(vec):
    output = {}

    vec = vec.flatten()
    i = 0

    output["hp"] = int(vec[i])
    i += 1

    # Type (one-hot)
    type_idx = np.argmax(vec[i:i+len(TYPE_LIST)])
    output["type"] = TYPE_LIST[type_idx]
    i += len(TYPE_LIST)

    output["stage"] = STAGE_MAP_REV.get(int(vec[i]), "Basic")
    i += 1

    output["retreat"] = int(vec[i])
    i += 1

    # Weak
    weak_idx = np.argmax(vec[i:i+len(TYPE_LIST)])
    output["weak"] = TYPE_LIST[weak_idx]
    i += len(TYPE_LIST)

    # Resist
    resist_idx = np.argmax(vec[i:i+len(TYPE_LIST)])
    output["resist"] = TYPE_LIST[resist_idx]
    i += len(TYPE_LIST)

    # Attacks
    output["attacks"] = []
    for _ in range(2):
        damage = int(vec[i])
        cost_len = int(vec[i+1])
        i += 2
        if damage > 0 or cost_len > 0:
            output["attacks"].append({
                "damage": damage,
                "cost_length": cost_len
            })

    return output

# === Run test ===
def test_on_image(img_path):
    img = preprocess_image(img_path)
    prediction = model.predict(img)
    decoded = decode_output(prediction)
    print(json.dumps(decoded, indent=2))

# === Usage ===
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_model.py path_to_image.jpg")
    else:
        test_on_image(sys.argv[1])
