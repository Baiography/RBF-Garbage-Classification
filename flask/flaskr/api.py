import os
import base64
import traceback
from datetime import datetime
from flask import Blueprint, jsonify, request
import numpy as np
from PIL import Image
import joblib
import tensorflow as tf

bp = Blueprint("api_bp", __name__)
STORAGE_PATH = os.path.join(os.getcwd(), "resources", "public")
pca = joblib.load("C:/Users/G.I/flask/models/pca_model.pkl")
scaler = joblib.load("C:/Users/G.I/flask/models/scaler_model.pkl")
interpreter = tf.lite.Interpreter(model_path="C:/Users/G.I/flask/models/model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def classify_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((64, 64))
        img_array = np.array(img)
        img_flattened = img_array.reshape(1, -1)
        img_pca = pca.transform(img_flattened)
        img_scaled = scaler.transform(img_pca)
        interpreter.set_tensor(input_details[0]['index'], img_scaled.astype(np.float32))
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.argmax(output_data)
        class_names = ["Organic", "Recyclable"]
        return class_names[prediction]
    except Exception as e:
        print(f"Error in classification: {e}")
        return "Error"

@bp.route("/api/upload-photo", methods=["POST"])
def upload_photo():
    try:
        image = request.json["image"]
        timeNow = datetime.now().strftime("%Y%m%d-%H%M%S")
        image_path = os.path.join(STORAGE_PATH, f"{timeNow}.png")
        with open(image_path, "wb") as f:
            f.write(base64.b64decode(image))
        classification_result = classify_image(image_path)
        return jsonify({
            'status': 'success',
            'message': 'Photo uploaded and classified successfully',
            'classification': classification_result
        }), 200
    except:
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': 'Internal server error'
        }), 500
