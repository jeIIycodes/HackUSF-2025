import os
import joblib
import numpy as np
import cv2
import traceback
from flask import Flask, request, jsonify
from skimage.color import rgb2lab
from skimage.util import img_as_float
from scipy.stats import skew, kurtosis
import xgboost as xgb
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the final model and scaler at startup
try:
    # Initialize an XGBClassifier and load the JSON model
    model = xgb.XGBClassifier()
    model.load_model('final_skin_cancer_model.json')
    scaler = joblib.load('final_scaler.pkl')
except Exception as e:
    print("Error loading model or scaler:", e)
    model = None
    scaler = None


def extract_color_features(img):
    """
    Extracts color features from an image using LAB conversion.
    The features include statistical moments and histograms from the L, A, and B channels.
    """
    img = img_as_float(img)
    if np.isnan(img).any() or np.isinf(img).any():
        raise ValueError("Invalid image data: contains NaN or infinite values")

    lab = rgb2lab(img)
    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    # Moments for each channel
    l_mean, l_std = np.mean(L), np.std(L)
    l_skew, l_kurt = skew(L.flatten()), kurtosis(L.flatten())
    a_mean, a_std = np.mean(A), np.std(A)
    a_skew, a_kurt = skew(A.flatten()), kurtosis(A.flatten())
    b_mean, b_std = np.mean(B), np.std(B)
    b_skew, b_kurt = skew(B.flatten()), kurtosis(B.flatten())

    moments = np.array([
        l_mean, l_std, l_skew, l_kurt,
        a_mean, a_std, a_skew, a_kurt,
        b_mean, b_std, b_skew, b_kurt
    ])

    # Histograms for each channel
    l_hist, _ = np.histogram(L.flatten(), bins=256, range=(0, 100), density=True)
    a_hist, _ = np.histogram(A.flatten(), bins=256, range=(-128, 128), density=True)
    b_hist, _ = np.histogram(B.flatten(), bins=256, range=(-128, 128), density=True)

    hist_features = np.concatenate([l_hist, a_hist, b_hist])
    final_vector = np.concatenate([moments, hist_features])
    return final_vector


@app.route("/hello", methods=["GET"])
def hello():
    return "hello", 200


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        # Ensure the model and scaler are loaded
        if model is None or scaler is None:
            return jsonify({"error": "Model or scaler not loaded."}), 500

        # Check if an image file was provided
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided."}), 400

        file = request.files['image']
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Invalid image file."}), 400

        # Convert image to RGB and resize to match the training target size (224x224)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))

        # Extract features and prepare for prediction
        features = extract_color_features(img)
        features = features.reshape(1, -1)
        features_scaled = scaler.transform(features)

        # Get prediction probabilities from the model
        probabilities = model.predict_proba(features_scaled)

        # Reduce class 2's probability to 60% of its value
        probabilities[0][2] *= 0.6

        # Renormalize so probabilities still sum to 1
        probabilities[0] /= np.sum(probabilities[0])

        # Get predicted class and its probability
        pred_class = np.argmax(probabilities, axis=1)[0]
        pred_prob = probabilities[0][pred_class]

        # Map predicted index to class label for three options
        class_mapping = {0: "Benign", 1: "Malignant", 2: "Neither"}
        label = class_mapping.get(pred_class, "Unknown")

        print({
            "classification": label,
            "probability": float(pred_prob)
        })
        return jsonify({
            "classification": label,
            "probability": float(pred_prob)
        }), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
