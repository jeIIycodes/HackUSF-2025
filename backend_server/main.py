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
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)
CORS(app)

# ------------------------------
# Load XGBoost Model & Scaler
# ------------------------------
try:
    # Load your XGBClassifier (assumes JSON format) and scaler
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model('final_skin_cancer_model.json')
    scaler = joblib.load('final_scaler.pkl')
except Exception as e:
    print("Error loading XGB model or scaler:", e)
    xgb_model = None
    scaler = None

# ------------------------------
# Load CNN Classification Model for Malignant Types
# ------------------------------
NUM_MALIGNANT_CLASSES = 2
# Define a simple model architecture identical to what you trained
def load_malignant_classifier(model_path):
    model = models.resnet50(pretrained=True)
    # Unfreeze last layer (or only replace fc)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_MALIGNANT_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

try:
    malignant_model = load_malignant_classifier("melanoma_classifier.pth")
except Exception as e:
    print("Error loading malignant type classifier:", e)
    malignant_model = None

# Define the mapping for malignant types (adjust indices as per your training)
malignant_class_mapping = {
    0: "Malignant melanoma",
    1: "Possible carcinoma",

}

# Define transforms for the CNN model (should match training)
cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------------------
# Helper function to extract color features for XGB model
# ------------------------------
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

    l_hist, _ = np.histogram(L.flatten(), bins=256, range=(0, 100), density=True)
    a_hist, _ = np.histogram(A.flatten(), bins=256, range=(-128, 128), density=True)
    b_hist, _ = np.histogram(B.flatten(), bins=256, range=(-128, 128), density=True)

    hist_features = np.concatenate([l_hist, a_hist, b_hist])
    final_vector = np.concatenate([moments, hist_features])
    return final_vector

# ------------------------------
# Flask Endpoints
# ------------------------------
@app.route("/hello", methods=["GET"])
def hello():
    return "hello", 200

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        # Ensure the XGB model and scaler are loaded
        if xgb_model is None or scaler is None:
            return jsonify({"error": "XGB model or scaler not loaded."}), 500

        # Check if an image file was provided
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided."}), 400

        file = request.files['image']
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_cv is None:
            return jsonify({"error": "Invalid image file."}), 400

        # For XGB model: Convert image to RGB and resize to 224x224 using cv2
        img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_cv_rgb = cv2.resize(img_cv_rgb, (224, 224))
        features = extract_color_features(img_cv_rgb)
        features = features.reshape(1, -1)
        features_scaled = scaler.transform(features)

        # Get prediction probabilities from the XGB model
        probabilities = xgb_model.predict_proba(features_scaled)
        # (Optional adjustment if needed)
        probabilities[0][2] *= 0.01
        probabilities[0][1] *= 0.7
        probabilities[0] /= np.sum(probabilities[0])

        pred_class = np.argmax(probabilities, axis=1)[0]
        pred_prob = probabilities[0][pred_class]

        # Map predicted index to class label for three options
        class_mapping = {0: "Benign", 1: "Malignant", 2: "Neither"}
        overall_label = class_mapping.get(pred_class, "Unknown")

        response = {
            "classification": overall_label,
            "probability": float(pred_prob)
        }

        # If overall classification is "Malignant", use the CNN model to predict the specific cancer type
        if overall_label == "Malignant" and malignant_model is not None:
            # Prepare the image for the CNN model using PIL and the defined transforms
            file.stream.seek(0)  # reset file pointer
            pil_img = Image.open(file.stream).convert("RGB")
            img_tensor = cnn_transform(pil_img).unsqueeze(0)  # add batch dim

            malignant_model.eval()
            with torch.no_grad():
                outputs = malignant_model(img_tensor)
                _, cnn_pred = torch.max(outputs, 1)
                cancer_type = malignant_class_mapping.get(cnn_pred.item(), "Unknown Malignant Type")
            response["malignant_type"] = cancer_type

        return jsonify(response), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
