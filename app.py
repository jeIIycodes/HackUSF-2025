from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the model once when the app starts
model = load_model('model/xgboost_skin_cancer.pkl')

def preprocess_image(image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/test', methods=['GET'])
def testing():
    return "test"


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        probability = float(prediction[0][0])
        print(prediction)
        label = "malignant" if probability > 0.5 else "benign"
        print(prediction)
        return jsonify({
            'prediction': label,
            'probability': probability
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

