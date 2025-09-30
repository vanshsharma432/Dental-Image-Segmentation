import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
import tensorflow as tf 

app = Flask(__name__)

model = tf.keras.models.load_model('your_model.h5') 

def preprocess_image(image_bytes):
    """
    Preprocesses the input image to match your model's input requirements.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # Resize to the size your model expects, e.g., (256, 256)
    img = img.resize((256, 256)) 
    img_array = np.array(img)
    # Normalize if needed, e.g., dividing by 255.0
    img_array = img_array / 255.0 
    # Add a batch dimension
    return np.expand_dims(img_array, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['image']
    img_bytes = file.read()

    # 1. Preprocess the image
    preprocessed_image = preprocess_image(img_bytes)

    # 2. Make a prediction
    pred_mask = model.predict(preprocessed_image)[0]

    # 3. Post-process the mask
    # Threshold the mask, as in your original code
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255 
    mask_image = Image.fromarray(pred_mask.squeeze(), mode='L') # 'L' for grayscale

    # 4. Return the mask as a base64 encoded string
    buffered = io.BytesIO()
    mask_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({'mask': img_str})

if __name__ == '__main__':
    # Run the app on port 5000
    app.run(host='0.0.0.0', port=5000)