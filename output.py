import requests
from PIL import Image
import io

# API URL
url = "http://127.0.0.1:8000/predict"

# Send image to API
with open("handsome-man-smiling-happy-face-portrait-close-up_53876-139608.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

# Parse JSON
data = response.json()

# Loop through predictions and display crops
for i, pred in enumerate(data.get("predictions", [])):
    print(f"{i+1}. Class: {pred['name']}, Confidence: {pred['confidence']:.2f}")

    # Convert hex string back to bytes
    crop_bytes = bytes.fromhex(pred["crop"])

    # Open image from bytes
    img_crop = Image.open(io.BytesIO(crop_bytes))

    # Display image
    img_crop.show()  # Opens default image viewer

    # Optional: save locally
    # img_crop.save(f"crop_{i+1}_{pred['name']}.png")
