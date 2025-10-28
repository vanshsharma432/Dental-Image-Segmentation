import requests
from PIL import Image
import io
import sys

# API URL
url = "http://127.0.0.1:8000/predict"

try:
    # Send image to API
    with open("smile.jpg", "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)

    # 1. Check for HTTP errors (e.g., 500 Internal Server Error)
    response.raise_for_status() 

    # 2. Parse JSON
    data = response.json()

    # 3. *** Check for the "error" key FIRST ***
    if "error" in data:
        print(f"Error from API: {data['error']}")
        
        # Optional: print the full server trace if it exists
        if "trace" in data:
            print("\n--- Server Traceback ---")
            print(data['trace'])
            print("------------------------")
        
        sys.exit(1) # Exit the script because an error occurred

    # 4. If no error, get the predictions
    predictions = data.get("predictions", [])

    if not predictions:
        print("API returned success, but no predictions were found.")
        sys.exit(0)

    print(f"Found {len(predictions)} detection(s):")

    # Loop through predictions and display crops
    for i, pred in enumerate(predictions):
        print(f"\n{i+1}. Class: {pred['name']}, Confidence: {pred['confidence']:.2f}")

        # Check if segmented_image was returned (it might be None)
        if pred.get("segmented_image"):
            # Convert hex string back to bytes
            crop_bytes = bytes.fromhex(pred["segmented_image"])

            # Open image from bytes
            img_crop = Image.open(io.BytesIO(crop_bytes))

            # Display image
            img_crop.show()  # Opens default image viewer
            
            # Optional: save locally
            # img_crop.save(f"crop_{i+1}_{pred['name']}.png")
        else:
            print("  (No segmented image was returned for this detection)")

except requests.exceptions.HTTPError as e:
    # Handle bad responses like 500, 404
    print(f"HTTP Error: {e.response.status_code} {e.response.reason}")
    print(f"Details: {e.response.text}") # Print server's full error page
except requests.exceptions.RequestException as e:
    # Handle connection errors
    print(f"Connection Error: {e}")
except Exception as e:
    # Handle other errors (like failed .json() parsing)
    print(f"An unexpected error occurred: {e}")