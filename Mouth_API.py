from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io

# Initialize FastAPI
app = FastAPI(title="YOLOv8 Object Detection API")

# Load YOLOv8 model
model = YOLO("C:/Users/Admin/Documents/Notebooks/Mouth_Detection.pt")  # path to your trained model

# Resize function
def resize_image(image: Image.Image, size=(640, 640)):
    return image.resize(size)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image bytes
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Resize to 640x640
        image_resized = resize_image(image, (640, 640))

        # Run inference
        results = model(image_resized)

        # Parse predictions
        predictions = []
        crops = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Crop the image inside bounding box
                cropped = image_resized.crop((x1, y1, x2, y2))

                # Save cropped image to bytes
                buf = io.BytesIO()
                cropped.save(buf, format="PNG")
                buf.seek(0)
                crop_bytes = buf.getvalue()

                predictions.append({
                    "class": int(box.cls),
                    "name": model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": [x1, y1, x2, y2],
                    "crop": crop_bytes.hex()  # return as hex string
                })

        return JSONResponse({"predictions": predictions})

    except Exception as e:
        return JSONResponse({"error": str(e)})
