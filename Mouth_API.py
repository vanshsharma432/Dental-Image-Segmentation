from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import torch
import numpy as np
import traceback
import tensorflow as tf

# Initialize FastAPI
app = FastAPI(title="YOLOv8 Detection & U-Net Segmentation API")

# 1. Load YOLOv8 detection model
detection_model = YOLO("C:/Users/Admin/Documents/Notebooks/Mouth_Detection.pt")

# Custom metrics and loss functions for U-Net
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def iou_coef(y_true, y_pred, smooth=1):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)  # Binarize predictions

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3]) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return tf.reduce_mean(iou)

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
    return 1 - dice

def iou_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred) + iou_loss(y_true, y_pred)

# 2. Load TensorFlow/Keras U-Net segmentation model
UNET_MODEL_PATH = "C:/Users/Admin/Documents/Notebooks/Dental_mkIV.h5" 
UNET_INPUT_SIZE = (256, 256)

try:
    segmentation_model = tf.keras.models.load_model(UNET_MODEL_PATH, custom_objects={
        'dice_coef': dice_coef,
        'iou_coef': iou_coef,
        'combined_loss': combined_loss,
        'dice_loss': dice_loss,
        'iou_loss': iou_loss})
    print(f"Successfully loaded U-Net model from {UNET_MODEL_PATH}")
except Exception as e:
    print(f"Error loading U-Net model: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image bytes
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # --- Stage 1: Run Detection on ORIGINAL image ---
        detection_results = detection_model(image, verbose=False)

        predictions = []
        if len(detection_results[0].boxes) == 0:
            return JSONResponse({"error": "No detections found."})

        for r in detection_results:
            for box in r.boxes:
                # BBox coordinates on original full-res image
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Crop from the ORIGINAL image (high-quality)
                cropped = image.crop((x1, y1, x2, y2))
                
                if cropped.size[0] == 0 or cropped.size[1] == 0:
                    continue

                # --- Stage 2 - U-Net Segmentation ---

                # 1. Pre-processing for U-Net
                # Resize the high-quality crop to the U-Net's required input size
                unet_input_image = cropped.resize(UNET_INPUT_SIZE, Image.NEAREST)
                
                # Convert PIL Image to NumPy array
                input_array = tf.keras.preprocessing.image.img_to_array(unet_input_image)
                
                # Normalize pixel values (assuming 0-1 range, adjust if needed)
                input_array = input_array / 255.0
                
                # Add batch dimension (H, W, C) -> (1, H, W, C)
                input_tensor = np.expand_dims(input_array, axis=0)

                # 2. Run U-Net Inference
                mask_prediction = segmentation_model.predict(input_tensor, verbose=0)

                # 3. Post-processing the U-Net mask
                # Get the mask from the batch (1, H, W, 1) -> (H, W, 1)
                # Squeeze to (H, W)
                mask_array = np.squeeze(mask_prediction[0]) 

                # Apply a threshold to create a binary mask
                threshold = 0.5
                binary_mask = (mask_array > threshold).astype(np.uint8)
                
                # Scale to 0-255 for PIL Image
                mask_np = binary_mask * 255
                
                # Convert NumPy array mask to PIL Image
                mask_pil = Image.fromarray(mask_np, mode='L') # 'L' = 8-bit grayscale

                # 4. *** CRITICAL STEP ***
                # Resize the small mask (e.g., 256x256) back up to the
                # size of the HIGH-QUALITY crop to match it.
                mask_pil = mask_pil.resize(cropped.size, Image.NEAREST)

                segmented_bytes_hex = None

                # Create a new transparent image (RGBA)
                segmented_image = Image.new("RGBA", cropped.size, (0, 0, 0, 0))
                
                # Paste the HIGH-QUALITY cropped content, using the resized mask
                segmented_image.paste(cropped.convert("RGBA"), (0, 0), mask_pil)

                # Save this final segmented image to bytes
                buf = io.BytesIO()
                segmented_image.save(buf, format="PNG") # Use PNG for transparency
                buf.seek(0)
                segmented_bytes_hex = buf.getvalue().hex()

                seg_buf = io.BytesIO()
                cropped.save(seg_buf, format="PNG") # Use PNG for transparency
                seg_buf.seek(0)
                cropped_bytes_hex = seg_buf.getvalue().hex()
                
                # Append the final result
                predictions.append({
                    "class": int(box.cls),
                    "name": detection_model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": [x1, y1, x2, y2], # Bbox on original image
                    "segmented_image": segmented_bytes_hex, # High-res segmented PNG
                    "crop": cropped_bytes_hex # High-res crop PNG
                })

        return JSONResponse({"predictions": predictions})

    except Exception as e:
        print(traceback.format_exc()) # Print full error trace to console
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()})