import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.models import load_model
import os
from PIL import Image

def load_efficientnet_model():
    """Load EfficientNet model specifically"""
    model_paths = [
        "../../model_images/saved/efficientnet_model.h5",
        "../model_images/saved/efficientnet_model.h5", 
        "model_images/saved/efficientnet_model.h5"
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"✅ Loading EfficientNet model from: {model_path}")
            return load_model(model_path)
    
    raise FileNotFoundError("❌ EfficientNet model not found!")

# Load EfficientNet model at startup
model = load_efficientnet_model()

# Use EXACT same preprocessing as training (validation datagen)
val_datagen = ImageDataGenerator(rescale=1./255)

def preprocess_single_image(image_path_or_array):
    """Preprocess single image using same pipeline as training"""
    if isinstance(image_path_or_array, str):
        # Load from file path
        img = tf.keras.preprocessing.image.load_img(
            image_path_or_array, 
            target_size=(224, 224)  # Same as training
        )
    else:
        # Convert numpy array to PIL Image
        if isinstance(image_path_or_array, np.ndarray):
            # Convert BGR to RGB if needed
            if len(image_path_or_array.shape) == 3:
                image_path_or_array = cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(image_path_or_array.astype('uint8'))
            img = img.resize((224, 224))  # Same as training
        else:
            img = image_path_or_array.resize((224, 224))
    
    # Convert to array and apply same preprocessing as validation
    img_array = img_to_array(img)
    img_array = val_datagen.standardize(img_array)  # Same rescale=1./255
    
    return img_array

def select_key_frames(video_path, max_frames=20):
    """Extract frames using same method as training data preprocessing"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames evenly (same as training preprocessing)
    step = max(1, total_frames // max_frames)
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % step == 0:
            frames.append(frame)  # Keep as numpy array for preprocessing
        
        frame_count += 1
    
    cap.release()
    return frames

def predict_image_batch(images):
    """Process images using exact same preprocessing as training"""
    # Preprocess each image using training pipeline
    processed_images = []
    for img in images:
        processed_img = preprocess_single_image(img)
        processed_images.append(processed_img)
    
    # Convert to batch
    batch = np.array(processed_images)
    
    # Predict using EfficientNet
    predictions = model.predict(batch, verbose=0)
    return predictions.flatten()

def predict_video(video_path, threshold=0.5, max_frames=20):
    """Predict using EfficientNet with exact training preprocessing"""
    try:
        # Handle both images and videos
        if video_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Single image
            processed_img = preprocess_single_image(video_path)
            batch = np.expand_dims(processed_img, axis=0)
            predictions = model.predict(batch, verbose=0)
            avg_score = predictions[0][0]
        else:
            # Video - extract frames
            frames = select_key_frames(video_path, max_frames)
            
            if len(frames) == 0:
                return "Unable to process", 0.0

            predictions = predict_image_batch(frames)
            avg_score = np.mean(predictions)
        
        # Fix class mapping: flow_from_directory assigns alphabetically (fake=0, real=1)
        # Model output: 0 = fake, 1 = real
        result = "Real" if avg_score > threshold else "Fake"
        confidence = float(avg_score) if avg_score > threshold else float(1 - avg_score)
        
        print(f"🎯 EfficientNet Prediction: {result} (raw_score: {avg_score:.3f}, confidence: {confidence:.3f})")
        return result, confidence
        
    except Exception as e:
        print(f"❌ Error processing: {e}")
        return "Error", 0.0
