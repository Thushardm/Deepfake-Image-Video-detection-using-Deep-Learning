import cv2
import numpy as np
import gc
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

def load_best_model():
    """Load the best available model"""
    models = [
        ("../../models/saved/efficientnet_model.h5", "EfficientNet"),
        ("../../models/saved/resnet50_model.h5", "ResNet50"),
        ("../../models/saved/vgg16_model.h5", "VGG16"),
        ("../../models/saved/cnn_model.h5", "CNN")
    ]
    
    for model_file, model_name in models:
        if os.path.exists(model_file):
            print(f"Loading {model_name} model...")
            return load_model(model_file)
    
    raise FileNotFoundError("No trained model found!")

model = load_best_model()

def select_key_frames(video_path, max_frames=20):
    """Select key frames from video"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames evenly
    step = max(1, total_frames // max_frames)
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % step == 0:
            frame_resized = cv2.resize(frame, (224, 224))
            frames.append(frame_resized)
        
        frame_count += 1
    
    cap.release()
    return frames

def predict_image_batch(images):
    """Process images in batch"""
    batch = np.array([img_to_array(img) / 255.0 for img in images])
    predictions = model.predict(batch, verbose=0)
    return predictions.flatten()

def predict_video(video_path, threshold=0.5, max_frames=20):
    """Simplified video prediction"""
    try:
        frames = select_key_frames(video_path, max_frames)
        
        if len(frames) == 0:
            return "Unable to process video", 0.0

        predictions = predict_image_batch(frames)
        avg_score = np.mean(predictions)
        
        result = "Fake" if avg_score > threshold else "Real"
        return result, float(avg_score)
        
    except Exception as e:
        print(f"Error processing video: {e}")
        return "Error", 0.0
