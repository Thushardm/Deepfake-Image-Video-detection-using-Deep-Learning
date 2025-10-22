import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import pickle

def extract_real_video_sequences(video_dir, sequence_length=5, max_sequences_per_video=10):
    """Extract real consecutive frames from videos"""
    sequences = []
    labels = []
    
    print(f"🎬 Extracting sequences from {video_dir}...")
    
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    for i, video_file in enumerate(video_files):
        if i % 50 == 0:
            print(f"   Processing video {i+1}/{len(video_files)}: {video_file}")
        
        video_path = os.path.join(video_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        
        # Determine label from filename/directory
        label = 1 if any(keyword in video_file.lower() for keyword in ['fake', 'synthesis', 'deepfake']) else 0
        
        frames = []
        frame_count = 0
        sequences_from_video = 0
        
        # Get total frames and calculate step size
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step_size = max(1, total_frames // (max_sequences_per_video * sequence_length))
        
        while sequences_from_video < max_sequences_per_video:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames based on step size for diversity
            if frame_count % step_size == 0:
                # Resize and normalize
                frame_resized = cv2.resize(frame, (224, 224))
                frame_normalized = frame_resized.astype(np.float32) / 255.0
                frames.append(frame_normalized)
                
                # Create sequence when we have enough frames
                if len(frames) == sequence_length:
                    sequences.append(np.array(frames))
                    labels.append(label)
                    sequences_from_video += 1
                    
                    # Sliding window: remove first frame, keep others
                    frames.pop(0)
            
            frame_count += 1
        
        cap.release()
    
    print(f"✅ Extracted {len(sequences)} sequences")
    return np.array(sequences), np.array(labels)

def create_video_sequence_dataset():
    """Create train/test splits from video sequences"""
    
    # Check if videos exist
    video_dirs = [
        "../../data/Celeb-DF/videos/real",
        "../../data/Celeb-DF/videos/fake"
    ]
    
    all_sequences = []
    all_labels = []
    
    for video_dir in video_dirs:
        if os.path.exists(video_dir):
            sequences, labels = extract_real_video_sequences(video_dir)
            all_sequences.extend(sequences)
            all_labels.extend(labels)
        else:
            print(f"⚠️ Directory not found: {video_dir}")
    
    if len(all_sequences) == 0:
        print("❌ No video sequences found! Using fallback method...")
        return create_fallback_sequences()
    
    # Convert to numpy arrays
    X = np.array(all_sequences)
    y = np.array(all_labels)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Dataset created:")
    print(f"   Train: {X_train.shape[0]} sequences")
    print(f"   Test: {X_test.shape[0]} sequences")
    print(f"   Real/Fake ratio: {np.sum(y==0)}/{np.sum(y==1)}")
    
    # Save dataset
    os.makedirs("../../data/video_sequences", exist_ok=True)
    
    with open("../../data/video_sequences/train_data.pkl", "wb") as f:
        pickle.dump((X_train, y_train), f)
    
    with open("../../data/video_sequences/test_data.pkl", "wb") as f:
        pickle.dump((X_test, y_test), f)
    
    return X_train, X_test, y_train, y_test

def create_fallback_sequences():
    """Fallback: Create better artificial sequences from existing frames"""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    print("🔄 Creating enhanced artificial sequences...")
    
    # Load existing frame data
    datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = datagen.flow_from_directory(
        "../../data/Celeb-DF/split_data/train",
        target_size=(224, 224),
        batch_size=1000,
        class_mode='binary',
        shuffle=True
    )
    
    # Get large batch of images
    batch_x, batch_y = next(train_gen)
    
    sequences = []
    labels = []
    sequence_length = 5
    
    # Group by class for better sequences
    real_images = batch_x[batch_y == 0]
    fake_images = batch_x[batch_y == 1]
    
    # Create sequences within same class
    for class_images, class_label in [(real_images, 0), (fake_images, 1)]:
        for i in range(0, len(class_images) - sequence_length + 1, sequence_length):
            sequence = class_images[i:i+sequence_length]
            
            # Add slight temporal variation
            for j in range(1, sequence_length):
                # Add progressive noise to simulate temporal change
                noise = np.random.normal(0, 0.02 * j, sequence[j].shape)
                sequence[j] = np.clip(sequence[j] + noise, 0, 1)
            
            sequences.append(sequence)
            labels.append(class_label)
    
    X = np.array(sequences)
    y = np.array(labels)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Fallback dataset created:")
    print(f"   Train: {X_train.shape[0]} sequences")
    print(f"   Test: {X_test.shape[0]} sequences")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    create_video_sequence_dataset()
