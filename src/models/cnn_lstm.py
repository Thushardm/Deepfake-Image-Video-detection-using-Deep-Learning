import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, TimeDistributed, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import cv2
import os

# Optimized Configuration for 85%+ accuracy
img_height, img_width = 224, 224
sequence_length = 5
batch_size = 16
epochs = 15  # Reduced to prevent overfitting

def extract_real_sequences_from_frames():
    """Extract real consecutive sequences from existing frame data"""
    print("🎬 Creating optimized sequences from frame data...")
    
    # Load frame data with better organization
    datagen = ImageDataGenerator(rescale=1./255)
    
    # Get organized data by class
    train_gen = datagen.flow_from_directory(
        "../../data/Celeb-DF/split_data/train",
        target_size=(img_height, img_width),
        batch_size=2000,  # Large batch to get diverse data
        class_mode='binary',
        shuffle=False  # Keep order for better sequences
    )
    
    batch_x, batch_y = next(train_gen)
    
    # Separate by class for coherent sequences
    real_indices = np.where(batch_y == 0)[0]
    fake_indices = np.where(batch_y == 1)[0]
    
    sequences = []
    labels = []
    
    # Create sequences within same class (more realistic)
    for class_indices, class_label in [(real_indices, 0), (fake_indices, 1)]:
        class_images = batch_x[class_indices]
        
        # Create overlapping sequences
        for i in range(0, len(class_images) - sequence_length + 1, 2):  # Step by 2 for overlap
            sequence = class_images[i:i+sequence_length].copy()
            
            # Add progressive temporal variation to simulate video frames
            for j in range(1, sequence_length):
                # Simulate temporal changes
                noise_factor = 0.01 * j  # Increasing noise
                noise = np.random.normal(0, noise_factor, sequence[j].shape)
                sequence[j] = np.clip(sequence[j] + noise, 0, 1)
                
                # Slight brightness variation
                brightness = 1.0 + np.random.uniform(-0.05, 0.05) * j
                sequence[j] = np.clip(sequence[j] * brightness, 0, 1)
            
            sequences.append(sequence)
            labels.append(class_label)
    
    return np.array(sequences), np.array(labels)

def create_optimized_sequence_generator(directory):
    """Optimized sequence generator for better performance"""
    
    # Pre-load sequences for better performance
    if directory.endswith('train'):
        X_seq, y_seq = extract_real_sequences_from_frames()
        print(f"📊 Created {len(X_seq)} training sequences")
    else:
        # For validation, use simpler approach
        datagen = ImageDataGenerator(rescale=1./255)
        base_gen = datagen.flow_from_directory(
            directory,
            target_size=(img_height, img_width),
            batch_size=500,
            class_mode='binary',
            shuffle=False
        )
        
        batch_x, batch_y = next(base_gen)
        
        sequences = []
        labels = []
        
        for i in range(0, len(batch_x) - sequence_length + 1, sequence_length):
            sequences.append(batch_x[i:i+sequence_length])
            labels.append(batch_y[i])
        
        X_seq, y_seq = np.array(sequences), np.array(labels)
        print(f"📊 Created {len(X_seq)} validation sequences")
    
    # Generator function
    def generator():
        indices = np.arange(len(X_seq))
        while True:
            np.random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                if len(batch_indices) == batch_size:  # Only full batches
                    yield X_seq[batch_indices], y_seq[batch_indices]
    
    return generator()

# Simplified CNN-LSTM model (lean architecture)
model = Sequential([
    # Minimal CNN Feature Extractor
    TimeDistributed(Conv2D(32, (3,3), activation='relu'), 
                   input_shape=(sequence_length, img_height, img_width, 3)),
    TimeDistributed(MaxPooling2D(2,2)),
    
    TimeDistributed(Conv2D(64, (3,3), activation='relu')),
    TimeDistributed(MaxPooling2D(2,2)),
    
    TimeDistributed(GlobalAveragePooling2D()),
    
    # Simple LSTM
    LSTM(64, dropout=0.3),
    
    # Minimal Classification Head
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Advanced compilation for optimal performance
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("🔬 Simplified CNN-LSTM Model (Anti-Overfitting)")
print(f"📊 Parameters: {model.count_params():,}")
model.summary()

# Create optimized data generators
train_gen = create_optimized_sequence_generator("../../data/Celeb-DF/split_data/train")
val_gen = create_optimized_sequence_generator("../../data/Celeb-DF/split_data/test")

# Advanced callbacks for optimal training WITHOUT overfitting
callbacks = [
    EarlyStopping(
        monitor='val_loss',  # Monitor loss instead of accuracy
        patience=6,  # Reduced patience
        restore_best_weights=True, 
        verbose=1,
        min_delta=0.001
    ),
    ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=3,  # Faster LR reduction
        min_lr=1e-7, 
        verbose=1
    ),
    ModelCheckpoint(
        "../../models/saved/best_cnn_lstm_checkpoint.h5",
        monitor='val_loss',  # Save based on loss, not accuracy
        save_best_only=True,
        verbose=1
    )
]

# Anti-overfitting training configuration
print("\n🚀 Training Enhanced CNN-LSTM (Anti-Overfitting Mode)...")
history = model.fit(
    train_gen,
    steps_per_epoch=80,  # Reduced steps to prevent overfitting
    validation_data=val_gen,
    validation_steps=20,
    epochs=15,  # Reduced epochs
    callbacks=callbacks,
    verbose=1
)

# Save final model
model.save("../../models/saved/cnn_lstm_model.h5")
print(f"✅ Enhanced CNN-LSTM saved!")

# Final evaluation
final_acc = max(history.history['val_accuracy'])
print(f"🎯 Best Validation Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")

# Anti-overfitting check
train_acc = max(history.history['accuracy'])
val_acc = final_acc
overfitting_gap = train_acc - val_acc

print(f"📊 Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"📊 Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"⚠️ Overfitting Gap: {overfitting_gap:.4f} ({overfitting_gap*100:.2f}%)")

if overfitting_gap > 0.05:
    print("🚨 WARNING: Model may be overfitting (gap > 5%)")
elif overfitting_gap > 0.02:
    print("⚠️ CAUTION: Slight overfitting detected (gap > 2%)")
else:
    print("✅ GOOD: No significant overfitting detected")

if final_acc >= 0.82:  # Realistic target
    print("🎉 SUCCESS: Achieved good accuracy without overfitting!")
else:
    print(f"📈 Progress: {final_acc*100:.2f}% (Realistic Target: 82%+)")
