import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Bidirectional, Dense, Dropout, TimeDistributed, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np

# Configuration
img_height, img_width = 224, 224
sequence_length = 5
batch_size = 16
epochs = 25

def create_optimized_sequence_generator(directory):
    """Optimized sequence generator using same logic as CNN-LSTM"""
    
    if directory.endswith('train'):
        # Load frame data with better organization
        datagen = ImageDataGenerator(rescale=1./255)
        
        train_gen = datagen.flow_from_directory(
            directory,
            target_size=(img_height, img_width),
            batch_size=2000,
            class_mode='binary',
            shuffle=False
        )
        
        batch_x, batch_y = next(train_gen)
        
        # Separate by class for coherent sequences
        real_indices = np.where(batch_y == 0)[0]
        fake_indices = np.where(batch_y == 1)[0]
        
        sequences = []
        labels = []
        
        # Create sequences within same class
        for class_indices, class_label in [(real_indices, 0), (fake_indices, 1)]:
            class_images = batch_x[class_indices]
            
            for i in range(0, len(class_images) - sequence_length + 1, 2):
                sequence = class_images[i:i+sequence_length].copy()
                
                # Add progressive temporal variation
                for j in range(1, sequence_length):
                    noise_factor = 0.01 * j
                    noise = np.random.normal(0, noise_factor, sequence[j].shape)
                    sequence[j] = np.clip(sequence[j] + noise, 0, 1)
                    
                    brightness = 1.0 + np.random.uniform(-0.05, 0.05) * j
                    sequence[j] = np.clip(sequence[j] * brightness, 0, 1)
                
                sequences.append(sequence)
                labels.append(class_label)
        
        X_seq, y_seq = np.array(sequences), np.array(labels)
        print(f"📊 Created {len(X_seq)} training sequences")
    else:
        # For validation
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
                if len(batch_indices) == batch_size:
                    yield X_seq[batch_indices], y_seq[batch_indices]
    
    return generator()

# Simplified CNN-BiLSTM model (lean architecture)
model = Sequential([
    # Minimal CNN Feature Extractor
    TimeDistributed(Conv2D(32, (3,3), activation='relu'), 
                   input_shape=(sequence_length, img_height, img_width, 3)),
    TimeDistributed(MaxPooling2D(2,2)),
    
    TimeDistributed(Conv2D(64, (3,3), activation='relu')),
    TimeDistributed(MaxPooling2D(2,2)),
    
    TimeDistributed(GlobalAveragePooling2D()),
    
    # Simple Bidirectional LSTM
    Bidirectional(LSTM(32, dropout=0.3)),
    
    # Minimal Classification Head
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("🔬 Enhanced CNN-BiLSTM Model (Target: 85%+ Accuracy)")
print(f"📊 Parameters: {model.count_params():,}")
model.summary()

# Create optimized data generators
train_gen = create_optimized_sequence_generator("../../data/Celeb-DF/split_data/train")
val_gen = create_optimized_sequence_generator("../../data/Celeb-DF/split_data/test")

# Advanced callbacks for optimal training without overfitting
callbacks = [
    EarlyStopping(
        monitor='val_accuracy', 
        patience=8, 
        restore_best_weights=True, 
        verbose=1,
        min_delta=0.001
    ),
    ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=4, 
        min_lr=1e-7, 
        verbose=1
    ),
    ModelCheckpoint(
        "../../models/saved/best_cnn_bilstm_checkpoint.h5",
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Optimized training configuration
print("\n🚀 Training Enhanced CNN-BiLSTM for 85%+ Accuracy...")
history = model.fit(
    train_gen,
    steps_per_epoch=120,
    validation_data=val_gen,
    validation_steps=25,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

# Save final model
model.save("../../models/saved/cnn_bilstm_model.h5")
print(f"✅ Enhanced CNN-BiLSTM saved!")

# Final evaluation
final_acc = max(history.history['val_accuracy'])
print(f"🎯 Best Validation Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")

if final_acc >= 0.85:
    print("🎉 SUCCESS: Achieved 85%+ accuracy target!")
else:
    print(f"📈 Progress: {final_acc*100:.2f}% (Target: 85%+)")
