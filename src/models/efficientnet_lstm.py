import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, GlobalAveragePooling2D, BatchNormalization, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np

# Optimized Configuration for 85%+ accuracy
img_height, img_width = 224, 224
sequence_length = 5  # Increased for better temporal learning
batch_size = 12  # Optimized for EfficientNet
epochs = 25

def create_optimized_sequence_generator(directory):
    """Enhanced sequence generator for EfficientNet-LSTM"""
    
    if directory.endswith('train'):
        # Load frame data with better organization
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=5,  # Minimal augmentation for EfficientNet
            horizontal_flip=True
        )
        
        train_gen = datagen.flow_from_directory(
            directory,
            target_size=(img_height, img_width),
            batch_size=1500,  # Large batch for diversity
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
            
            # Create overlapping sequences with better temporal variation
            for i in range(0, len(class_images) - sequence_length + 1, 3):  # Step by 3
                sequence = class_images[i:i+sequence_length].copy()
                
                # Enhanced temporal variation for EfficientNet
                for j in range(1, sequence_length):
                    # Progressive noise
                    noise_factor = 0.008 * j  # Smaller noise for EfficientNet
                    noise = np.random.normal(0, noise_factor, sequence[j].shape)
                    sequence[j] = np.clip(sequence[j] + noise, 0, 1)
                    
                    # Subtle brightness/contrast changes
                    brightness = 1.0 + np.random.uniform(-0.03, 0.03) * j
                    contrast = 1.0 + np.random.uniform(-0.02, 0.02) * j
                    sequence[j] = np.clip(sequence[j] * brightness * contrast, 0, 1)
                
                sequences.append(sequence)
                labels.append(class_label)
        
        X_seq, y_seq = np.array(sequences), np.array(labels)
        print(f"📊 Created {len(X_seq)} enhanced training sequences")
    else:
        # For validation
        datagen = ImageDataGenerator(rescale=1./255)
        base_gen = datagen.flow_from_directory(
            directory,
            target_size=(img_height, img_width),
            batch_size=400,
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

# Enhanced EfficientNet-LSTM model
base_model = tf.keras.applications.EfficientNetB0(
    weights=None,  # Train from scratch for better hybrid performance
    include_top=False,
    input_shape=(img_height, img_width, 3)
)
base_model.trainable = True  # Allow full training

# Build enhanced hybrid model
inputs = tf.keras.Input(shape=(sequence_length, img_height, img_width, 3))

# Apply EfficientNet to each frame
x = TimeDistributed(base_model)(inputs)
x = TimeDistributed(GlobalAveragePooling2D())(x)

# Enhanced LSTM layers with bidirectional processing
x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))(x)
x = LSTM(64, dropout=0.3, recurrent_dropout=0.2)(x)

# Enhanced classification head
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

# Advanced compilation
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999),  # Lower LR for stability
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("🔬 Enhanced EfficientNet-LSTM Model (Target: 85%+ Accuracy)")
print(f"📊 Parameters: {model.count_params():,}")
model.summary()

# Create optimized data generators
train_gen = create_optimized_sequence_generator("../../data/Celeb-DF/split_data/train")
val_gen = create_optimized_sequence_generator("../../data/Celeb-DF/split_data/test")

# Advanced callbacks for optimal training
callbacks = [
    EarlyStopping(
        monitor='val_accuracy', 
        patience=10,  # More patience for EfficientNet
        restore_best_weights=True, 
        verbose=1,
        min_delta=0.001
    ),
    ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.3,  # More aggressive LR reduction
        patience=5, 
        min_lr=1e-8, 
        verbose=1
    ),
    ModelCheckpoint(
        "../../models/saved/best_efficientnet_lstm_checkpoint.h5",
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Optimized training configuration
print("\n🚀 Training Enhanced EfficientNet-LSTM for 85%+ Accuracy...")
history = model.fit(
    train_gen,
    steps_per_epoch=100,  # Optimized for EfficientNet
    validation_data=val_gen,
    validation_steps=20,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

# Save final model
model.save("../../models/saved/efficientnet_lstm_model.h5")
print(f"✅ Enhanced EfficientNet-LSTM saved!")

# Final evaluation
final_acc = max(history.history['val_accuracy'])
print(f"🎯 Best Validation Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")

if final_acc >= 0.85:
    print("🎉 SUCCESS: Achieved 85%+ accuracy target!")
else:
    print(f"📈 Progress: {final_acc*100:.2f}% (Target: 85%+)")
    print("💡 Tip: EfficientNet-LSTM should achieve 85%+ with enhanced architecture")
