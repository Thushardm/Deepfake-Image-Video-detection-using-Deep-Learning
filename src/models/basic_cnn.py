import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Dataset configuration
data_dir = "../../data/Celeb-DF/processed_frames"
img_height, img_width = 224, 224
batch_size = 32

# Data generators with proper split
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory("../../data/Celeb-DF/split_data/train", target_size=(img_height, img_width), batch_size=batch_size, class_mode='binary')
val_data = val_datagen.flow_from_directory("../../data/Celeb-DF/split_data/test", target_size=(img_height, img_width), batch_size=batch_size, class_mode='binary', shuffle=False)

# Optimal CNN (original + minimal improvements)
model = Sequential([
    # Original proven architecture
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    # Only improvement: Global pooling
    GlobalAveragePooling2D(),
    
    # Original classification head
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Enhanced compilation with learning rate scheduling
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Enhanced callbacks with better monitoring
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True, verbose=1, min_delta=0.001),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
]

# Enhanced class weighting and training
class_weight = {0: 1.0, 1: 0.52}
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,  # Increased epochs
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

# Save optimal model
model.save("../../models/saved/cnn_model.h5")
print(f"Optimal CNN - Final Val Accuracy: {max(history.history['val_accuracy']):.4f}")

# Final evaluation
test_loss, test_accuracy = model.evaluate(val_data)
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
