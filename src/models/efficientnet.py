import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Optimized data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    brightness_range=[0.7, 1.3],
    zoom_range=0.15,
    horizontal_flip=True,
    shear_range=0.1,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory("Celeb-DF/split_data/train", target_size=(224, 224), batch_size=16, class_mode='binary')
val_data = val_datagen.flow_from_directory("Celeb-DF/split_data/test", target_size=(224, 224), batch_size=16, class_mode='binary')

# Optimized EfficientNet architecture
base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))

# Enhanced head with batch normalization
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.6)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Optimized compilation
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Advanced callbacks
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)
]

# Optimized training
class_weight = {0: 1.0, 1: 0.35}
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=25,
    class_weight=class_weight,
    callbacks=callbacks
)

model.save("efficientnet_model.h5")
print(f"Optimized EfficientNet - Final Val Accuracy: {max(history.history['val_accuracy']):.4f}")
