# Development Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM
- 50GB+ storage space

### Installation

```bash
# Clone repository
git clone <repository-url>
cd DeepFake-Images-Videos-Detection-and-Authentication

# Create virtual environment
python -m venv .env
source .env/bin/activate  # Linux/Mac
# .env\Scripts\activate   # Windows

# Install dependencies
pip install -r config/requirements.txt
```

## 📊 Dataset Setup

### Option 1: Use Existing Split Data
```bash
# Data is already processed and split
# Located in: data/Celeb-DF/split_data/
# Ready for training: 20,720 train + 5,180 test images
```

### Option 2: Process Raw Videos
```bash
# Extract frames from videos
python scripts/data_preprocessing.py

# Create train/test split
python scripts/create_train_test_split.py
```

## 🏋️ Model Training

### Standalone Models

```bash
# Train CNN (fastest, lightweight)
cd src/models
python basic_cnn.py

# Train EfficientNet (best accuracy)
python efficientnet.py

# Train VGG16
python vgg16.py

# Train ResNet50
python resnet50.py
```

### Hybrid Models (Corrected)

```bash
# Train CNN-LSTM (temporal analysis)
python cnn_lstm.py

# Train CNN-BiLSTM (bidirectional temporal)
python cnn_bilstm.py

# Train EfficientNet-LSTM (advanced hybrid)
python efficientnet_lstm.py
```

## 📈 Model Evaluation

### Comprehensive Comparison
```bash
# Compare all models on same test set
python comprehensive_model_comparison.py
```

### Individual Model Testing
```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load model
model = tf.keras.models.load_model('model_images/saved/efficientnet_model.keras')

# Test on single image
test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
    'data/Celeb-DF/split_data/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Evaluate
loss, accuracy = model.evaluate(test_data)
print(f"Accuracy: {accuracy:.4f}")
```

## 🌐 Web Application

### Start Flask Server
```bash
cd src
python app.py
```

### API Usage

#### Async Processing (Large Files)
```bash
# Upload file
curl -X POST -F "media=@video.mp4" http://localhost:5000/upload

# Response: {"task_id": "uuid", "status": "processing"}

# Check status
curl http://localhost:5000/status/uuid

# Response: {"status": "completed", "prediction": "Fake", "confidence": "0.85"}
```

#### Sync Processing (Small Files)
```bash
curl -X POST -F "media=@image.jpg" http://localhost:5000/upload_sync
```

## 🔧 Configuration

### Model Selection Priority
Edit `src/utils/video_utils.py`:
```python
models = [
    ("../../models/saved/efficientnet_model.keras", "EfficientNet"),  # Best accuracy
    ("../../models/saved/cnn_model.keras", "CNN"),                   # Fastest
    ("../../models/saved/vgg16_model.keras", "VGG16"),               # Balanced
]
```

### Training Parameters
Edit model files to adjust:
```python
# Training configuration
batch_size = 32        # Adjust based on GPU memory
epochs = 20           # Increase for better convergence
learning_rate = 0.001 # Lower for stability

# Class weighting (handle imbalance)
class_weight = {0: 1.0, 1: 0.52}  # Reduce fake class weight
```

## 📁 Project Structure

```
DeepFake-Images-Videos-Detection-and-Authentication/
├── src/
│   ├── models/              # Model implementations
│   │   ├── basic_cnn.py     # CNN model
│   │   ├── efficientnet.py  # EfficientNet model
│   │   ├── cnn_lstm.py      # CNN-LSTM hybrid
│   │   └── ...
│   ├── utils/               # Utility functions
│   │   └── video_utils.py   # Video processing
│   ├── blockchain/          # Blockchain integration
│   └── app.py              # Flask web application
├── data/
│   └── Celeb-DF/           # Dataset
├── model_images/saved/      # Trained models (.keras)
├── scripts/                # Utility scripts
├── config/                 # Configuration files
└── docs/                   # Documentation
```

## 🐛 Troubleshooting

### Common Issues

#### GPU Memory Error
```python
# Reduce batch size in model files
batch_size = 16  # or 8 for limited GPU memory
```

#### Model Loading Error
```python
# Check file format and path
import os
model_path = "model_images/saved/cnn_model.keras"
print(f"File exists: {os.path.exists(model_path)}")

# Try loading with custom objects
model = tf.keras.models.load_model(model_path, compile=False)
```

#### Data Loading Error
```python
# Verify dataset structure
import os
train_path = "data/Celeb-DF/split_data/train"
print(f"Real images: {len(os.listdir(os.path.join(train_path, 'real')))}")
print(f"Fake images: {len(os.listdir(os.path.join(train_path, 'fake')))}")
```

### Performance Optimization

#### For Training
```python
# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Use data prefetching
train_data = train_data.prefetch(tf.data.AUTOTUNE)
```

#### For Inference
```python
# Batch processing for multiple images
predictions = model.predict(batch_images, batch_size=32)

# Use TensorFlow Lite for mobile deployment
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

## 📊 Monitoring Training

### TensorBoard Integration
```python
# Add to model training
from tensorflow.keras.callbacks import TensorBoard

callbacks = [
    TensorBoard(log_dir='logs', histogram_freq=1),
    EarlyStopping(monitor='val_accuracy', patience=6),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]
```

### View Training Progress
```bash
tensorboard --logdir=logs
# Open http://localhost:6006
```

## 🔄 Model Updates

### Retrain Existing Model
```python
# Load and continue training
model = tf.keras.models.load_model('model_images/saved/cnn_model.keras')

# Continue training with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(train_data, epochs=10, validation_data=val_data)
```

### Model Versioning
```python
# Save with version
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model.save(f"model_images/saved/cnn_model_{timestamp}.keras")
```

## 🚀 Deployment

### Production Checklist
- [ ] Model accuracy > 80%
- [ ] Inference time < 5ms per image
- [ ] Memory usage < 2GB
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] Security measures in place

### Docker Deployment
```dockerfile
FROM tensorflow/tensorflow:2.13.0-gpu

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "src/app.py"]
```

## 📞 Support

For issues and questions:
1. Check this development guide
2. Review error logs in `logs/` directory
3. Consult model training outputs
4. Create GitHub issue with detailed error information

---

**Happy coding! 🎉**
