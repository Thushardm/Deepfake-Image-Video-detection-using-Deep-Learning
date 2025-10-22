# DeepFake Images & Videos Detection and Authentication

## 📌 Project Overview

A comprehensive deep learning system for detecting manipulated media content including deepfakes, splicing, and AI-generated images/videos. This project implements multiple state-of-the-art architectures with blockchain-based authenticity verification for real-world deployment scenarios.

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input Media   │───▶│  Model Pipeline  │───▶│ Blockchain Log  │
│ (Images/Videos) │    │   (7 Models)     │    │  (Immutable)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Flask Web API   │
                    │ (Async/Sync)     │
                    └──────────────────┘
```

## 🧠 Model Architectures

### **Standalone Models (Single Image Input)**
| Model | Accuracy | Size | Parameters | Use Case |
|-------|----------|------|------------|----------|
| **EfficientNet** | **84.50%** | 51MB | 4.4M | Production (Best) |
| **CNN** | **81.12%** | 1.3MB | 110K | Mobile/Edge |
| **VGG16** | 76.04% | 60MB | 15M | Transfer Learning |
| **ResNet50** | 67.39% | 97MB | 24M | Research |

### **Hybrid Models (Temporal Sequence Input)**
| Model | Accuracy | Size | Parameters | Sequence Length |
|-------|----------|------|------------|-----------------|
| **CNN-LSTM** | 65.64%* | 1.7MB | 600K | 5 frames |
| **CNN-BiLSTM** | 65.64%* | 2.4MB | 700K | 5 frames |
| **EfficientNet-LSTM** | 65.70%* | 55MB | 4.8M | 5 frames |

*_Original performance - corrected models expected to achieve 75-80%_

## 📊 Dataset Structure

```
data/Celeb-DF/
├── split_data/           # Training Data (80/20 split)
│   ├── train/
│   │   ├── real/         # 7,120 images
│   │   └── fake/         # 13,600 images (65.6% fake ratio)
│   └── test/
│       ├── real/         # 1,780 images  
│       └── fake/         # 3,400 images
├── videos/              # Original video files
│   ├── YouTube-real/    # Authentic videos
│   └── Celeb-synthesis/ # Deepfake videos
└── processed_frames/    # Extracted frames (50 per video)
```

## 🔧 Technical Implementation

### **Data Processing Pipeline**
1. **Video Processing**: Extract 50 frames per video using OpenCV
2. **Preprocessing**: Resize to 224×224, normalize (0-1 range)
3. **Augmentation**: Rotation, brightness, zoom, horizontal flip
4. **Class Balancing**: Weight adjustment (Real:1.0, Fake:0.35-0.52)

### **Training Strategy**
```python
# Standalone Models
batch_size = 32 (CNN) / 16 (EfficientNet)
epochs = 20-25
learning_rate = 0.001 (CNN) / 0.0005 (EfficientNet)
callbacks = [EarlyStopping, ReduceLROnPlateau]

# Hybrid Models (Corrected)
sequence_length = 5 frames
data_generation = sliding_window(step=1)
architecture = TimeDistributed(CNN) + LSTM/BiLSTM
```

### **Model Storage**
- **Format**: `.keras` (recommended) with `.h5` fallback
- **Location**: `model_images/saved/`
- **Loading Priority**: EfficientNet → CNN → VGG16 → ResNet50

## 🌐 Web Application

### **Flask API Endpoints**
- `GET /` - Web interface
- `POST /upload` - Async video processing
- `GET /status/<task_id>` - Check processing status
- `POST /upload_sync` - Sync processing (small files)

### **Features**
- **Async Processing**: Background threads for large files
- **Blockchain Logging**: Immutable detection records
- **Multi-format Support**: Images (.jpg) and videos (.mp4)
- **Real-time Feedback**: Progress tracking and status updates

## ⛓️ Blockchain Integration

```python
# Detection Logging
media_hash = SHA256(file_content)
blockchain_tx = log_detection_to_chain(
    media_hash=media_hash,
    prediction=label,
    confidence=score
)
```

## 📈 Performance Benchmarks

### **Inference Speed**
- CNN: 0.36ms (fastest)
- EfficientNet: 2.25ms (best accuracy)
- Hybrid Models: 3-10ms (temporal analysis)

### **Deployment Recommendations**
- **🏆 Production**: EfficientNet (84.50% accuracy)
- **📱 Mobile/Edge**: CNN (81.12%, 1.3MB)
- **⚡ Real-time**: CNN (fastest inference)
- **🔗 Temporal**: Corrected hybrid models (75-80% expected)

## 🛠️ Technology Stack

### **Core Technologies**
- **Python 3.10+** - Primary language
- **TensorFlow 2.13+** - Deep learning framework
- **Keras** - High-level neural network API
- **OpenCV** - Computer vision operations
- **Flask** - Web framework
- **Web3.py** - Blockchain integration

### **Development Tools**
- **scikit-learn** - ML utilities and metrics
- **NumPy** - Numerical computing
- **Matplotlib/Seaborn** - Visualization
- **Werkzeug** - WSGI utilities

## 🎯 Key Innovations

1. **Corrected Hybrid Training**: Fixed sequence generation for better temporal learning
2. **Multi-Model Architecture**: 7 different approaches for various use cases
3. **Blockchain Verification**: Immutable authenticity records
4. **Production-Ready API**: Async processing with status tracking
5. **Modern Model Format**: `.keras` format for better compatibility

## 🌍 Real-World Applications

- **Social Media Platforms**: Content moderation at scale
- **News Organizations**: Verify media authenticity
- **Legal Systems**: Evidence verification
- **Security Agencies**: Threat detection
- **E-commerce**: Product image verification
- **Healthcare**: Medical image authenticity

## 📚 Research Contributions

- Comprehensive comparison of standalone vs hybrid approaches
- Analysis of temporal sequence generation methods
- Blockchain integration for media authenticity
- Production deployment strategies for deepfake detection

## 👥 Contributors

- [S. Ashwin Reddy](https://github.com/ashcode18) - Model Architecture & Training
- [Sudeep Patil](https://github.com/imsudeeppatil) - Data Processing & Analysis  
- [Thushar D M](https://github.com/Thushardm) - Web Application & Integration
- [Vinayak Rajput](https://github.com/Vinayak-Rajput) - Blockchain & Deployment

## 📄 Documentation

- **[DEVELOPMENT.md](./DEVELOPMENT.md)** - Setup and usage instructions
- **[PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md)** - Detailed file organization
- **[docs/Details.pdf](./docs/Details.pdf)** - Complete technical documentation

## 🛡️ License

This project is intended for academic and research purposes. For commercial deployment or extended use, please contact the contributors.

---

**⭐ Star this repository if you find it useful for your deepfake detection research!**

