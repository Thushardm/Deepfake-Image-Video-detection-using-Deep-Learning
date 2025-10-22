# DeepFake Images & Videos Detection and Authentication

## 📌 Project Overview

A production-ready deep learning system for detecting manipulated media content including deepfakes, splicing, and AI-generated images/videos. This project implements **EfficientNet model** achieving **84.50% accuracy** with comprehensive input tracking and Flask web interface.

## 🏗️ Current System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input Media   │───▶│  EfficientNet    │───▶│ Input Tracking  │
│ (Images/Videos) │    │   (84.50%)       │    │  (JSON + Files) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Flask Web API   │
                    │ (Async/Sync)     │
                    └──────────────────┘
```

## 🤖 **Active Model: EfficientNet**

| Metric | Value | Details |
|--------|-------|---------|
| **Accuracy** | **84.50%** | Best performing model |
| **Model Size** | 50.9MB | Production ready |
| **Inference Time** | 2.30ms | Real-time capable |
| **Parameters** | 4.4M | Optimized architecture |
| **Input Format** | 224×224×3 | RGB images |

## 📊 Dataset Information

### **Celeb-DF Dataset Structure:**
```
Total Images: 25,900 (from 590 videos)
├── Training Set (80%): 20,720 images
│   ├── Real: 7,120 images (34.4%)
│   └── Fake: 13,600 images (65.6%)
└── Test Set (20%): 5,180 images
    ├── Real: 1,780 images (34.4%)
    └── Fake: 3,400 images (65.6%)
```

### **Class Mapping:**
- **Real Images**: Model output > 0.5 → "Real"
- **Fake Images**: Model output ≤ 0.5 → "Fake"

## 🌐 Flask Web Application

### **API Endpoints:**
| Endpoint | Method | Purpose | Response |
|----------|--------|---------|----------|
| **`/`** | GET | Web interface | HTML page |
| **`/upload`** | POST | **Async processing** | Task ID + status |
| **`/upload_sync`** | POST | **Sync processing** | Immediate results |
| **`/status/<task_id>`** | GET | **Check progress** | Processing status |
| **`/tracking`** | GET | **View all inputs** | Complete file history |

### **Supported Formats:**
- **Images**: `.jpg`, `.jpeg`, `.png`
- **Videos**: `.mp4`, `.avi`, `.mov` (any OpenCV supported format)

## 📁 Project Structure

```
DeepFake-Images-Videos-Detection-and-Authentication/
├── 📂 src/                          # Source code
│   ├── app.py                      # Flask web server (MAIN)
│   ├── templates/                  # HTML templates
│   ├── 📂 models/                   # ML model implementations
│   │   ├── efficientnet.py        # EfficientNet (ACTIVE)
│   │   ├── basic_cnn.py           # CNN model
│   │   └── *.py                   # Other models
│   ├── 📂 utils/                   # Utility functions
│   │   └── video_utils.py         # Preprocessing & prediction
│   └── 📂 blockchain/              # Blockchain (DISABLED)
├── 📂 data/                        # Dataset storage
│   └── 📂 Celeb-DF/               # Celeb-DF dataset
│       ├── split_data/            # Processed training data
│       └── videos/                # Original videos
├── 📂 model_images/saved/          # Trained models (.h5)
├── 📂 test/                        # Validation samples
│   ├── images/                    # Test images (real/fake)
│   └── videos/                    # Test videos (real/fake)
├── 📂 stored_inputs/               # User uploaded files
├── 📂 uploads/                     # Temporary upload folder
├── input_tracking.json            # Processing history
└── 📂 config/                      # Configuration files
```

## 🚀 Quick Start

### **1. Setup Environment**
```bash
cd DeepFake-Images-Videos-Detection-and-Authentication
source .env/bin/activate  # Activate virtual environment
pip install -r config/requirements.txt
```

### **2. Run Flask Application**
```bash
cd src
python app.py
```

### **3. Access Web Interface**
- Open browser: `http://localhost:5000`
- Upload images or videos for detection
- View results with confidence scores

## 📈 Performance Metrics

### **Model Comparison Results:**
| Rank | Model | Accuracy | Size | Speed | Status |
|------|-------|----------|------|-------|--------|
| 🥇 | **EfficientNet** | **84.50%** | 50.9MB | 2.30ms | **ACTIVE** |
| 🥈 | CNN | 81.12% | 1.3MB | 0.39ms | Available |
| 🥉 | VGG16 | 76.04% | 60.0MB | 3.67ms | Available |
| 4 | ResNet50 | 67.39% | 96.6MB | 3.01ms | Available |

## 🧪 Testing & Validation

### **Test Samples Available:**
- **`test/images/`**: Real and fake image samples
- **`test/videos/`**: Original video samples from dataset
- **Manual Testing**: Upload via web interface

### **Expected Results:**
- **Real Images**: Should predict "Real" with high confidence
- **Fake Images**: Should predict "Fake" with high confidence
- **Processing Time**: < 5 seconds for most files

## 📊 Input Tracking

### **Tracking Features:**
- **File Preservation**: All uploads stored permanently
- **Processing History**: Complete log in `input_tracking.json`
- **Metadata**: Filename, size, upload time, predictions
- **API Access**: View history via `/tracking` endpoint

## 🎯 Key Features

✅ **Production Ready**: Flask web interface with async processing  
✅ **High Accuracy**: 84.50% detection accuracy with EfficientNet  
✅ **Multi-Format**: Supports both images and videos  
✅ **Input Tracking**: Complete audit trail of all processing  
✅ **Real-Time**: Fast inference (2.30ms per image)  
✅ **Scalable**: Threaded processing for concurrent requests  
✅ **User Friendly**: Simple web interface with progress tracking

## 👥 Contributors

- S. Ashwin Reddy
- Sudeep Patil  
- Thushar D M
- Vinayak Rajput

## 📄 Documentation

- **[DEVELOPMENT.md](./DEVELOPMENT.md)** - Setup and usage instructions
- **[PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md)** - Detailed file organization
- **[docs/Details.pdf](./docs/Details.pdf)** - Complete technical documentation

## 🛡️ License

This project is intended for academic and research purposes. For commercial deployment or extended use, please contact the contributors.

---

**⭐ Star this repository if you find it useful for your deepfake detection research!**

