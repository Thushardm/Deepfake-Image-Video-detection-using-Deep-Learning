# 📁 Project Structure

```
DeepFake-Images-Videos-Detection-and-Authentication/
├── 📂 src/                          # Source code
│   ├── 📂 models/                   # ML model implementations
│   │   ├── basic_cnn.py            # Basic CNN architecture
│   │   ├── efficientnet.py         # EfficientNet model
│   │   ├── resnet50.py             # ResNet50 model
│   │   └── vgg16.py                # VGG16 model
│   ├── 📂 web/                     # Web application
│   │   ├── app.py                  # Flask web server
│   │   └── templates/              # HTML templates
│   ├── 📂 blockchain/              # Blockchain integration
│   │   ├── log_to_blockchain.py    # Blockchain logging
│   │   ├── contracts/              # Smart contracts
│   │   └── *.json                  # Contract configs
│   ├── 📂 utils/                   # Utility functions
│   │   └── video_utils.py          # Video processing
│   └── 📂 data/                    # Data processing modules
├── 📂 data/                        # Dataset storage
│   ├── 📂 raw/                     # Original datasets
│   │   └── Celeb-DF/              # Celeb-DF dataset
│   └── 📂 processed/               # Processed data
├── 📂 models/                      # Trained models
│   ├── 📂 saved/                   # Final trained models
│   └── 📂 checkpoints/             # Training checkpoints
├── 📂 scripts/                     # Utility scripts
│   ├── data_preprocessing.py       # Data preprocessing
│   ├── model_trainer.py           # Model training
│   ├── compare_models.py          # Model comparison
│   └── *.py                       # Other scripts
├── 📂 docs/                        # Documentation
│   ├── 📂 reports/                 # Project reports
│   ├── 📂 literature/              # Research papers
│   └── Details.pdf                # Main documentation
├── 📂 config/                      # Configuration files
│   ├── requirements.txt           # Python dependencies
│   └── settings.py                # Project settings
├── 📂 notebooks/                   # Jupyter notebooks
├── 📂 tests/                       # Unit tests
├── 📂 logs/                        # Training logs
├── 📂 outputs/                     # Results and visualizations
│   ├── 📂 results/                 # Model results
│   └── 📂 visualizations/          # Charts and plots
├── .gitignore                      # Git ignore rules
└── README.md                       # Project overview
```

## 🎯 Key Benefits

✅ **Organized**: Clear separation of concerns  
✅ **Scalable**: Easy to add new models/features  
✅ **Maintainable**: Logical file grouping  
✅ **Professional**: Industry-standard structure  
✅ **Git-friendly**: Proper ignore patterns  

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r config/requirements.txt

# Train models
python src/models/basic_cnn.py

# Run web app
python src/web/app.py

# Compare models
python scripts/compare_models.py
```