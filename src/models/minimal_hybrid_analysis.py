#!/usr/bin/env python3
"""
OVERFITTING ANALYSIS: Minimal Hybrid Architecture
Addresses excessive complexity in current hybrid models
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, TimeDistributed, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential

def analyze_model_complexity():
    """Analyze parameter counts and overfitting risks"""
    
    print("🔍 OVERFITTING ANALYSIS: HYBRID MODELS")
    print("="*50)
    
    # Current Complex Models
    print("\n📊 CURRENT MODEL COMPLEXITY:")
    
    # 1. Current CNN-LSTM (Improved)
    current_cnn_lstm = Sequential([
        TimeDistributed(Conv2D(32, (3,3), activation='relu'), input_shape=(5, 224, 224, 3)),
        TimeDistributed(MaxPooling2D(2,2)),
        TimeDistributed(Conv2D(64, (3,3), activation='relu')),
        TimeDistributed(MaxPooling2D(2,2)),
        TimeDistributed(GlobalAveragePooling2D()),
        LSTM(64, dropout=0.3),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    print(f"CNN-LSTM Parameters: {current_cnn_lstm.count_params():,}")
    
    # 2. Current CNN-BiLSTM
    current_bilstm = Sequential([
        TimeDistributed(Conv2D(32, (3,3), activation='relu'), input_shape=(5, 224, 224, 3)),
        TimeDistributed(MaxPooling2D(2,2)),
        TimeDistributed(Conv2D(64, (3,3), activation='relu')),
        TimeDistributed(MaxPooling2D(2,2)),
        TimeDistributed(GlobalAveragePooling2D()),
        tf.keras.layers.Bidirectional(LSTM(32, dropout=0.3)),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    print(f"CNN-BiLSTM Parameters: {current_bilstm.count_params():,}")
    
    # 3. EfficientNet-LSTM (Extremely Complex)
    base_efficientnet = tf.keras.applications.EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
    efficientnet_params = base_efficientnet.count_params()
    lstm_params = 128*4*(128+64+1) + 64*4*(64+1+1)  # Approximate LSTM params
    dense_params = 128*64 + 64*1  # Dense layer params
    total_efficientnet_lstm = efficientnet_params + lstm_params + dense_params
    
    print(f"EfficientNet-LSTM Parameters: ~{total_efficientnet_lstm:,}")
    
    print("\n⚠️ OVERFITTING RISK ASSESSMENT:")
    print("CNN-LSTM: MODERATE (improved architecture)")
    print("CNN-BiLSTM: HIGH (bidirectional complexity)")
    print("EfficientNet-LSTM: CRITICAL (4M+ parameters)")
    
    # Minimal Anti-Overfitting Model
    print("\n✅ RECOMMENDED MINIMAL ARCHITECTURE:")
    
    minimal_hybrid = Sequential([
        # Ultra-minimal CNN (1 layer only)
        TimeDistributed(Conv2D(16, (5,5), activation='relu'), input_shape=(5, 224, 224, 3)),
        TimeDistributed(MaxPooling2D(4,4)),  # Aggressive pooling
        TimeDistributed(GlobalAveragePooling2D()),
        
        # Single LSTM
        LSTM(32, dropout=0.5),
        
        # Direct classification
        Dense(1, activation='sigmoid')
    ])
    
    print(f"Minimal Hybrid Parameters: {minimal_hybrid.count_params():,}")
    
    # Performance Comparison
    print("\n📈 PERFORMANCE vs COMPLEXITY:")
    print("Standalone CNN: 81.12% (1.3MB)")
    print("Standalone EfficientNet: 84.50% (5.3MB)")
    print("Current CNN-LSTM: 74.37% (overfitting)")
    print("Current CNN-BiLSTM: 73.12% (overfitting)")
    print("Current EfficientNet-LSTM: 77.78% (severe overfitting)")
    
    print("\n🎯 RECOMMENDATIONS:")
    print("1. Use minimal hybrid architecture (<100K parameters)")
    print("2. Reduce epochs to 10-12 maximum")
    print("3. Focus on data quality over model complexity")
    print("4. Consider ensemble of standalone models instead")
    print("5. Target realistic 75-80% for hybrid models")

if __name__ == "__main__":
    analyze_model_complexity()
