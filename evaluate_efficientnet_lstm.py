import tensorflow as tf
import numpy as np
import time
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_test_data():
    """Load test dataset"""
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_data = test_datagen.flow_from_directory(
        'data/Celeb-DF/split_data/test',
        target_size=(224, 224),
        batch_size=100,
        class_mode='binary',
        shuffle=False
    )
    
    all_images = []
    all_labels = []
    
    for i in range(len(test_data)):
        batch_x, batch_y = next(test_data)
        all_images.append(batch_x)
        all_labels.append(batch_y)
    
    images = np.concatenate(all_images, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    return images, labels

def create_sequences_3_frames(images, labels):
    """Create 3-frame sequences for EfficientNet-LSTM"""
    sequences = []
    seq_labels = []
    
    # Create sequences of 3 consecutive frames
    for i in range(0, len(images) - 2, 3):  # Step by 3
        sequence = images[i:i+3]
        sequences.append(sequence)
        seq_labels.append(labels[i])  # Use first label
    
    return np.array(sequences), np.array(seq_labels)

def evaluate_efficientnet_lstm():
    """Evaluate EfficientNet-LSTM with correct input format"""
    
    model_path = "model_images/saved/efficientnet_lstm_model.h5"
    
    print("🔍 EVALUATING EFFICIENTNET-LSTM")
    print("="*50)
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model loaded")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Expected sequence length: {model.input_shape[1]}")
    
    # Load test data
    test_images, test_labels = load_test_data()
    print(f"📊 Test data: {len(test_images)} images")
    
    # Create 3-frame sequences
    test_sequences, seq_labels = create_sequences_3_frames(test_images, test_labels)
    print(f"📊 Created sequences: {test_sequences.shape}")
    print(f"   Real: {np.sum(seq_labels == 0)}, Fake: {np.sum(seq_labels == 1)}")
    
    # Evaluate
    print(f"\n🚀 Running inference...")
    start_time = time.time()
    predictions = model.predict(test_sequences, verbose=1, batch_size=16)
    inference_time = (time.time() - start_time) / len(test_sequences)
    
    # Calculate metrics
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    true_classes = seq_labels.astype(int)
    
    accuracy = accuracy_score(true_classes, predicted_classes)
    roc_auc = roc_auc_score(true_classes, predictions.flatten())
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_classes, predicted_classes, average='binary', zero_division=0
    )
    
    # Results
    print(f"\n📊 RESULTS:")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   ROC-AUC: {roc_auc:.4f}")
    print(f"   Inference Time: {inference_time*1000:.2f}ms per sequence")
    print(f"   Model Size: 55.0MB")
    print(f"   Parameters: {model.count_params():,}")
    
    # Compare with other models
    print(f"\n🏆 COMPARISON WITH TOP MODELS:")
    print(f"   EfficientNet-LSTM: {accuracy*100:.2f}% (Hybrid)")
    print(f"   EfficientNet:      84.50% (Standalone)")
    print(f"   CNN:               81.12% (Standalone)")
    
    if accuracy > 0.84:
        print(f"   🎉 EfficientNet-LSTM WINS! ({accuracy*100:.2f}% > 84.50%)")
    elif accuracy > 0.81:
        print(f"   🥈 EfficientNet-LSTM 2nd place ({accuracy*100:.2f}%)")
    else:
        print(f"   📈 EfficientNet-LSTM needs improvement ({accuracy*100:.2f}%)")

if __name__ == "__main__":
    evaluate_efficientnet_lstm()
