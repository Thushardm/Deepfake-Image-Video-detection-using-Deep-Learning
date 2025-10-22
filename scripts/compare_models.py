import tensorflow as tf
import numpy as np
import time
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def get_model_size(model_path):
    """Get model file size in MB"""
    return os.path.getsize(model_path) / (1024 * 1024)

def evaluate_standalone_model(model_path, model_name):
    """Evaluate standalone models (single image input)"""
    if not os.path.exists(model_path):
        print(f"❌ {model_name}: Model file not found")
        return None
    
    try:
        model = tf.keras.models.load_model(model_path)
        model_size_mb = get_model_size(model_path)
        
        # Standard test data generator
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_data = test_datagen.flow_from_directory(
            '../data/Celeb-DF/split_data/test',
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary',
            shuffle=False
        )
        
        print(f"\n🔍 Evaluating {model_name}...")
        
        # Evaluate model
        test_loss, test_accuracy = model.evaluate(test_data, verbose=0)
        
        # Get predictions
        test_data.reset()
        start_time = time.time()
        predictions = model.predict(test_data, verbose=0)
        inference_time = (time.time() - start_time) / len(predictions)
        
        predicted_classes = (predictions > 0.5).astype(int).flatten()
        true_classes = test_data.classes
        
        # Calculate metrics
        accuracy = accuracy_score(true_classes, predicted_classes)
        roc_auc = roc_auc_score(true_classes, predictions.flatten())
        
        # Classification report
        class_report = classification_report(true_classes, predicted_classes, 
                                           target_names=['Real', 'Fake'], 
                                           output_dict=True, zero_division=0)
        
        results = {
            'name': model_name,
            'type': 'Standalone',
            'accuracy': accuracy,
            'loss': test_loss,
            'roc_auc': roc_auc,
            'precision_fake': class_report['Fake']['precision'],
            'recall_fake': class_report['Fake']['recall'],
            'f1_fake': class_report['Fake']['f1-score'],
            'model_size_mb': model_size_mb,
            'inference_time_sec': inference_time,
            'total_params': model.count_params()
        }
        
        print(f"✅ {model_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")
        return results
        
    except Exception as e:
        print(f"❌ {model_name}: Error - {str(e)}")
        return None

def evaluate_hybrid_model(model_path, model_name):
    """Evaluate hybrid models (sequence input)"""
    if not os.path.exists(model_path):
        print(f"❌ {model_name}: Model file not found")
        return None
    
    try:
        model = tf.keras.models.load_model(model_path)
        model_size_mb = get_model_size(model_path)
        
        print(f"\n🔍 Evaluating {model_name}...")
        
        # Create sequence test data
        test_datagen = ImageDataGenerator(rescale=1./255)
        base_gen = test_datagen.flow_from_directory(
            '../data/Celeb-DF/split_data/test',
            target_size=(224, 224),
            batch_size=160,  # Larger batch for better sampling
            class_mode='binary',
            shuffle=False
        )
        
        # Get balanced test data
        batch_x, batch_y = next(base_gen)
        
        # Create sequences (5 frames per sequence)
        sequences = []
        labels = []
        sequence_length = 5
        
        for i in range(0, len(batch_x), sequence_length):
            if i + sequence_length <= len(batch_x):
                # Create sequence from consecutive images
                sequence = batch_x[i:i+sequence_length]
                sequences.append(sequence)
                labels.append(batch_y[i])  # Use first label of sequence
        
        if len(sequences) == 0:
            print(f"❌ {model_name}: No sequences created")
            return None
        
        test_sequences = np.array(sequences)
        test_labels = np.array(labels)
        
        print(f"   Test sequences: {test_sequences.shape}, Labels: {len(np.unique(test_labels))} classes")
        
        # Evaluate
        start_time = time.time()
        predictions = model.predict(test_sequences, verbose=0)
        inference_time = (time.time() - start_time) / len(test_sequences)
        
        predicted_classes = (predictions > 0.5).astype(int).flatten()
        true_classes = test_labels.astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(true_classes, predicted_classes)
        
        # Safe ROC-AUC calculation
        try:
            if len(np.unique(true_classes)) > 1:
                roc_auc = roc_auc_score(true_classes, predictions.flatten())
            else:
                roc_auc = 0.5
        except:
            roc_auc = 0.5
        
        # Classification report
        try:
            class_report = classification_report(true_classes, predicted_classes, 
                                               target_names=['Real', 'Fake'], 
                                               output_dict=True, zero_division=0)
        except:
            class_report = {
                'Real': {'precision': 0, 'recall': 0, 'f1-score': 0},
                'Fake': {'precision': 0, 'recall': 0, 'f1-score': 0}
            }
        
        results = {
            'name': model_name,
            'type': 'Hybrid',
            'accuracy': accuracy,
            'loss': 0.0,  # Not available for custom evaluation
            'roc_auc': roc_auc,
            'precision_fake': class_report['Fake']['precision'],
            'recall_fake': class_report['Fake']['recall'],
            'f1_fake': class_report['Fake']['f1-score'],
            'model_size_mb': model_size_mb,
            'inference_time_sec': inference_time,
            'total_params': model.count_params()
        }
        
        print(f"✅ {model_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")
        return results
        
    except Exception as e:
        print(f"❌ {model_name}: Error - {str(e)}")
        return None

def main():
    """Run unified comprehensive model comparison"""
    print("🔍 UNIFIED MODEL EVALUATION")
    print("="*60)
    
    # All models with their types
    models_config = [
        # Standalone models
        ("../models/saved/cnn_model.h5", "CNN", "standalone"),
        ("../models/saved/efficientnet_model.h5", "EfficientNet", "standalone"),
        ("../models/saved/resnet50_model.h5", "ResNet50", "standalone"),
        ("../models/saved/vgg16_model.h5", "VGG16", "standalone"),
        # Hybrid models
        ("../models/saved/cnn_lstm_model.h5", "CNN-LSTM", "hybrid"),
        ("../models/saved/cnn_bilstm_model.h5", "CNN-BiLSTM", "hybrid"),
        ("../models/saved/efficientnet_lstm_model.h5", "EfficientNet-LSTM", "hybrid")
    ]
    
    all_results = []
    
    # Evaluate all models
    for model_path, model_name, model_type in models_config:
        if model_type == "standalone":
            result = evaluate_standalone_model(model_path, model_name)
        else:  # hybrid
            result = evaluate_hybrid_model(model_path, model_name)
        
        if result:
            all_results.append(result)
    
    if not all_results:
        print("❌ No models could be evaluated!")
        return
    
    # Comprehensive summary
    print("\n" + "="*100)
    print("📊 FINAL COMPREHENSIVE COMPARISON")
    print("="*100)
    
    # Sort by accuracy
    all_results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    header = f"{'Rank':<6} {'Model':<18} {'Type':<12} {'Accuracy':<10} {'ROC-AUC':<8} {'F1-Fake':<8} {'Size(MB)':<10} {'Time(s)':<10}"
    print(header)
    print("-"*100)
    
    for i, r in enumerate(all_results, 1):
        rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        row = f"{rank_emoji:<6} {r['name']:<18} {r['type']:<12} {r['accuracy']:<10.4f} {r['roc_auc']:<8.4f} {r['f1_fake']:<8.4f} {r['model_size_mb']:<10.1f} {r['inference_time_sec']:<10.4f}"
        print(row)
    
    # Analysis
    print(f"\n🏆 WINNERS:")
    best_overall = all_results[0]
    best_standalone = next((r for r in all_results if r['type'] == 'Standalone'), None)
    best_hybrid = next((r for r in all_results if r['type'] == 'Hybrid'), None)
    
    print(f"   🎯 Best Overall: {best_overall['name']} ({best_overall['accuracy']*100:.2f}%)")
    if best_standalone:
        print(f"   📱 Best Standalone: {best_standalone['name']} ({best_standalone['accuracy']*100:.2f}%)")
    if best_hybrid:
        print(f"   🔗 Best Hybrid: {best_hybrid['name']} ({best_hybrid['accuracy']*100:.2f}%)")
    
    # Efficiency analysis
    most_efficient = max(all_results, key=lambda x: x['accuracy'] / x['model_size_mb'])
    fastest = min(all_results, key=lambda x: x['inference_time_sec'])
    
    print(f"   ⚡ Most Efficient: {most_efficient['name']} ({most_efficient['accuracy']/most_efficient['model_size_mb']:.3f})")
    print(f"   🚀 Fastest: {fastest['name']} ({fastest['inference_time_sec']:.4f}s)")
    
    print(f"\n✅ Evaluation completed! Total models evaluated: {len(all_results)}")

if __name__ == "__main__":
    main()
