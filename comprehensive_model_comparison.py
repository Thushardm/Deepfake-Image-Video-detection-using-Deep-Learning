import tensorflow as tf
import numpy as np
import time
import os
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, precision_recall_fscore_support
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_test_data():
    """Load consistent test dataset"""
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_data = test_datagen.flow_from_directory(
        'data/Celeb-DF/split_data/test',
        target_size=(224, 224),
        batch_size=100,
        class_mode='binary',
        shuffle=False
    )
    
    # Get all test data
    all_images = []
    all_labels = []
    
    for i in range(len(test_data)):
        batch_x, batch_y = next(test_data)
        all_images.append(batch_x)
        all_labels.append(batch_y)
    
    images = np.concatenate(all_images, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    print(f"📊 Test dataset loaded: {len(images)} samples")
    print(f"   Real: {np.sum(labels == 0)}, Fake: {np.sum(labels == 1)}")
    
    return images, labels

def create_sequences(images, labels, sequence_length=5):
    """Convert images to sequences for hybrid models"""
    sequences = []
    seq_labels = []
    
    # Create sequences from consecutive images
    for i in range(0, len(images) - sequence_length + 1, sequence_length):
        sequence = images[i:i+sequence_length]
        sequences.append(sequence)
        seq_labels.append(labels[i])  # Use first label
    
    return np.array(sequences), np.array(seq_labels)

def evaluate_model(model_path, model_name, test_images, test_labels):
    """Evaluate a single model"""
    if not os.path.exists(model_path):
        return {'name': model_name, 'status': 'NOT_FOUND'}
    
    try:
        model = tf.keras.models.load_model(model_path)
        
        # Determine model type by input shape
        input_shape = model.input_shape
        is_hybrid = len(input_shape) == 5  # (batch, sequence, height, width, channels)
        
        # Prepare appropriate input data
        if is_hybrid:
            test_x, test_y = create_sequences(test_images, test_labels)
            model_type = "Hybrid"
        else:
            test_x, test_y = test_images, test_labels
            model_type = "Standalone"
        
        if len(test_x) == 0:
            return {'name': model_name, 'status': 'NO_DATA'}
        
        print(f"\n🔍 Evaluating {model_name} ({model_type})")
        print(f"   Input shape: {test_x.shape}")
        
        # Inference timing
        start_time = time.time()
        predictions = model.predict(test_x, verbose=0, batch_size=32)
        inference_time = (time.time() - start_time) / len(test_x)
        
        # Convert predictions
        predicted_classes = (predictions > 0.5).astype(int).flatten()
        true_classes = test_y.astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(true_classes, predicted_classes)
        
        # ROC-AUC (handle edge cases)
        try:
            roc_auc = roc_auc_score(true_classes, predictions.flatten())
        except:
            roc_auc = 0.5
        
        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_classes, predicted_classes, average='binary', zero_division=0
        )
        
        # Model specifications
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        params = model.count_params()
        
        result = {
            'name': model_name,
            'type': model_type,
            'status': 'SUCCESS',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'inference_time': inference_time,
            'size_mb': size_mb,
            'parameters': params,
            'test_samples': len(test_x)
        }
        
        print(f"   ✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        return result
        
    except Exception as e:
        return {'name': model_name, 'status': 'ERROR', 'error': str(e)}

def main():
    print("🔍 COMPREHENSIVE MODEL COMPARISON")
    print("="*80)
    
    # Load test data once
    test_images, test_labels = load_test_data()
    
    # Model configurations
    models = [
        ("model_images/saved/cnn_model.h5", "CNN"),
        ("model_images/saved/efficientnet_model.h5", "EfficientNet"),
        ("model_images/saved/resnet50_model.h5", "ResNet50"),
        ("model_images/saved/vgg16_model.h5", "VGG16"),
        ("model_images/saved/cnn_lstm_model.h5", "CNN-LSTM"),
        ("model_images/saved/cnn_bilstm_model.h5", "CNN-BiLSTM"),
        ("model_images/saved/efficientnet_lstm_model.h5", "EfficientNet-LSTM")
    ]
    
    results = []
    
    # Evaluate all models
    for model_path, model_name in models:
        result = evaluate_model(model_path, model_name, test_images, test_labels)
        results.append(result)
    
    # Filter successful results
    successful_results = [r for r in results if r['status'] == 'SUCCESS']
    
    if not successful_results:
        print("❌ No models could be evaluated successfully!")
        for r in results:
            if r['status'] != 'SUCCESS':
                print(f"   {r['name']}: {r['status']}")
        return
    
    # Sort by accuracy
    successful_results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print(f"\n" + "="*100)
    print("📊 COMPREHENSIVE COMPARISON RESULTS")
    print("="*100)
    
    # Main comparison table
    header = f"{'Rank':<6} {'Model':<18} {'Type':<12} {'Accuracy':<10} {'F1':<8} {'ROC-AUC':<8} {'Size(MB)':<10} {'Time(ms)':<10}"
    print(header)
    print("-"*100)
    
    for i, r in enumerate(successful_results, 1):
        rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        row = f"{rank_emoji:<6} {r['name']:<18} {r['type']:<12} {r['accuracy']:<10.4f} {r['f1_score']:<8.4f} {r['roc_auc']:<8.4f} {r['size_mb']:<10.1f} {r['inference_time']*1000:<10.2f}"
        print(row)
    
    # Detailed metrics table
    print(f"\n📈 DETAILED METRICS")
    print("-"*80)
    detail_header = f"{'Model':<18} {'Precision':<10} {'Recall':<10} {'Parameters':<12} {'Samples':<10}"
    print(detail_header)
    print("-"*80)
    
    for r in successful_results:
        detail_row = f"{r['name']:<18} {r['precision']:<10.4f} {r['recall']:<10.4f} {r['parameters']:<12,} {r['test_samples']:<10}"
        print(detail_row)
    
    # Analysis
    print(f"\n🏆 PERFORMANCE ANALYSIS")
    print("-"*50)
    
    best_overall = successful_results[0]
    standalone_models = [r for r in successful_results if r['type'] == 'Standalone']
    hybrid_models = [r for r in successful_results if r['type'] == 'Hybrid']
    
    print(f"🎯 Best Overall: {best_overall['name']} ({best_overall['accuracy']*100:.2f}%)")
    
    if standalone_models:
        best_standalone = max(standalone_models, key=lambda x: x['accuracy'])
        print(f"📱 Best Standalone: {best_standalone['name']} ({best_standalone['accuracy']*100:.2f}%)")
    
    if hybrid_models:
        best_hybrid = max(hybrid_models, key=lambda x: x['accuracy'])
        print(f"🔗 Best Hybrid: {best_hybrid['name']} ({best_hybrid['accuracy']*100:.2f}%)")
    
    # Efficiency analysis
    most_efficient = max(successful_results, key=lambda x: x['accuracy'] / x['size_mb'])
    fastest = min(successful_results, key=lambda x: x['inference_time'])
    smallest = min(successful_results, key=lambda x: x['size_mb'])
    
    print(f"⚡ Most Efficient: {most_efficient['name']} ({most_efficient['accuracy']/most_efficient['size_mb']:.4f} acc/MB)")
    print(f"🚀 Fastest: {fastest['name']} ({fastest['inference_time']*1000:.2f}ms)")
    print(f"📱 Smallest: {smallest['name']} ({smallest['size_mb']:.1f}MB)")
    
    # Model type comparison
    print(f"\n📊 MODEL TYPE COMPARISON")
    print("-"*40)
    
    if standalone_models:
        avg_standalone_acc = np.mean([r['accuracy'] for r in standalone_models])
        avg_standalone_size = np.mean([r['size_mb'] for r in standalone_models])
        print(f"📱 Standalone Models ({len(standalone_models)}):")
        print(f"   Average Accuracy: {avg_standalone_acc:.4f} ({avg_standalone_acc*100:.2f}%)")
        print(f"   Average Size: {avg_standalone_size:.1f}MB")
    
    if hybrid_models:
        avg_hybrid_acc = np.mean([r['accuracy'] for r in hybrid_models])
        avg_hybrid_size = np.mean([r['size_mb'] for r in hybrid_models])
        print(f"🔗 Hybrid Models ({len(hybrid_models)}):")
        print(f"   Average Accuracy: {avg_hybrid_acc:.4f} ({avg_hybrid_acc*100:.2f}%)")
        print(f"   Average Size: {avg_hybrid_size:.1f}MB")
    
    # Deployment recommendations
    print(f"\n🚀 DEPLOYMENT RECOMMENDATIONS")
    print("-"*40)
    print(f"🏆 Production (Best Accuracy): {best_overall['name']}")
    print(f"📱 Mobile/Edge (Lightweight): {smallest['name']}")
    print(f"⚡ Real-time (Fastest): {fastest['name']}")
    print(f"⚖️ Balanced (Efficient): {most_efficient['name']}")
    
    print(f"\n✅ Evaluation completed! Models tested: {len(successful_results)}/{len(models)}")

if __name__ == "__main__":
    main()
