from flask import Flask, render_template, request, jsonify
import os
import uuid
import threading
import sys
import datetime
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.video_utils import predict_video
# from blockchain.log_to_blockchain import log_detection_to_chain
from werkzeug.utils import secure_filename
import hashlib
import json

UPLOAD_FOLDER = "uploads"
STORAGE_FOLDER = "stored_inputs"
TRACKING_FILE = "input_tracking.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STORAGE_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB limit

# Store processing results
processing_results = {}

def save_input_tracking(file_info):
    """Save input file tracking information"""
    try:
        # Load existing tracking data
        if os.path.exists(TRACKING_FILE):
            with open(TRACKING_FILE, 'r') as f:
                tracking_data = json.load(f)
        else:
            tracking_data = []
        
        # Add new entry
        tracking_data.append(file_info)
        
        # Save updated data
        with open(TRACKING_FILE, 'w') as f:
            json.dump(tracking_data, f, indent=2)
            
    except Exception as e:
        print(f"⚠️ Tracking save error: {e}")

def store_input_file(original_path, filename, task_id):
    """Store input file for tracking"""
    try:
        # Create unique filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_ext = os.path.splitext(filename)[1]
        stored_filename = f"{timestamp}_{task_id[:8]}{file_ext}"
        stored_path = os.path.join(STORAGE_FOLDER, stored_filename)
        
        # Copy file to storage
        shutil.copy2(original_path, stored_path)
        
        # Get file info
        file_size = os.path.getsize(original_path)
        
        # Create tracking entry
        file_info = {
            "task_id": task_id,
            "original_filename": filename,
            "stored_filename": stored_filename,
            "stored_path": stored_path,
            "file_size": file_size,
            "upload_time": datetime.datetime.now().isoformat(),
            "file_type": "image" if file_ext.lower() in ['.jpg', '.jpeg', '.png'] else "video"
        }
        
        save_input_tracking(file_info)
        return stored_path
        
    except Exception as e:
        print(f"⚠️ Storage error: {e}")
        return None

def process_video_async(task_id, file_path, stored_path):
    """Process video in background thread using EfficientNet"""
    try:
        print(f"🔍 Processing {file_path} with EfficientNet model...")
        
        # Predict using EfficientNet
        label, confidence = predict_video(file_path)
        
        # Generate hash
        with open(file_path, 'rb') as f:
            media_hash = hashlib.sha256(f.read()).hexdigest()
        
        # Log to blockchain
        # tx_hash = log_detection_to_chain(media_hash, label, confidence)
        tx_hash = "blockchain_disabled"
        
        # Update tracking with results
        if os.path.exists(TRACKING_FILE):
            with open(TRACKING_FILE, 'r') as f:
                tracking_data = json.load(f)
            
            for entry in tracking_data:
                if entry["task_id"] == task_id:
                    entry.update({
                        "prediction": label,
                        "confidence": confidence,
                        "media_hash": media_hash,
                        "blockchain_tx": tx_hash,
                        "processing_time": datetime.datetime.now().isoformat()
                    })
                    break
            
            with open(TRACKING_FILE, 'w') as f:
                json.dump(tracking_data, f, indent=2)
        
        # Store result
        processing_results[task_id] = {
            'status': 'completed',
            'model': 'EfficientNet',
            'accuracy': '84.50%',
            'prediction': label,
            'confidence': f"{confidence:.3f}",
            'blockchain_tx': tx_hash,
            'media_hash': media_hash,
            'stored_path': stored_path
        }
        
        print(f"✅ Processing complete: {label} ({confidence:.3f})")
        
    except Exception as e:
        print(f"❌ Processing error: {e}")
        processing_results[task_id] = {
            'status': 'error',
            'error': str(e)
        }
    finally:
        # Clean up temp file (keep stored copy)
        try:
            os.remove(file_path)
        except:
            pass

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=["POST"])
def upload():
    file = request.files["media"]
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    
    # Store input file for tracking
    stored_path = store_input_file(save_path, filename, task_id)
    
    # Mark as processing
    processing_results[task_id] = {'status': 'processing'}
    
    # Start background processing
    thread = threading.Thread(target=process_video_async, args=(task_id, save_path, stored_path))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'task_id': task_id,
        'status': 'processing',
        'message': 'File stored and processing started.',
        'stored': stored_path is not None
    })

@app.route('/upload_sync', methods=["POST"])
def upload_sync():
    """Synchronous upload using EfficientNet model"""
    file = request.files["media"]
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    # Generate task ID for tracking
    task_id = str(uuid.uuid4())
    
    # Store input file
    stored_path = store_input_file(save_path, filename, task_id)

    print(f"🔍 Processing {filename} with EfficientNet...")
    label, confidence = predict_video(save_path)
    
    media_hash = hashlib.sha256(open(save_path, 'rb').read()).hexdigest()
    # tx_hash = log_detection_to_chain(media_hash, label, confidence)
    tx_hash = "blockchain_disabled"

    # Update tracking with results
    if os.path.exists(TRACKING_FILE):
        with open(TRACKING_FILE, 'r') as f:
            tracking_data = json.load(f)
        
        for entry in tracking_data:
            if entry["task_id"] == task_id:
                entry.update({
                    "prediction": label,
                    "confidence": confidence,
                    "media_hash": media_hash,
                    "blockchain_tx": tx_hash,
                    "processing_time": datetime.datetime.now().isoformat()
                })
                break
        
        with open(TRACKING_FILE, 'w') as f:
            json.dump(tracking_data, f, indent=2)

    # Clean up temp file
    try:
        os.remove(save_path)
    except:
        pass

    return f"""
    🤖 <strong>EfficientNet Model</strong> (84.50% accuracy)<br>
    ✅ Prediction: <strong>{label}</strong><br>
    🔢 Confidence: <strong>{confidence:.3f}</strong><br>
    📁 Stored: <code>{stored_path}</code><br>
    🔗 Media Hash: <code>{media_hash[:16]}...</code><br>
    ⛓️ Blockchain Tx: <code>{tx_hash}</code>
    """

@app.route('/tracking')
def view_tracking():
    """View all tracked inputs"""
    try:
        if os.path.exists(TRACKING_FILE):
            with open(TRACKING_FILE, 'r') as f:
                tracking_data = json.load(f)
            return jsonify({
                'total_files': len(tracking_data),
                'files': tracking_data
            })
        else:
            return jsonify({'total_files': 0, 'files': []})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status/<task_id>')
def get_status(task_id):
    """Check processing status"""
    if task_id not in processing_results:
        return jsonify({'error': 'Task not found'}), 404
    
    result = processing_results[task_id]
    
    # Clean up completed/error tasks after returning result
    if result['status'] in ['completed', 'error']:
        threading.Timer(1.0, lambda: processing_results.pop(task_id, None)).start()
    
    return jsonify(result)

if __name__ == "__main__":
    print("🚀 Starting Flask app with EfficientNet and input tracking...")
    print(f"📁 Inputs stored in: {STORAGE_FOLDER}")
    print(f"📊 Tracking file: {TRACKING_FILE}")
    app.run(debug=True, threaded=True)
