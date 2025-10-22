from flask import Flask, render_template, request, jsonify
import os
import uuid
import threading
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.video_utils import predict_video
from blockchain.log_to_blockchain import log_detection_to_chain
from werkzeug.utils import secure_filename
import hashlib

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB limit

# Store processing results
processing_results = {}

def process_video_async(task_id, file_path):
    """Process video in background thread"""
    try:
        # Predict
        label, confidence = predict_video(file_path)
        
        # Generate hash
        with open(file_path, 'rb') as f:
            media_hash = hashlib.sha256(f.read()).hexdigest()
        
        # Log to blockchain
        tx_hash = log_detection_to_chain(media_hash, label, confidence)
        
        # Store result
        processing_results[task_id] = {
            'status': 'completed',
            'prediction': label,
            'confidence': f"{confidence:.2f}",
            'blockchain_tx': tx_hash,
            'media_hash': media_hash
        }
        
    except Exception as e:
        processing_results[task_id] = {
            'status': 'error',
            'error': str(e)
        }
    finally:
        # Clean up file
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
    
    # Mark as processing
    processing_results[task_id] = {'status': 'processing'}
    
    # Start background processing
    thread = threading.Thread(target=process_video_async, args=(task_id, save_path))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'task_id': task_id,
        'status': 'processing',
        'message': 'Video processing started. Use /status endpoint to check progress.'
    })

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

@app.route('/upload_sync', methods=["POST"])
def upload_sync():
    """Synchronous upload for smaller files"""
    file = request.files["media"]
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    label, confidence = predict_video(save_path)
    media_hash = hashlib.sha256(open(save_path, 'rb').read()).hexdigest()
    tx_hash = log_detection_to_chain(media_hash, label, confidence)

    # Clean up file
    try:
        os.remove(save_path)
    except:
        pass

    return f"""
    ✅ Prediction: <strong>{label}</strong><br>
    🔢 Confidence: {confidence:.2f}<br>
    ⛓️ Blockchain Tx: <code>{tx_hash}</code>
    """

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
