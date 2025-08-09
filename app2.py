from flask import Flask, render_template, request
import os
from utils.video_utils import predict_video
from blockchain.log_to_blockchain import log_detection_to_chain
from werkzeug.utils import secure_filename
import hashlib

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template("index2.html")

@app.route('/upload', methods=["POST"])
def upload():
    file = request.files["media"]
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    label, confidence = predict_video(save_path)
    media_hash = hashlib.sha256(open(save_path, 'rb').read()).hexdigest()
    tx_hash = log_detection_to_chain(media_hash, label, confidence)

    return f"""
    ✅ Prediction: <strong>{label}</strong><br>
    🔢 Confidence: {confidence:.2f}<br>
    ⛓️ Blockchain Tx: <code>{tx_hash}</code>
    """

if __name__ == "__main__":
    app.run(debug=True)
