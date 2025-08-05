from flask import Flask, render_template, request
import os
import cv2
from utils.video_utils import predict_video, predict_image
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['media']
    filename = file.filename
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    ext = filename.split('.')[-1].lower()
    if ext in ['jpg', 'jpeg', 'png']:
        # Handle image
        img = cv2.imread(path)
        pred = predict_image(img)
        label = "Real" if pred > 0.5 else "Fake"
        return f"Prediction: {label} (Confidence: {pred:.2f})"
    elif ext in ['mp4', 'avi', 'mov']:
        # Handle video
        label, avg = predict_video(path)
        return f"Video Prediction: {label} (Avg Confidence: {avg:.2f})"
    else:
        return "Unsupported file format."

if __name__ == '__main__':
    app.run(debug=True)
