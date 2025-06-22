from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['media']
    filename = file.filename
    # Just save temporarily (real processing comes later)
    file.save(f"uploads/{filename}")
    return f"Received: {filename}"

if __name__ == '__main__':
    app.run(debug=True)
