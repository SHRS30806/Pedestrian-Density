import os
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pathlib import Path

from real_world_inference import process_video_stream

app = Flask(__name__)
CORS(app) # Allow frontend to talk to backend

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variable to hold the current processing video path
current_video_path = None

@app.route('/upload', methods=['POST'])
def upload_video():
    global current_video_path
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        current_video_path = filepath
        return jsonify({'message': 'Upload successful', 'filename': filename}), 200

@app.route('/stream_video')
def stream_video():
    global current_video_path
    if not current_video_path or not os.path.exists(current_video_path):
        return "No video uploaded", 404
        
    checkpoint = "results/drl_pa.pt"
    if not os.path.exists(checkpoint):
        return "Model checkpoint not found", 404

    # Returning a multipart response which allows the browser to render live frames
    return Response(process_video_stream(current_video_path, checkpoint),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Starting TrafficDRL Backend Server on port 5000...")
    app.run(debug=True, host='0.0.0.0', port=5000)
