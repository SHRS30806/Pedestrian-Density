import os
from flask import Flask, request, Response, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pathlib import Path
import yt_dlp

from real_world_inference import process_video_stream

# Serve the frontend folder as static files
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
app = Flask(__name__, static_folder=frontend_dir)
CORS(app) # Allow frontend to talk to backend

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variable to hold the current processing video path
current_video_path = None

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

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

    return Response(process_video_stream(current_video_path, checkpoint),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream_live')
def stream_live():
    youtube_url = request.args.get('url')
    if not youtube_url:
        return "No URL provided", 400
        
    checkpoint = "results/drl_pa.pt"
    if not os.path.exists(checkpoint):
        return "Model checkpoint not found", 404

    try:
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'quiet': True,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=False)
            stream_url = info_dict.get('url', None)
            
        if not stream_url:
            return "Could not extract stream URL", 500
            
        return Response(process_video_stream(stream_url, checkpoint),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error fetching live stream: {e}")
        return str(e), 500

if __name__ == '__main__':
    print("Starting TrafficDRL Backend Server on port 5000...")
    app.run(debug=True, host='0.0.0.0', port=5000)
