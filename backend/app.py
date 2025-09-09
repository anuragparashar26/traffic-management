"""Flask app exposing video upload and traffic optimization endpoints.

Public endpoint:
- POST /upload with form-data field 'videos' containing exactly 4 files
    Returns JSON with optimized green times.
"""

from __future__ import annotations

import os
from typing import List

from flask import Flask, jsonify, request
from flask_cors import CORS

from traffic_detection import detect_cars, optimize_traffic

app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle upload of exactly four videos and return optimized timings."""
    files = request.files.getlist('videos')
    if len(files) != 4:
        return jsonify({'error': 'Please upload exactly 4 videos'}), 400

    # Resolve upload directory relative to this file for robustness
    upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    os.makedirs(upload_dir, exist_ok=True)

    video_paths: List[str] = []
    for i, file in enumerate(files):
        video_path = os.path.join(upload_dir, f'video_{i}.mp4')
        file.save(video_path)
        video_paths.append(video_path)

    num_cars_list: List[float] = []
    for video_file in video_paths:
        num_cars = detect_cars(video_file)
        num_cars_list.append(num_cars)

    result = optimize_traffic(num_cars_list)

    return jsonify(result)

if __name__ == '__main__':
    # Ensure upload directory exists when running directly
    upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    app.run(debug=True)
