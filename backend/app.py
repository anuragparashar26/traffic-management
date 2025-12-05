"""Flask app exposing video upload and traffic optimization endpoints.

Public endpoint:
- POST /upload with form-data field 'videos' containing exactly 4 files
    Returns JSON with optimized green times.
- POST /detect_helmets with form-data field 'video' containing 1 file
    Returns JSON with helmet detection counts.
"""

from __future__ import annotations

import os
from typing import List

from flask import Flask, jsonify, request, send_from_directory, Response
from flask import stream_with_context
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor

from traffic_detection import (
    detect_cars,
    optimize_traffic,
    detect_helmets,
    stream_car_frames,
)

app = Flask(__name__)
CORS(app)

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files (violation images)."""
    return send_from_directory('static', path)


@app.route('/violations/<path:path>')
def serve_violations(path):
    """Serve violation images saved under static/violations."""
    return send_from_directory(os.path.join('static', 'violations'), path)

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle upload of exactly four videos and return optimized timings."""
    files = request.files.getlist('videos')
    if len(files) != 4:
        return jsonify({'error': 'Please upload exactly 4 videos'}), 400

    upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    os.makedirs(upload_dir, exist_ok=True)

    video_paths: List[str] = []
    for i, file in enumerate(files):
        video_path = os.path.join(upload_dir, f'video_{i}.mp4')
        file.save(video_path)
        video_paths.append(video_path)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(detect_cars, video_file) for video_file in video_paths]
        num_cars_list: List[float] = [f.result() for f in futures]

    result = optimize_traffic(num_cars_list)

    car_counts = {
        'north': num_cars_list[0],
        'south': num_cars_list[1],
        'west': num_cars_list[2],
        'east': num_cars_list[3],
    }

    payload = {
        **result,
        'car_counts': car_counts,
    }

    return jsonify(payload)

@app.route('/detect_helmets', methods=['POST'])
def detect_helmets_endpoint():
    """Handle upload of one video and return helmet detection results."""
    file = request.files.get('video')
    if not file:
        return jsonify({'error': 'Please upload a video'}), 400

    upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    video_path = os.path.join(upload_dir, 'helmet_video.mp4')
    file.save(video_path)

    result = detect_helmets(video_path)

    return jsonify(result)


@app.route('/stream/<int:index>')
def stream_video(index: int):
    """Stream annotated frames for a previously uploaded video (direction index)."""
    upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    video_path = os.path.join(upload_dir, f'video_{index}.mp4')
    if not os.path.exists(video_path):
        return jsonify({'error': f'video_{index}.mp4 not found. Run optimization upload first.'}), 404

    generator = stream_with_context(stream_car_frames(video_path))
    return Response(generator, mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    app.run(debug=True)
