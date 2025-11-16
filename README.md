# AI-Based Traffic Management and Accident Prevention System

## Overview

This project utilizes computer vision and artificial intelligence for real-time traffic management and safety enforcement in urban settings. By analyzing feeds from CCTV cameras at intersections, the system uses a genetic algorithm to compute optimized traffic signal timings for four directions (North, South, East, West) to minimize vehicle delay and reduce congestion. The system includes comprehensive safety features with helmet detection for two-wheeler riders and automated violation tracking with license plate recognition.

## Features

### Traffic Signal Optimization

- **Multi-direction Analysis**: Processes 4 intersection videos simultaneously (North, South, East, West)
- **YOLOv4-tiny Vehicle Detection**: Real-time vehicle counting with peak detection over 30-second rolling windows
- **Genetic Algorithm Optimization**: Computes optimal green light durations (10-60 seconds) within cycle constraints to minimize aggregate delay
- **Interactive Dashboard**: React-based UI for video upload, real-time processing, and visualization of optimized timings

### Safety Enforcement System

- **Helmet Detection**: YOLOv8-based detection of riders, helmets, and no-helmet violations
- **License Plate Recognition**: Automated OCR using PaddleOCR with 90%+ confidence threshold
- **Violation Tracking**:
  - Captures and saves images of violators (rider + license plate)
  - Prevents duplicate violation records for the same plate
  - Stores violation data with timestamps in JSON format
  - Web-accessible violation gallery with confidence scores
- **Multi-class Detection**: Identifies riders, helmets, no-helmets, and license plates

## Technology Stack

### Backend

- **Flask**: REST API server with CORS support
- **OpenCV**: Video processing and frame analysis
- **YOLOv4-tiny**: Fast vehicle detection for traffic counting
- **YOLOv8**: Advanced helmet and license plate detection
- **PaddleOCR**: License plate text recognition
- **NumPy & SciPy**: Numerical computations and peak detection
- **PyTorch**: Deep learning model inference
- **Ultralytics**: YOLOv8 implementation

### Frontend

- **React 18**: Component-based UI framework
- **Axios**: HTTP client for API communication
- **CSS3**: Custom styling with responsive design

### Algorithms

- **Genetic Algorithm**: Population-based optimization for traffic signal timings
- **Fitness Function**: Webster's delay model for traffic flow optimization
- **Peak Detection**: SciPy-based rolling window analysis for vehicle counting

## Installation

### Prerequisites

- Python 3.10+
- Node.js 14+
- Git

### Backend Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/anuragparashar26/traffic-management.git
   cd traffic-management/backend
   ```

2. Create and activate a Python virtual environment:

   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Ensure model files are present:
   - `yolov4-tiny.weights` - Vehicle detection model
   - `yolov4-tiny.cfg` - YOLOv4 configuration
   - `best.pt` - YOLOv8 helmet detection model
   - `classes.txt` - Class labels for detection

### Frontend Setup

1. Navigate to frontend directory:

   ```bash
   cd ../frontend
   ```

2. Install Node dependencies:

   ```bash
   npm install
   ```

3. (Optional) Configure API URL in `.env`:
   ```bash
   REACT_APP_API_URL=http://localhost:5000
   ```

## Usage

### Starting the Application

1. **Start the Flask backend** (from `backend/` directory):

   ```bash
   source myenv/bin/activate  # Activate virtual environment
   python app.py
   ```

   The API will run on `http://localhost:5000`

2. **Start the React frontend** (from `frontend/` directory):
   ```bash
   npm start
   ```
   The dashboard will open at `http://localhost:3000`

### Using the Dashboard

#### Traffic Signal Optimization

1. Click "Select Videos" or drag & drop 4 video files
2. Assign videos to directions (North, South, East, West)
3. Click "Run Optimization"
4. View optimized green light timings for each direction

#### Helmet Detection & Violation Tracking

1. Navigate to "Helmet Detection" section
2. Upload a video file containing two-wheeler traffic
3. Click "Run Detection"
4. View detection statistics (riders, helmets, no-helmets)
5. Review violation cards showing:
   - License plate number with confidence score
   - Captured images of violator and plate
   - Timestamp of violation

## Project Structure

```
traffic-management/
├── backend/
│   ├── app.py                    # Flask REST API server
│   ├── traffic_detection.py     # Core detection & optimization logic
│   ├── image_to_text.py         # OCR utilities for license plates
│   ├── requirements.txt         # Python dependencies
│   ├── yolov4-tiny.weights      # YOLOv4 vehicle detection weights
│   ├── yolov4-tiny.cfg          # YOLOv4 configuration
│   ├── classes.txt              # Detection class labels
│   ├── violations.json          # Violation records database
│   ├── myenv/                   # Python virtual environment
│   ├── uploads/                 # Temporary video upload directory
│   └── static/violations/       # Saved violation images
├── frontend/
│   ├── src/
│   │   ├── App.js               # Main React component
│   │   ├── styles.css           # Dashboard styling
│   │   └── index.js             # React entry point
│   ├── public/
│   └── package.json             # Node dependencies
└── README.md
```

## Algorithm Details

### Traffic Optimization

- **Population Size**: 400 individuals
- **Generations**: 25 iterations
- **Green Time Range**: 10-60 seconds per direction
- **Cycle Time**: 148 seconds (160s - 12s buffer)
- **Mutation Rate**: 2%
- **Selection**: Roulette wheel with exponential fitness scaling
- **Operators**: Crossover, mutation, and inversion

### Detection Thresholds

- **Vehicle Detection**: 60% confidence (YOLOv4)
- **Rider Detection**: 45% confidence (YOLOv8)
- **Helmet Detection**: 50% confidence (YOLOv8)
- **License Plate OCR**: 90% confidence (PaddleOCR)

## Key Features Implemented

✅ Multi-video traffic analysis  
✅ Genetic algorithm optimization  
✅ Real-time vehicle counting  
✅ Helmet violation detection  
✅ License plate recognition  
✅ Automated violation tracking  
✅ Duplicate violation prevention  
✅ Web-based dashboard  
✅ Violation image gallery  
✅ JSON-based violation storage

## Future Enhancements

- Database integration (PostgreSQL/MongoDB) for scalable storage
- Real-time video streaming support
- SMS/Email notifications for violations
- Cloud deployment with scalability

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Dataset

The helmet detection model was trained on the following dataset:

- **Rider with Helmet, Without Helmet & Number Plate Dataset**
  - Source: [Kaggle - Aneesa Rom](https://www.kaggle.com/datasets/aneesarom/rider-with-helmet-without-helmet-number-plate)
  - Contains labeled images of:
    - Riders with helmets
    - Riders without helmets
    - License/number plates
  - Used for training the YOLOv8 model

## Acknowledgments

- YOLOv4 and YOLOv8 models by Ultralytics
- PaddleOCR by PaddlePaddle team
- Genetic algorithm based on Webster's traffic delay model
- React community for frontend libraries
- Helmet detection dataset by Aneesa Rom on Kaggle
