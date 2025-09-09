# AI - Based Traffic Management and Accident Prevention System

## Overview

This project utilizes computer vision and artificial intelligence for real-time traffic management in urban settings. By analyzing feeds from CCTV cameras at intersections, the system dynamically give optimized traffic signal timings to optimize vehicle flow and reduce congestion. It features a dashboard for visualizing traffic data and system operations.

## Features

- Real-time traffic monitoring using 4 intersection videos
- AI-driven traffic signal optimization
- Dashboard for visualization

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/anuragparashar26/traffic-management.git
   cd traffic-management
   ```
2. Create a new Python environment (recommended):
   ```bash
   python -m venv myenv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
   (or follow the setup instructions for your environment)

## Usage

- Run the main application:
  ```bash
  python main.py
  ```
- To start the frontend:
  ```bash
  cd frontend
  npm install
  npm start
  ```

## Future Updates

- Helmet detection for rider safety
- License plate recognition for traffic violation detection
- Database integration options for saving events and analytics

## License

This project is licensed under the MIT License.