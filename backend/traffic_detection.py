"""Traffic detection and green time optimization utilities.

This module provides:
- A simple genetic algorithm to optimize traffic light green times.
- YOLOv4-tiny based vehicle detection over video streams.
- YOLOv8 based helmet detection for bike riders.

Public API:
- optimize_traffic(cars) -> dict[str, int]
- detect_cars(video_file) -> float
- record_and_detect(video_file, output_file) -> None
- detect_helmets(video_file) -> dict
"""

from __future__ import annotations

import os
import time
import json
import math
import uuid
from collections import deque
from typing import Deque, List, Sequence, Tuple
from datetime import datetime

import cv2 as cv
import numpy as np
from scipy.signal import find_peaks
import cvzone
import torch
from ultralytics import YOLO
from paddleocr import PaddleOCR
from image_to_text import predict_number_plate
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL") or os.getenv("REACT_APP_SUPABASE_URL")
SUPABASE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    or os.getenv("SUPABASE_ANON_KEY")
    or os.getenv("REACT_APP_SUPABASE_ANON_KEY")
    or os.getenv("SUPABASE_KEY")
)
SUPABASE_BUCKET = os.getenv("SUPABASE_VIOLATIONS_BUCKET", "violations")

_supabase_client: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Failed to initialize Supabase client: {exc}")

# --- Genetic Algorithm Section ---
def fitness_function(C: float, g: float, x: float, c: float) -> float:
    """Compute delay fitness for a single approach.

    Args:
        C: Cycle time.
        g: Green time for the approach.
        x: Degree of saturation/congestion (0..1 typical).
        c: Capacity parameter for the approach.

    Returns:
        Aggregate delay metric to be minimized.
    """
    # Clamp to avoid divide-by-zero and negative roots when congestion spikes
    x = float(np.clip(x, 0.0, 1.5))
    c = max(float(c), 1.0)
    g_over_C = max(g / C, 1e-3)
    p = max(1 - (g_over_C * x), 1e-3)

    a = (1 - g_over_C) ** 2
    d1i = (0.38 * C * a) / p
    a2 = 173 * (x ** 2)
    ri1 = np.sqrt(max((x - 1) + (x - 1) ** 2 + ((16 * x) / c), 0.0))
    d2i = a2 * ri1
    return d1i + d2i

def initialize_population(
    pop_size: int,
    num_lights: int,
    green_min: int,
    green_max: int,
    cycle_time: int,
    cars: Sequence[float],
) -> List[Tuple[np.ndarray, float]]:
    """Initialize a feasible population sorted by fitness (ascending)."""
    population: List[Tuple[np.ndarray, float]] = []
    road_capacity = [20.0] * num_lights
    road_congestion = np.clip(np.array(cars, dtype=float) / np.array(road_capacity), 0.0, 2.0)
    while len(population) < pop_size:
        green_times = np.random.randint(green_min, green_max + 1, num_lights)
        if np.sum(green_times) <= cycle_time:
            total_delay = np.sum([
                cars[i] * fitness_function(cycle_time, green_times[i], road_congestion[i], road_capacity[i])
                for i in range(num_lights)
            ])
            population.append((green_times, total_delay))
    return sorted(population, key=lambda x: x[1])

def roulette_wheel_selection(population: Sequence[Tuple[np.ndarray, float]], total_delays: Sequence[float], beta: float) -> int:
    worst_delay = max(total_delays)
    probabilities = np.exp(-beta * np.array(total_delays) / worst_delay)
    probabilities /= np.sum(probabilities)
    return np.random.choice(len(population), p=probabilities)

def crossover(parent1: np.ndarray, parent2: np.ndarray, num_lights: int) -> Tuple[np.ndarray, np.ndarray]:
    point = np.random.randint(1, num_lights)
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2

def mutate(individual: np.ndarray, mutation_rate: float, green_min: int, green_max: int) -> np.ndarray:
    num_lights = len(individual)
    mutated = individual.copy()
    for _ in range(int(mutation_rate * num_lights)):
        idx = np.random.randint(0, num_lights)
        sigma = np.random.choice([-1, 1]) * 0.02 * (green_max - green_min)
        mutated[idx] = np.clip(individual[idx] + sigma, green_min, green_max)
    return mutated

def inversion(individual: np.ndarray, num_lights: int) -> np.ndarray:
    idx1, idx2 = np.random.randint(0, num_lights, 2)
    if idx1 > idx2:
        idx1, idx2 = idx2, idx1
    individual[idx1:idx2+1] = individual[idx1:idx2+1][::-1]
    return individual


def repair(individual: np.ndarray, green_min: int, green_max: int, cycle_time: int) -> np.ndarray:
    """Clamp bounds and rescale to keep total within cycle time."""
    repaired = np.clip(np.rint(individual), green_min, green_max)
    total = np.sum(repaired)
    if total > cycle_time:
        factor = cycle_time / total
        repaired = np.clip(np.rint(repaired * factor), green_min, green_max)
        total = np.sum(repaired)
        if total > cycle_time:
            surplus = int(total - cycle_time)
            for idx in np.argsort(-repaired): 
                if surplus <= 0:
                    break
                reducible = int(repaired[idx] - green_min)
                take = min(reducible, surplus)
                repaired[idx] -= take
                surplus -= take
    return repaired.astype(int)

def genetic_algorithm(
    pop_size: int,
    num_lights: int,
    max_iter: int,
    green_min: int,
    green_max: int,
    cycle_time: int,
    mutation_rate: float,
    pinv: float,
    beta: float,
    cars: Sequence[float],
) -> Tuple[Tuple[np.ndarray, float], List[float]]:
    population = initialize_population(pop_size, num_lights, green_min, green_max, cycle_time, cars)
    if not population:
        # Fallback to equal split if cars data is unusable
        base = int(cycle_time / num_lights)
        eq = np.array([base] * num_lights)
        return (eq, float('inf')), [float('inf')]

    best_sol = population[0]
    best_delays = [best_sol[1]]
    road_capacity = [20.0] * num_lights
    road_congestion = np.clip(np.array(cars, dtype=float) / np.array(road_capacity), 0.0, 2.0)
    for _ in range(max_iter):
        total_delays = [ind[1] for ind in population]
        new_population = []
        while len(new_population) < pop_size:
            i1 = roulette_wheel_selection(population, total_delays, beta)
            i2 = roulette_wheel_selection(population, total_delays, beta)
            parent1, parent2 = population[i1][0], population[i2][0]
            child1, child2 = crossover(parent1, parent2, num_lights)
            for child in (child1, child2):
                if np.random.rand() < pinv:
                    child = inversion(child, num_lights)
                child = mutate(child, mutation_rate, green_min, green_max)
                child = repair(child, green_min, green_max, cycle_time)
                if np.sum(child) <= cycle_time:
                    total_delay = np.sum([
                        cars[i] * fitness_function(cycle_time, child[i], road_congestion[i], road_capacity[i])
                        for i in range(num_lights)
                    ])
                    new_population.append((child, total_delay))
        while len(new_population) < pop_size:
            i = np.random.randint(0, len(population))
            individual = population[i][0].copy()
            if np.random.rand() < pinv:
                individual = inversion(individual, num_lights)
            individual = mutate(individual, mutation_rate, green_min, green_max)
            individual = repair(individual, green_min, green_max, cycle_time)
            if np.sum(individual) <= cycle_time:
                total_delay = np.sum([
                    cars[i] * fitness_function(cycle_time, individual[i], road_congestion[i], road_capacity[i])
                    for i in range(num_lights)
                ])
                new_population.append((individual, total_delay))
        population += new_population
        population = sorted(population, key=lambda x: x[1])[:pop_size]
        if population[0][1] < best_sol[1]:
            best_sol = population[0]
        best_delays.append(best_sol[1])
        print(f"Iteration: Best Total Delay = {best_sol[1]}")
        print(f"Green Times: North = {best_sol[0][0]}, South = {best_sol[0][1]}, West = {best_sol[0][2]}, East = {best_sol[0][3]}")
    return best_sol, best_delays

def optimize_traffic(cars: Sequence[float]) -> dict:
    """Optimize green times for four approaches using a GA.

    Args:
        cars: A sequence with 4 elements (north, south, west, east) indicating
              congestion proxy values (e.g., detected vehicle counts).

    Returns:
        Dict with integer green times for each approach.
    """
    pop_size = 400
    num_lights = 4
    if len(cars) != num_lights:
        raise ValueError(f"Expected {num_lights} car counts, got {len(cars)}")
    cars = [max(float(c or 0), 0.0) for c in cars]
    max_iter = 25
    green_min = 10
    green_max = 60
    cycle_time = 160 - 12
    mutation_rate = 0.02
    pinv = 0.2
    beta = 8
    best_sol, best_delays = genetic_algorithm(pop_size, num_lights, max_iter, green_min, green_max, cycle_time, mutation_rate, pinv, beta, cars)
    if not best_sol[0].size:
        equal = cycle_time // num_lights
        return {k: equal for k in ['north', 'south', 'west', 'east']}
    result = {
        'north': int(best_sol[0][0]),
        'south': int(best_sol[0][1]),
        'west': int(best_sol[0][2]),
        'east': int(best_sol[0][3])
    }
    print('Optimal Solution:')
    print(f'North Green Time = {result["north"]} seconds')
    print(f'South Green Time = {result["south"]} seconds')
    print(f'West Green Time = {result["west"]} seconds')
    print(f'East Green Time = {result["east"]} seconds')
    return result

# --- YOLOv4 Detection Section ---

# Shared palette for drawing
COLORS: List[Tuple[int, int, int]] = [
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 0),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]


def _resolve_path(filename: str) -> str:
    """Resolve a data file relative to this module."""
    return os.path.join(os.path.dirname(__file__), filename)


def _load_class_names() -> List[str]:
    with open(_resolve_path('classes.txt'), 'r') as f:
        return [cname.strip() for cname in f.readlines()]


_DETECTION_CLASSES: List[str] | None = None


def get_class_names() -> List[str]:
    """Load class names once; safe for multi-threaded use."""
    global _DETECTION_CLASSES
    if _DETECTION_CLASSES is None:
        _DETECTION_CLASSES = _load_class_names()
    return _DETECTION_CLASSES


def build_detection_model() -> cv.dnn_DetectionModel:
    """Create a fresh DetectionModel instance (OpenCV models are not thread-safe)."""
    net = cv.dnn.readNet(_resolve_path('yolov4-tiny.weights'), _resolve_path('yolov4-tiny.cfg'))
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    model = cv.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)
    return model


def upload_image_to_supabase(image: np.ndarray, prefix: str) -> str | None:
    """Upload an image (BGR array) to Supabase Storage and return a public URL."""
    if _supabase_client is None:
        print("Supabase client is not initialized; skipping upload")
        return None

    ok, buffer = cv.imencode('.jpg', image)
    if not ok:
        print("Failed to encode image for upload")
        return None

    storage = _supabase_client.storage.from_(SUPABASE_BUCKET)
    path = f"{prefix}/{datetime.utcnow().strftime('%Y/%m/%d')}/{uuid.uuid4().hex}.jpg"

    try:
        file_opts = {
            "content-type": "image/jpeg",
            "upsert": "true",
            "cache-control": "3600",
        }
        res = storage.upload(path, buffer.tobytes(), file_options=file_opts)
        if res and getattr(res, "error", None):
            print(f"Supabase upload error for {path}: {res.error}")
            return None
        public_url = storage.get_public_url(path)
        return public_url
    except Exception as exc: 
        print(f"Supabase upload failed for {path}: {exc}")
        return None


def detect_cars(video_file: str) -> float:
    Conf_threshold = 0.5
    NMS_threshold = 0.4
    model = build_detection_model()
    class_name = get_class_names()
    allowed_classes = {"car", "bus", "truck"}
    cap = cv.VideoCapture(video_file)
    mean_peak_value: float = 0.0  # default in case video can't be read
    starting_time = time.time()
    frame_counter = 0
    car_counts: Deque[Tuple[float, int]] = deque()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_counter += 1
            classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
            car_count = 0
            for (classid, score, box) in zip(classes, scores, boxes):
                # OpenCV may return classid as array([[id]]) or scalar; normalize to int
                class_id_int = int(classid) if np.isscalar(classid) else int(np.array(classid).item())
                if class_name[class_id_int] in allowed_classes:
                    car_count += 1
                    color = COLORS[class_id_int % len(COLORS)]
                    label = f"{class_name[class_id_int]} : {score:.2f}"
                    cv.rectangle(frame, box, color, 2)
                    cv.putText(frame, label, (box[0], box[1] - 10), cv.FONT_HERSHEY_COMPLEX, 0.5, color, 2)
            current_time = time.time()
            car_counts.append((current_time, car_count))
            # Maintain a 30-second rolling window
            while car_counts and car_counts[0][0] < current_time - 30:
                car_counts.popleft()
            car_count_values = [count for _, count in car_counts]
            peaks, _ = find_peaks(car_count_values)
            if len(peaks) > 0:
                mean_peak_value = float(np.mean([car_count_values[i] for i in peaks]))
            else:
                mean_peak_value = 0.0
            ending_time = time.time()
            fps = frame_counter / (ending_time - starting_time)
            cv.putText(frame, f'FPS: {fps:.2f}', (20, 50), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(frame, f'Mean Peak Cars : {mean_peak_value:.2f}', (20, 80), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2)

    finally:
        cap.release()
        
    return float(mean_peak_value)

def record_and_detect(video_file: str, output_file: str) -> None:
    Conf_threshold = 0.6
    NMS_threshold = 0.4
    model = build_detection_model()
    class_name = get_class_names()
    cap = cv.VideoCapture(video_file)
    frame_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
    dim = (int(frame_width/4), int(frame_height/4))
    print(dim)
    out = cv.VideoWriter(output_file, fourcc, 30.0, dim)
    starting_time = time.time()
    frame_counter = 0
    try:
        while True:
            ret, frame = cap.read()
            frame_counter += 1
            if not ret:
                break
            frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
            classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
            for (classid, score, box) in zip(classes, scores, boxes):
                class_id_int = int(classid) if np.isscalar(classid) else int(np.array(classid).item())
                color = COLORS[class_id_int % len(COLORS)]
                label = "%s : %f" % (class_name[class_id_int], score)
                cv.rectangle(frame, box, color, 1)
                cv.rectangle(frame, (box[0]-2, box[1]-20), (box[0]+120, box[1]-4), (100, 130, 100), -1)
                cv.putText(frame, label, (box[0], box[1]-10), cv.FONT_HERSHEY_COMPLEX, 0.4, color, 1)
            endingTime = time.time() - starting_time
            fps = frame_counter/endingTime
            cv.line(frame, (18, 43), (140, 43), (0, 0, 0), 27)
            cv.putText(frame, f'FPS: {round(fps,2)}', (20, 50), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2)
            cv.imshow('frame', frame)
            out.write(frame)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
    finally:
        out.release()
        cap.release()
        cv.destroyAllWindows()
        print('done')


def detect_helmets(video_file: str) -> dict:
    """Detect helmet violations in video using YOLOv8.
    
    Args:
        video_file: Path to the video file to analyze.
    
    Returns:
        Dict with counts and violation details:
        {
            'helmet': int,
            'no_helmet': int,
            'rider': int,
            'violations': list of violation records
        }
    """
    # Load the YOLOv8 model
    model_path = os.path.join(os.path.dirname(__file__), 'best.pt')
    model = YOLO(model_path)
    
    # Setup device
    device = torch.device("cpu")  # change to cuda for GPU
    
    classNames = ["with helmet", "without helmet", "rider", "number plate"]
    
    # Violation saving setup
    VIOLATION_DIR = os.path.join(os.path.dirname(__file__), "static", "violations")
    VIOLATIONS_JSON = os.path.join(os.path.dirname(__file__), "violations.json")
    os.makedirs(VIOLATION_DIR, exist_ok=True)
    
    def load_violations():
        if os.path.exists(VIOLATIONS_JSON):
            with open(VIOLATIONS_JSON, 'r') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return []
        return []
    
    def save_violations(data):
        with open(VIOLATIONS_JSON, 'w') as f:
            json.dump(data, f, indent=4)
    
    violations_data = load_violations()
    detected_plates_in_session = set(v['plate_text'] for v in violations_data)
    
    # Initialize OCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
    # Open video
    cap = cv.VideoCapture(video_file)
    
    # Counters
    total_helmets = 0
    total_no_helmets = 0
    total_riders = 0
    violations_found = []
    
    while True:
        success, img = cap.read()
        if not success:
            break
            
        new_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = model(new_img, stream=True, device="cpu")
        
        frame_riders = {}  # To hold info about riders in the current frame
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                
                if class_name == "rider" and conf > 0.45:
                    # Use rider's bounding box center as a unique ID for the frame
                    rider_id = f"rider_{(x1+x2)//2}_{(y1+y2)//2}"
                    if rider_id not in frame_riders:
                        frame_riders[rider_id] = {
                            "box": (x1, y1, x2, y2),
                            "has_helmet": True,  # Assume has helmet until "without helmet" is found
                            "plate": None,
                            "plate_confidence": 0
                        }
                    total_riders += 1
                
                elif class_name == "with helmet" and conf > 0.5:
                    total_helmets += 1
                    # Associate with the closest rider
                    for rider_id, rider_info in frame_riders.items():
                        rx1, ry1, rx2, ry2 = rider_info["box"]
                        if x1 > rx1 and y1 > ry1 and x2 < rx2 and y2 < ry2:
                            rider_info["has_helmet"] = True
                
                elif class_name == "without helmet" and conf > 0.5:
                    total_no_helmets += 1
                    # Associate with the closest rider
                    for rider_id, rider_info in frame_riders.items():
                        rx1, ry1, rx2, ry2 = rider_info["box"]
                        if x1 > rx1 and y1 > ry1 and x2 < rx2 and y2 < ry2:
                            rider_info["has_helmet"] = False
                
                elif class_name == "number plate" and conf > 0.5:
                    # Associate with the closest rider
                    for rider_id, rider_info in frame_riders.items():
                        rx1, ry1, rx2, ry2 = rider_info["box"]
                        if x1 > rx1 and y1 > ry1 and x2 < rx2 and y2 < ry2:
                            plate_crop = img[y1:y2, x1:x2]
                            try:
                                vechicle_number, ocr_conf = predict_number_plate(plate_crop, ocr)
                                if vechicle_number and ocr_conf > rider_info["plate_confidence"]:
                                    rider_info["plate"] = {
                                        "box": (x1, y1, x2, y2),
                                        "text": vechicle_number,
                                        "ocr_confidence": ocr_conf,
                                        "crop": plate_crop
                                    }
                                    rider_info["plate_confidence"] = ocr_conf
                            except Exception as e:
                                print(f"OCR Error: {e}")
        
        # Process and save violations
        for rider_id, rider_info in frame_riders.items():
            if not rider_info["has_helmet"] and rider_info["plate"]:
                plate_text = rider_info["plate"]["text"]
                
                if plate_text and plate_text not in detected_plates_in_session:
                    print(f"VIOLATION: Rider without helmet, Plate: {plate_text}")
                    detected_plates_in_session.add(plate_text)
                    
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    
                    # Save rider crop
                    rx1, ry1, rx2, ry2 = rider_info["box"]
                    rider_crop = img[ry1:ry2, rx1:rx2]
                    rider_filename = f"rider_{timestamp}.jpg"
                    cv.imwrite(os.path.join(VIOLATION_DIR, rider_filename), rider_crop)
                    
                    # Save plate crop
                    plate_filename = f"plate_{timestamp}.jpg"
                    cv.imwrite(os.path.join(VIOLATION_DIR, plate_filename), rider_info["plate"]["crop"])

                    # Upload to Supabase Storage (fallback to local path if unavailable)
                    rider_public_url = upload_image_to_supabase(rider_crop, "riders")
                    plate_public_url = upload_image_to_supabase(rider_info["plate"]["crop"], "plates")
                    rider_path = rider_public_url or f"violations/{rider_filename}"
                    plate_path = plate_public_url or f"violations/{plate_filename}"
                    
                    # Create violation record
                    violation_record = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "rider_image": rider_path,
                        "plate_image": plate_path,
                        "plate_text": plate_text,
                        "plate_confidence": rider_info["plate"]["ocr_confidence"]
                    }
                    
                    violations_data.append(violation_record)
                    violations_found.append(violation_record)
                    save_violations(violations_data)
    
    cap.release()
    print("Helmet detection processing finished.")
    
    return {
        'helmet': total_helmets,
        'no_helmet': total_no_helmets,
        'rider': total_riders,
        'violations': violations_found
    }


def stream_car_frames(video_file: str):
    """Yield MJPEG frames with detections for a given video file."""
    Conf_threshold = 0.5
    NMS_threshold = 0.4
    model = build_detection_model()
    class_name = get_class_names()
    allowed_classes = {"car", "bus", "truck"}
    cap = cv.VideoCapture(video_file)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv.resize(frame, (640, 360), interpolation=cv.INTER_AREA)
            classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
            for (classid, score, box) in zip(classes, scores, boxes):
                class_id_int = int(classid) if np.isscalar(classid) else int(np.array(classid).item())
                if class_name[class_id_int] not in allowed_classes:
                    continue
                color = COLORS[class_id_int % len(COLORS)]
                label = f"{class_name[class_id_int]} : {score:.2f}"
                cv.rectangle(frame, box, color, 2)
                cv.putText(frame, label, (box[0], box[1] - 8), cv.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

            ok, buffer = cv.imencode('.jpg', frame)
            if not ok:
                continue

            frame_bytes = buffer.tobytes()
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
    finally:
        cap.release()