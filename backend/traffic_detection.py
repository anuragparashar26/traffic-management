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
    a = (1 - (g / C)) ** 2
    p = 1 - ((g / C) * x)
    d1i = (0.38 * C * a) / p
    a2 = 173 * (x ** 2)
    ri1 = np.sqrt((x - 1) + (x - 1) ** 2 + ((16 * x) / c))
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
    road_capacity = [20] * num_lights
    road_congestion = np.array(road_capacity) - np.array(cars)
    road_congestion = road_congestion / np.array(road_capacity)
    while len(population) < pop_size:
        green_times = np.random.randint(green_min, green_max + 1, num_lights)
        if np.sum(green_times) <= cycle_time:
            total_delay = np.sum([fitness_function(cycle_time, green_times[i], road_congestion[i], road_capacity[i]) for i in range(num_lights)])
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
    best_sol = population[0]
    best_delays = [best_sol[1]]
    road_capacity = [20] * num_lights
    road_congestion = np.array(road_capacity) - np.array(cars)
    road_congestion = road_congestion / np.array(road_capacity)
    for _ in range(max_iter):
        total_delays = [ind[1] for ind in population]
        new_population = []
        while len(new_population) < pop_size:
            i1 = roulette_wheel_selection(population, total_delays, beta)
            i2 = roulette_wheel_selection(population, total_delays, beta)
            parent1, parent2 = population[i1][0], population[i2][0]
            child1, child2 = crossover(parent1, parent2, num_lights)
            if np.sum(child1) <= cycle_time:
                child1 = mutate(child1, mutation_rate, green_min, green_max)
                child1 = np.clip(child1, green_min, green_max)
                total_delay = np.sum([fitness_function(cycle_time, child1[i], road_congestion[i], road_capacity[i]) for i in range(num_lights)])
                new_population.append((child1, total_delay))
            if np.sum(child2) <= cycle_time:
                child2 = mutate(child2, mutation_rate, green_min, green_max)
                child2 = np.clip(child2, green_min, green_max)
                total_delay = np.sum([fitness_function(cycle_time, child2[i], road_congestion[i], road_capacity[i]) for i in range(num_lights)])
                new_population.append((child2, total_delay))
        while len(new_population) < pop_size:
            i = np.random.randint(0, len(population))
            individual = inversion(population[i][0], num_lights)
            if np.sum(individual) <= cycle_time:
                individual = mutate(individual, mutation_rate, green_min, green_max)
                total_delay = np.sum([fitness_function(cycle_time, individual[i], road_congestion[i], road_capacity[i]) for i in range(num_lights)])
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
    max_iter = 25
    green_min = 10
    green_max = 60
    cycle_time = 160 - 12
    mutation_rate = 0.02
    pinv = 0.2
    beta = 8
    best_sol, best_delays = genetic_algorithm(pop_size, num_lights, max_iter, green_min, green_max, cycle_time, mutation_rate, pinv, beta, cars)
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


def _load_detection_assets() -> Tuple[cv.dnn_DetectionModel, List[str]]:
    """Load class names and YOLOv4-tiny model.

    Returns:
        (model, class_names)
    """
    with open(_resolve_path('classes.txt'), 'r') as f:
        class_names = [cname.strip() for cname in f.readlines()]

    net = cv.dnn.readNet(_resolve_path('yolov4-tiny.weights'), _resolve_path('yolov4-tiny.cfg'))
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    model = cv.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)
    return model, class_names


def detect_cars(video_file: str) -> float:
    Conf_threshold = 0.4
    NMS_threshold = 0.4
    model, class_name = _load_detection_assets()
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
                if class_name[class_id_int] == "car":
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
    model, class_name = _load_detection_assets()
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
                    
                    # Create violation record
                    violation_record = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "rider_image": f"violations/{rider_filename}",
                        "plate_image": f"violations/{plate_filename}",
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