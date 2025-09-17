"""Traffic detection and green time optimization utilities.

This module provides:
- A simple genetic algorithm to optimize traffic light green times.
- YOLOv4-tiny based vehicle detection over video streams.
- YOLOv8 based helmet and bike rider detection for safety compliance.

Public API:
- optimize_traffic(cars) -> dict[str, int]
- detect_cars(video_file) -> float
- detect_helmets(video_file) -> dict[str, int]
- record_and_detect(video_file, output_file) -> None
"""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Deque, List, Sequence, Tuple

import cv2 as cv
import numpy as np
from scipy.signal import find_peaks
from ultralytics import YOLO

# Check for CUDA availability
try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except Exception:
    _HAS_CUDA = False

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
    cv.namedWindow('frame', cv.WINDOW_NORMAL)
    cv.setWindowProperty('frame', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
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
            cv.imshow('frame', frame)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
    finally:
        cap.release()
        cv.destroyAllWindows()
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

# --- YOLOv8 Helmet Detection Section ---

def _load_helmet_model() -> YOLO:
    """Load the YOLOv8 model for helmet detection."""
    model_path = _resolve_path('weights/best.pt')
    try:
        model = YOLO(model_path)
        print(f"Model loaded successfully from {model_path}")
        print(f"Model classes: {model.names}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def _norm_name(name: str) -> str:
    """Normalize class name for consistent matching."""
    return name.lower().replace('-', ' ').strip()

def _iou(box1: Tuple[int,int,int,int], box2: Tuple[int,int,int,int]) -> float:
    """Compute Intersection over Union (IoU) for two bounding boxes."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def detect_helmets(video_file: str) -> dict[str, int]:
    """Continuous helmet/rider detection with tracking and robust no-helmet inference."""
    model = _load_helmet_model()
    cap = cv.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_file}")
        return {'helmet': 0, 'no_helmet': 0, 'rider': 0}

    # Stable, per-track voting to avoid oscillations
    track_state: dict[int, dict] = {}  # tid -> {'frames', 'helmet_votes', 'no_helmet_votes', 'rider_votes', 'last_box'}
    helmet_ids: set[int] = set()
    no_helmet_ids: set[int] = set()
    rider_ids: set[int] = set()

    # Tunables
    min_track_frames = 6
    min_rider_votes = 2
    min_helmet_votes = 3  # increased to reduce false helmet commits
    min_no_helmet_votes = 3
    helmet_conf_min = 0.60  # only trust helmet boxes >= 0.60
    no_helmet_conf_min = 0.55

    # Precompute name variants
    model_names = {cid: _norm_name(n) for cid, n in getattr(model, 'names', {}).items()}
    has_explicit_nohelmet = any(
        kw in model_names.values()
        for kw in ('no helmet', 'without helmet', 'nohelmet')
    )
    print(f"Has explicit no-helmet class: {has_explicit_nohelmet}")

    cv.namedWindow('Helmet Detection', cv.WINDOW_NORMAL)
    cv.setWindowProperty('Helmet Detection', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    start = time.time()
    frames = 0
    debug_mode = True

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1

            H, W = frame.shape[:2]
            min_area = max(900, int(0.0004 * W * H))  # dynamic area threshold

            # Helpers (local)
            def box_area(b): return max(0, b[2] - b[0]) * max(0, b[3] - b[1])
            def center(b): return ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)
            def head_region(rbox):
                x1, y1, x2, y2 = rbox
                return (x1, y1, x2, y1 + int(0.5 * (y2 - y1)))  # top half
            def helmet_on_head(rbox, hbox) -> bool:
                cx, cy = center(hbox)
                x1, y1, x2, y2 = head_region(rbox)
                inside = (x1 <= cx <= x2) and (y1 <= cy <= y2)
                return inside and _iou(rbox, hbox) > 0.05

            results = model.track(
                frame,
                conf=0.5,
                iou=0.7,
                imgsz=640,
                persist=True,
                tracker='bytetrack.yaml',
                device=(0 if _HAS_CUDA else 'cpu'),
                verbose=False
            )

            # Per-frame collections
            helmet_boxes: List[Tuple[Tuple[int,int,int,int], float]] = []  # (box, conf)
            motorbike_boxes: List[Tuple[int,int,int,int]] = []
            rider_candidates: List[Tuple[int, Tuple[int,int,int,int], str]] = []  # (tid, rbox, src)
            person_candidates: List[Tuple[int, Tuple[int,int,int,int]]] = []
            no_helmet_candidates: List[Tuple[Tuple[int,int,int,int], float]] = []  # (box, conf)

            # First pass: gather boxes and update track states (no direct helmet votes here)
            for res in results:
                names = getattr(res, 'names', model_names)
                for b in res.boxes:
                    cls_id = int(b.cls.item())
                    cname = _norm_name(names.get(cls_id, str(cls_id)))
                    conf = float(b.conf.item()) if hasattr(b, 'conf') else 0.0
                    xyxy = tuple(map(int, b.xyxy[0].tolist()))
                    tid = int(b.id.item()) if getattr(b, 'id', None) is not None else -1

                    if box_area(xyxy) < min_area:
                        continue

                    if tid >= 0:
                        st = track_state.setdefault(tid, {'frames': 0, 'helmet_votes': 0, 'no_helmet_votes': 0, 'rider_votes': 0, 'last_box': xyxy})
                        st['frames'] += 1
                        st['last_box'] = xyxy

                    is_helmet = ('helmet' in cname) and not any(k in cname for k in ('no helmet', 'without helmet', 'nohelmet'))
                    is_nohelmet = any(k in cname for k in ('no helmet', 'without helmet', 'nohelmet'))
                    is_rider_label = any(k in cname for k in ('rider', 'motorcyclist'))
                    is_person = cname == 'person'
                    is_bike = cname in ('motorbike', 'motorcycle')

                    if is_helmet:
                        helmet_boxes.append((xyxy, conf))  # collect only; associate later
                    elif is_nohelmet:
                        no_helmet_candidates.append((xyxy, conf))  # collect only

                    if is_rider_label and tid >= 0:
                        track_state[tid]['rider_votes'] += 1
                        rider_candidates.append((tid, xyxy, 'label:rider'))
                    elif is_person:
                        person_candidates.append((tid, xyxy))

                    if is_bike:
                        motorbike_boxes.append(xyxy)

            # Associate persons with motorbikes to infer riders (adds rider vote)
            for tid, pbox in person_candidates:
                if tid < 0:
                    continue
                if any(_iou(pbox, m) > 0.1 for m in motorbike_boxes):
                    track_state.setdefault(tid, {'frames': 0, 'helmet_votes': 0, 'no_helmet_votes': 0, 'rider_votes': 0, 'last_box': pbox})
                    track_state[tid]['rider_votes'] += 1
                    rider_candidates.append((tid, pbox, 'assoc:person+motorbike'))

            # Associate helmets/no-helmets to rider head regions and cast votes on rider tracks
            if rider_candidates:
                # Build fast lists
                hb = [(b, c) for (b, c) in helmet_boxes if c >= helmet_conf_min]
                nb = [(b, c) for (b, c) in no_helmet_candidates if c >= no_helmet_conf_min]

                for tid, rbox, _src in rider_candidates:
                    if tid < 0:
                        continue
                    # Helmet vote if any helmet is on head region
                    has_helmet = any(helmet_on_head(rbox, hbox) for (hbox, hconf) in hb)
                    if has_helmet:
                        track_state[tid]['helmet_votes'] += 1
                    elif has_explicit_nohelmet:
                        # Only add a no-helmet vote if explicit detection overlaps head region
                        hreg = head_region(rbox)
                        overlaps_nohelmet = any(_iou(hreg, nbox) > 0.10 for (nbox, nconf) in nb)
                        if overlaps_nohelmet:
                            track_state[tid]['no_helmet_votes'] += 1

            # Commit stable decisions per track (only for confirmed riders)
            for tid, st in list(track_state.items()):
                frames_seen = st['frames']
                if frames_seen < min_track_frames:
                    continue

                is_rider = st['rider_votes'] >= min_rider_votes
                if is_rider:
                    rider_ids.add(tid)

                    # Helmet/no-helmet commit rules (avoid flips, and only for riders)
                    if st['helmet_votes'] >= min_helmet_votes and st['helmet_votes'] > st['no_helmet_votes']:
                        helmet_ids.add(tid)
                        no_helmet_ids.discard(tid)
                    elif st['no_helmet_votes'] >= min_no_helmet_votes and st['helmet_votes'] == 0:
                        if tid not in helmet_ids:
                            no_helmet_ids.add(tid)
                else:
                    # Ensure we don't count non-rider tracks
                    helmet_ids.discard(tid)
                    no_helmet_ids.discard(tid)

            # Overlay
            fps = frames / max(1e-3, (time.time() - start))
            cv.putText(frame, f'FPS: {fps:.2f}', (20, 50), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(frame, f'Helmets (tracks): {len(helmet_ids)}', (20, 80), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2)
            cv.putText(frame, f'No Helmets (tracks): {len(no_helmet_ids)}', (20, 110), cv.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 0), 2)
            cv.putText(frame, f'Riders (tracks): {len(rider_ids)}', (20, 140), cv.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)

            if debug_mode and frames % 30 == 0:
                print(f"Frame {frames}: riders={len(rider_ids)} helmet={len(helmet_ids)} no_helmet={len(no_helmet_ids)}")

            cv.imshow('Helmet Detection', frame)
            key = cv.waitKey(1)
            if key == ord('q') or key == 27:
                break
    finally:
        cap.release()
        cv.destroyAllWindows()

    # Final summary
    summary = f"Video processed: {frames} frames. Unique tracks -> Helmets={len(helmet_ids)}, No Helmets={len(no_helmet_ids)}, Riders={len(rider_ids)}"
    print(summary)

    return {
        'helmet': len(helmet_ids),
        'no_helmet': len(no_helmet_ids),
        'rider': len(rider_ids)
    }
