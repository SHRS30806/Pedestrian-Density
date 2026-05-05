import cv2
import torch
import numpy as np
from ultralytics import YOLO
import sys
from pathlib import Path

# Add python dir to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))
from ppo_agent import PPOAgent
from config import TrafficConfig

# Map YOLO classes to our categories
# COCO classes: 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASSES = [2, 3, 5, 7]
PEDESTRIAN_CLASSES = [0, 1]

def process_video(video_path, output_path, checkpoint_path):
    print("--- TrafficDRL Real-World Inference ---")
    
    # 1. Load PPO Model
    print(f"Loading PPO Agent from {checkpoint_path}...")
    cfg = TrafficConfig()
    agent = PPOAgent(cfg.ppo)
    agent.load(checkpoint_path)


    # 2. Load YOLO Model
    print("Loading YOLOv8 object detection model...")
    yolo_model = YOLO('yolov8n.pt')  # Will download nano model automatically

    # 3. Setup Video Input/Output
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing Video: {width}x{height} @ {fps} FPS")

    # Define simple spatial regions for the 4 queues (mocking an intersection geometry)
    cx, cy = width // 2, height // 2
    
    # Simple state tracking over time
    frame_count = 0
    current_phase = 0 # 0: NS, 1: EW, 2: PED
    frames_since_change = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        frames_since_change += 1

        # Run YOLO inference
        results = yolo_model(frame, verbose=False)
        
        qNS, qEW, peds = 0, 0, 0
        
        # Parse Bounding Boxes
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                if conf < 0.3:
                    continue
                    
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bx, by = (x1 + x2) // 2, (y1 + y2) // 2
                
                color = (0, 255, 0)
                if cls in VEHICLE_CLASSES:
                    # Spatial mapping logic: Is it NS or EW?
                    # For generic video, let's just split diagonally
                    if abs(bx - cx) < abs(by - cy):
                        qNS += 1
                        color = (255, 0, 0) # Blue for NS
                    else:
                        qEW += 1
                        color = (0, 0, 255) # Red for EW
                elif cls in PEDESTRIAN_CLASSES:
                    peds += 1
                    color = (0, 255, 255) # Yellow for Peds
                
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Build 24D State Vector (Mocked for inference)
        # Our PPO expects: [queue_lens (4), wait_times (4), peds (4), ped_waits (4), phase_onehot (8)]
        state = np.zeros(24, dtype=np.float32)
        state[0] = (qNS / 2) / 10.0 # North queue
        state[1] = (qNS / 2) / 10.0 # South queue
        state[2] = (qEW / 2) / 10.0 # East queue
        state[3] = (qEW / 2) / 10.0 # West queue
        
        state[4:8] = state[0:4] * 0.5 # Mock wait times proportional to queue
        
        state[8:12] = (peds / 4) / 10.0 # Spread peds across 4 corners
        state[12:16] = state[8:12] * 0.5 # Mock ped wait times
        
        # One hot phase
        state[16 + current_phase] = 1.0
        state[20] = frames_since_change / 300.0 # Elapsed phase time

        # Get PPO Action every 30 frames (1 second)
        if frames_since_change > 30 and frame_count % 30 == 0:
            ped_waiting = [True] if peds > 0 else [False]
            action, _, _ = agent.select_action(state, ped_waiting=ped_waiting, deterministic=True)
            if action != current_phase:
                current_phase = action
                frames_since_change = 0

        # Draw Overlay
        phase_str = "N-S GREEN" if current_phase == 0 else ("E-W GREEN" if current_phase == 1 else "PEDESTRIAN CROSSING")
        color_str = (0, 255, 0) if current_phase == 0 else ((0, 0, 255) if current_phase == 1 else (0, 255, 255))
        
        # Black background box
        cv2.rectangle(frame, (10, 10), (450, 150), (0, 0, 0), -1)
        cv2.putText(frame, "PPO Agent Real-Time Inference", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Signal Phase: {phase_str}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_str, 2)
        cv2.putText(frame, f"N-S Cars: {qNS} | E-W Cars: {qEW} | Peds: {peds}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"\nSuccess! Annotated video saved to {output_path}")

def process_video_stream(video_path, checkpoint_path):
    print("--- TrafficDRL Live Streaming Inference ---")
    cfg = TrafficConfig()
    agent = PPOAgent(cfg.ppo)
    agent.load(checkpoint_path)

    yolo_model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cx, cy = width // 2, height // 2
    
    frame_count = 0
    current_phase = 0
    frames_since_change = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        frames_since_change += 1

        results = yolo_model(frame, verbose=False)
        qNS, qEW, peds = 0, 0, 0
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < 0.3: continue
                    
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bx, by = (x1 + x2) // 2, (y1 + y2) // 2
                
                color = (0, 255, 0)
                if cls in VEHICLE_CLASSES:
                    if abs(bx - cx) < abs(by - cy):
                        qNS += 1
                        color = (255, 0, 0)
                    else:
                        qEW += 1
                        color = (0, 0, 255)
                elif cls in PEDESTRIAN_CLASSES:
                    peds += 1
                    color = (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        state = np.zeros(24, dtype=np.float32)
        state[0] = (qNS / 2) / 10.0
        state[1] = (qNS / 2) / 10.0
        state[2] = (qEW / 2) / 10.0
        state[3] = (qEW / 2) / 10.0
        state[4:8] = state[0:4] * 0.5
        state[8:12] = (peds / 4) / 10.0
        state[12:16] = state[8:12] * 0.5
        state[16 + current_phase] = 1.0
        state[20] = frames_since_change / 300.0

        if frames_since_change > 30 and frame_count % 30 == 0:
            ped_waiting = [True] if peds > 0 else [False]
            action, _, _ = agent.select_action(state, ped_waiting=ped_waiting, deterministic=True)
            if action != current_phase:
                current_phase = action
                frames_since_change = 0

        phase_str = "N-S GREEN" if current_phase == 0 else ("E-W GREEN" if current_phase == 1 else "PEDESTRIAN CROSSING")
        color_str = (0, 255, 0) if current_phase == 0 else ((0, 0, 255) if current_phase == 1 else (0, 255, 255))
        
        cv2.rectangle(frame, (10, 10), (450, 150), (0, 0, 0), -1)
        cv2.putText(frame, "PPO Agent Real-Time Inference", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Signal Phase: {phase_str}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_str, 2)
        cv2.putText(frame, f"N-S Cars: {qNS} | E-W Cars: {qEW} | Peds: {peds}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        ret_jpg, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

if __name__ == "__main__":
    video_in = "sample_traffic.mp4"
    video_out = "real_world_demo.mp4"
    checkpoint = "results/drl_pa.pt"
    
    if not Path(video_in).exists():
        print(f"Could not find {video_in}")
        sys.exit(1)
        
    if not Path(checkpoint).exists():
        print(f"Could not find checkpoint {checkpoint}")
        sys.exit(1)
        
    process_video(video_in, video_out, checkpoint)
