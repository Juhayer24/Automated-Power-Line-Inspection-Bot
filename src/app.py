
"""
The brains of the operation - this is where everything comes together.
Grabs camera feed, looks for hazards, keeps track of what's going on,
and shows you what it's seeing. Also logs everything in case something goes wrong.
"""

import argparse
import csv
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from src.detectors.classic_cv import detect_edges_contours
from src.detectors.yolo import YOLOHazard
from src.io.camera import VideoSource
from src.slim1.ui import draw_bounding_boxes, draw_status_overlay, draw_servo_indicator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HazardStateMachine:
    """Simple state machine with debounce for hazard detection."""
    
    def __init__(self, debounce_frames: int = 3):
        """
        Initialize state machine with debounce.
        
        Args:
            debounce_frames: Number of consecutive frames needed to change state
        """
        self.debounce_frames = debounce_frames
        self.hazard_count = 0
        self.current_state = "SAFE"
        
    def update(self, detections: List[Dict]) -> Tuple[str, float]:
        """
        Update state based on current detections.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Tuple of (state, servo_angle)
            - state: "SAFE" or "HAZARD"
            - servo_angle: Angle for servo (-90 to 90, or 0 for centered)
        """
        if len(detections) > 0:
            self.hazard_count += 1
        else:
            self.hazard_count = 0
            
        # State transition with debounce
        if self.hazard_count >= self.debounce_frames and self.current_state == "SAFE":
            self.current_state = "HAZARD"
        elif self.hazard_count == 0 and self.current_state == "HAZARD":
            self.current_state = "SAFE"
            
        # Calculate servo angle based on detection position
        servo_angle = 0.0
        if len(detections) > 0:
            # Use the first detection to determine servo angle
            det = detections[0]
            x_center = det.get('x', 0) + det.get('w', 0) / 2
            # Map x_center to angle range (-90 to 90)
            servo_angle = (x_center / 640.0 * 180.0) - 90.0
            servo_angle = max(-90.0, min(90.0, servo_angle))
            
        return self.current_state, servo_angle

def setup_output_video(source_fps: float, width: int, height: int, 
                      output_path: str) -> Optional[cv2.VideoWriter]:
    """Initialize video writer if recording is enabled."""
    if not output_path:
        return None
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(
        output_path,
        fourcc,
        source_fps,
        (width, height)
    )

def setup_event_log(output_dir: str) -> Tuple[str, csv.DictWriter]:
    """Initialize CSV event logger."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"events_{timestamp}.csv")
    
    fieldnames = ['timestamp', 'state', 'num_detections', 'servo_angle']
    csv_file = open(log_path, 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    
    return log_path, writer

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Power Line Inspection Bot")
    parser.add_argument("--source", type=str, default="0",
                      help="Video source (camera index or file path)")
    parser.add_argument("--detector", choices=["classic", "yolo"],
                      default="classic", help="Detection method to use")
    parser.add_argument("--yolo-weights", type=str,
                      help="Path to YOLO weights file")
    parser.add_argument("--output", type=str,
                      help="Path for output video file")
    parser.add_argument("--record", action="store_true",
                      help="Record annotated video")
    parser.add_argument("--use-pi", action="store_true",
                      help="Use Raspberry Pi camera")
    parser.add_argument("--yolo-every", type=int, default=3,
                      help="Run YOLO detector every N frames")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.detector == "yolo" and not args.yolo_weights:
        parser.error("--yolo-weights is required when using YOLO detector")
    
    # Initialize detector
    if args.detector == "yolo":
        try:
            detector = YOLOHazard(
                weights_path=args.yolo_weights,
                conf=0.35,
                device="cpu"
            )
            detect_fn = detector.detect
        except Exception as e:
            logger.error(f"Failed to initialize YOLO detector: {e}")
            sys.exit(1)
    else:
        detect_fn = detect_edges_contours
    
    # Initialize video source
    try:
        source = int(args.source) if args.source.isdigit() else args.source
        video = VideoSource(source, use_picamera=args.use_pi)
        video.start()
    except Exception as e:
        logger.error(f"Failed to initialize video source: {e}")
        sys.exit(1)
    
    # Initialize state machine
    state_machine = HazardStateMachine()
    
    # Initialize video writer if recording
    writer = None
    if args.record:
        if not args.output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = f"output/video_{timestamp}.mp4"
        writer = setup_output_video(30.0, 640, 480, args.output)
    
    # Initialize event logging
    log_path, csv_writer = setup_event_log("logs")
    logger.info(f"Logging events to: {log_path}")
    
    # Main processing loop
    frame_count = 0
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break
                
            frame_count += 1
            
            # Run detection
            if args.detector == "yolo":
                if frame_count % args.yolo_every == 0:
                    detections = detect_fn(frame)
                else:
                    detections = []
            else:
                detections = detect_fn(frame)
            
            # Update state machine
            state, servo_angle = state_machine.update(detections)
            
            # Draw visualizations
            draw_bounding_boxes(frame, detections)
            draw_status_overlay(frame, state)
            draw_servo_indicator(frame, servo_angle)
            
            # Display frame
            cv2.imshow("Power Line Inspection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Record if enabled
            if writer:
                writer.write(frame)
            
            # Log event
            csv_writer.writerow({
                'timestamp': datetime.now().isoformat(),
                'state': state,
                'num_detections': len(detections),
                'servo_angle': servo_angle
            })
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Cleanup
        video.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        logger.info("Cleanup complete")

if __name__ == "__main__":
    main()
