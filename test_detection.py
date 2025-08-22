#!/usr/bin/env python3
"""
Test script to run the power line inspection model.
Supports both image and video/webcam input.
"""

import cv2
import numpy as np
import os
import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.detectors.classic_cv import detect_hazards_classic
try:
    from src.detectors.yolo import YOLODetector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO detector not available - continuing with classic detector only")

from src.io.camera import VideoSource

def process_image(image_path, detector_type='classic'):
    """Process a single image."""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Create detectors
    detections = []
    if detector_type == 'classic':
        boxes, lines, mask = detect_hazards_classic(image)
        detections = [{'bbox': box, 'score': None, 'method': 'classic'} for box in boxes]
    elif detector_type == 'yolo':
        if not YOLO_AVAILABLE:
            print("YOLO detector not available - falling back to classic detector")
            boxes, lines, mask = detect_hazards_classic(image)
            detections = [{'bbox': box, 'score': None, 'method': 'classic'} for box in boxes]
        else:
            detector = YOLODetector(weights='yolov8n.pt')  # Using YOLOv8 nano model
            detections = detector.detect(image)
    
    # Draw detections
    for det in detections:
        bbox = det['bbox']
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), 
                     (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        
        if 'score' in det and det['score'] is not None:
            score = f"{det['score']:.2f}"
            cv2.putText(image, score, (int(bbox[0]), int(bbox[1])-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

def process_video(source=0, detector_type='classic'):
    """Process video or webcam feed."""
    video = VideoSource(source, width=640, height=480)
    
    try:
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            # Process frame
            if detector_type == 'classic':
                boxes, lines, mask = detect_hazards_classic(frame)
                detections = [{'bbox': box, 'score': None, 'method': 'classic'} for box in boxes]
            elif detector_type == 'yolo':
                if not YOLO_AVAILABLE:
                    print("YOLO detector not available - falling back to classic detector")
                    boxes, lines, mask = detect_hazards_classic(frame)
                    detections = [{'bbox': box, 'score': None, 'method': 'classic'} for box in boxes]
                else:
                    detector = YOLODetector(weights='yolov8n.pt')
                    detections = detector.detect(frame)
            
            # Draw detections
            for det in detections:
                bbox = det['bbox']
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                
                if 'score' in det and det['score'] is not None:
                    score = f"{det['score']:.2f}"
                    cv2.putText(frame, score, (int(bbox[0]), int(bbox[1])-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Show results
            cv2.imshow('Power Line Inspection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        video.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run power line inspection model')
    parser.add_argument('--input', type=str, default='0',
                        help='Input source. Can be image path, video path, or camera index (default: 0)')
    parser.add_argument('--detector', type=str, choices=['classic', 'yolo'],
                        default='classic', help='Detector type (default: classic)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for processed image/video (optional)')
    
    args = parser.parse_args()
    
    # Check if input is an image file
    if os.path.isfile(args.input) and args.input.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Process image
        result = process_image(args.input, args.detector)
        
        # Show result
        cv2.imshow('Power Line Inspection', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save if output path provided
        if args.output:
            cv2.imwrite(args.output, result)
            print(f"Saved result to {args.output}")
    
    else:
        # Process video/webcam
        try:
            source = int(args.input)  # Try to convert to camera index
        except ValueError:
            source = args.input  # Use as video path
        
        process_video(source, args.detector)
