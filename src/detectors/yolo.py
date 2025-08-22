"""
The fancy neural net detector. Uses YOLO because it's fast and actually works.
Wraps the Ultralytics API to make it less of a pain to work with.
Much more reliable than classical CV, but needs a trained model.
"""

import logging
import os
from pathlib import Path
from typing import Union, List, Dict, Optional

import numpy as np
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOHazard:
    """Wrapper for Ultralytics YOLO model with normalized detection interface."""
    
    def __init__(self, weights_path: str, conf: float = 0.35, device: str = "cpu"):
        """
        Initialize YOLO model for hazard detection.
        
        Args:
            weights_path: Path to YOLO weights file (.pt)
            conf: Confidence threshold (0-1)
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.conf = conf
        self.device = device
        self._frame_count = 0
        
        # Validate weights path
        weights_path = str(Path(weights_path).resolve())
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
            
        try:
            logger.info(f"Loading YOLO model from {weights_path} on {device}")
            self.model = YOLO(weights_path)
            self.model.to(device)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {str(e)}")
            
        # Get class names if available
        self.class_names = self.model.names if hasattr(self.model, 'names') else {}
        logger.info(f"Model loaded successfully. Classes: {list(self.class_names.values())}")

    def detect(self, frame: Union[str, np.ndarray], every_n_frames: int = 1) -> List[Dict]:
        """
        Detect hazards in an image frame.
        
        Args:
            frame: BGR numpy array or path to image file
            every_n_frames: Only process every nth frame (1 = process all frames)
            
        Returns:
            List of detections, each a dict with:
            - x, y: Top-left corner coordinates (pixels)
            - w, h: Width and height (pixels)
            - conf: Confidence score (0-1)
            - class_id: Class ID number
            - label: Class name if available
        """
        # Skip frames based on counter
        self._frame_count += 1
        if every_n_frames > 1 and self._frame_count % every_n_frames != 0:
            return []
            
        # Input validation
        if isinstance(frame, str):
            if not os.path.exists(frame):
                raise FileNotFoundError(f"Image file not found: {frame}")
        elif not isinstance(frame, np.ndarray):
            raise TypeError("Frame must be a numpy array or path to image file")
            
        try:
            # Run inference
            results = self.model(frame, conf=self.conf, device=self.device)
            
            # Process results
            detections = []
            if len(results) > 0 and hasattr(results[0], 'boxes'):
                boxes = results[0].boxes
                
                # Convert boxes to normalized format
                for box in boxes:
                    # Get box coordinates (convert to xywh format)
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    w = x2 - x1
                    h = y2 - y1
                    
                    # Get class info
                    class_id = int(box.cls[0])
                    label = self.class_names.get(class_id, f"class_{class_id}")
                    
                    detections.append({
                        'x': x1,
                        'y': y1,
                        'w': w,
                        'h': h,
                        'conf': float(box.conf[0]),
                        'class_id': class_id,
                        'label': label
                    })
                    
            return detections
            
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            return []
            
    def __repr__(self) -> str:
        """String representation of the detector."""
        return f"YOLOHazard(conf={self.conf}, device='{self.device}', classes={len(self.class_names)})"
