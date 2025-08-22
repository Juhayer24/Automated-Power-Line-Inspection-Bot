"""
Makes the video feed look nice - draws boxes around stuff we found,
adds status info, and shows where the camera is pointing.
Keep it simple but informative.
"""

import cv2
import numpy as np

def draw_bounding_boxes(frame: np.ndarray, detections: list) -> None:
    """
    Draw detection bounding boxes on frame.
    
    Args:
        frame: BGR numpy array
        detections: List of detection dictionaries with x,y,w,h coordinates
    """
    for det in detections:
        # Get coordinates
        x = det.get('x', 0)
        y = det.get('y', 0)
        w = det.get('w', 0)
        h = det.get('h', 0)
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Draw confidence if available
        if 'conf' in det:
            conf = det['conf']
            cv2.putText(
                frame,
                f"{conf:.2f}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )

def draw_status_overlay(frame: np.ndarray, state: str) -> None:
    """
    Draw status text and LED indicator.
    
    Args:
        frame: BGR numpy array
        state: "SAFE" or "HAZARD"
    """
    # Status text
    color = (0, 255, 0) if state == "SAFE" else (0, 0, 255)
    cv2.putText(
        frame,
        state,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2
    )
    
    # LED indicator
    center = (30, 60)
    radius = 10
    cv2.circle(frame, center, radius, color, -1)
    cv2.circle(frame, center, radius, (255, 255, 255), 1)

def draw_servo_indicator(frame: np.ndarray, angle: float) -> None:
    """
    Draw arrow indicating servo angle.
    
    Args:
        frame: BGR numpy array
        angle: Servo angle in degrees (-90 to 90)
    """
    height = frame.shape[0]
    center = (frame.shape[1] - 50, height - 50)
    
    # Convert angle to radians and calculate arrow end point
    angle_rad = np.radians(angle)
    length = 30
    end_x = int(center[0] + length * np.cos(angle_rad))
    end_y = int(center[1] - length * np.sin(angle_rad))
    
    # Draw arrow
    cv2.arrowedLine(
        frame,
        center,
        (end_x, end_y),
        (255, 255, 0),
        2,
        tipLength=0.3
    )
    
    # Draw angle text
    cv2.putText(
        frame,
        f"{angle:.1f}°",
        (center[0] - 40, center[1] + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        1
    )
