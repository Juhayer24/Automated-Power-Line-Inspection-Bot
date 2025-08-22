import cv2
import numpy as np

def draw_led(frame, is_hazard):
    """Draw a hazard indicator LED in top-left corner.
    
    Args:
        frame: OpenCV BGR image to draw on
        is_hazard: Boolean indicating if hazard is detected
    """
    color = (0, 0, 255) if is_hazard else (0, 255, 0)  # Red if hazard, green if safe
    cv2.circle(frame, (30, 30), 15, color, -1)
    cv2.circle(frame, (30, 30), 15, (255, 255, 255), 2)  # White border

def draw_servo_arrow(frame, angle_deg):
    """Draw an arrow indicating servo angle.
    
    Args:
        frame: OpenCV BGR image to draw on
        angle_deg: Servo angle in degrees
    """
    h, w = frame.shape[:2]
    center = (w - 50, h - 50)  # Bottom right position
    length = 40
    # Convert angle to radians and calculate arrow endpoint
    angle_rad = np.deg2rad(angle_deg - 90)  # Subtract 90 to make 0 degrees point up
    end_point = (
        int(center[0] + length * np.cos(angle_rad)),
        int(center[1] + length * np.sin(angle_rad))
    )
    cv2.arrowedLine(frame, center, end_point, (0, 255, 255), 2, tipLength=0.3)

def draw_bbox_with_label(frame, bbox, label, conf):
    """Draw bounding box with label and confidence.
    
    Args:
        frame: OpenCV BGR image to draw on
        bbox: Tuple of (x1, y1, x2, y2) coordinates
        label: String label for the detection
        conf: Confidence score (0-1)
    """
    x1, y1, x2, y2 = map(int, bbox)
    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Prepare label text
    text = f"{label} {conf:.2f}"
    # Get text size and background size
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    # Draw label background
    cv2.rectangle(frame, (x1, y1-text_h-4), (x1+text_w+2, y1), (0, 255, 0), -1)
    # Draw text
    cv2.putText(frame, text, (x1+1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

def draw_hud(frame, fps, status_text):
    """Draw heads-up display with FPS and status.
    
    Args:
        frame: OpenCV BGR image to draw on
        fps: Float FPS value
        status_text: Status message to display
    """
    # Draw FPS in top-right corner
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (frame.shape[1]-120, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw status text in bottom-left corner
    cv2.putText(frame, status_text, (10, frame.shape[0]-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
