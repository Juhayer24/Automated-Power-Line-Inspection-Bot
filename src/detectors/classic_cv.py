"""Good old-fashioned computer vision for finding power lines and hazards.
No fancy AI here - just tried and true edge detection and line finding.
Warning: Can be twitchy in bad lighting or with lots of background clutter.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# Default parameters for the detection pipeline
DEFAULT_PARAMS = {
    'canny_low': 50,       # Lower threshold for Canny edge detector
    'canny_high': 150,     # Upper threshold for Canny
    'hough_rho': 1,        # Distance resolution in pixels
    'hough_theta': np.pi/180,  # Angle resolution in radians
    'hough_threshold': 50, # Minimum number of intersections
    'min_line_length': 50, # Minimum length of line
    'max_line_gap': 10,    # Maximum gap between line segments
    'min_area': 800,      # Minimum area for hazard regions
}

def detect_hazards_classic(frame: np.ndarray, 
                         params: Optional[Dict[str, Any]] = None) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray, np.ndarray]:
    """Detect potential hazards using classic computer vision techniques.
    
    Uses Canny edge detection and Hough line transform to identify power lines,
    then looks for significant regions that might represent hazards.
    
    Args:
        frame: BGR input image
        params: Optional dictionary of parameters to override defaults
        
    Returns:
        Tuple containing:
        - List of bounding boxes as (x, y, w, h)
        - Binary mask showing detected power lines
        - Binary mask showing non-line regions of interest
        
    Example:
        >>> import cv2
        >>> img = cv2.imread('power_line.jpg')
        >>> # Use default parameters
        >>> boxes, line_mask, non_line = detect_hazards_classic(img)
        >>> 
        >>> # Override parameters
        >>> custom_params = {
        ...     'canny_low': 30,
        ...     'canny_high': 100,
        ...     'min_line_length': 80
        ... }
        >>> boxes, line_mask, non_line = detect_hazards_classic(img, custom_params)
    """
    # Merge default params with any provided overrides
    p = DEFAULT_PARAMS.copy()
    if params:
        p.update(params)
        
    # Convert to grayscale and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges using Canny
    edges = cv2.Canny(blur, p['canny_low'], p['canny_high'])
    
    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(edges, p['hough_rho'], p['hough_theta'],
                           p['hough_threshold'], None,
                           minLineLength=p['min_line_length'],
                           maxLineGap=p['max_line_gap'])
    
    # Create binary masks
    h, w = frame.shape[:2]
    line_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Draw detected lines on the line mask
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
    
    # Dilate edges to get region around them
    kernel = np.ones((5,5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Find regions that might be hazards (not part of power lines)
    non_line = cv2.bitwise_xor(dilated_edges, line_mask)
    
    # Find contours in the non-line regions
    contours, _ = cv2.findContours(non_line, cv2.RETR_EXTERNAL, 
                                 cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and convert to bounding boxes
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < p['min_area']:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter out extremely tall/narrow or tiny shapes
        if w < 10 or h < 10:
            continue
        boxes.append((x, y, w, h))
            
    return boxes, line_mask, non_line

def visualize_masks(frame: np.ndarray,
                   line_mask: np.ndarray,
                   non_line: np.ndarray) -> np.ndarray:
    """Create a visualization of the detection masks overlaid on the input frame.
    
    Args:
        frame: Original BGR input image
        line_mask: Binary mask showing detected lines
        non_line: Binary mask showing non-line regions
        
    Returns:
        BGR image with color-coded overlay of both masks
        
    Example:
        >>> img = cv2.imread('power_line.jpg')
        >>> boxes, line_mask, non_line = detect_hazards_classic(img)
        >>> vis = visualize_masks(img, line_mask, non_line)
        >>> cv2.imshow('Visualization', vis)
    """
    # Create copy of input frame
    vis = frame.copy()
    
    # Add blue overlay for power lines
    vis[line_mask > 0] = cv2.addWeighted(
        vis[line_mask > 0], 0.7,
        np.full_like(vis[line_mask > 0], [255, 0, 0]), 0.3,
        0
    )
    
    # Add red overlay for potential hazard regions
    vis[non_line > 0] = cv2.addWeighted(
        vis[non_line > 0], 0.7,
        np.full_like(vis[non_line > 0], [0, 0, 255]), 0.3,
        0
    )
    
    return vis
