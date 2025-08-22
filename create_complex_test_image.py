import cv2
import numpy as np

def create_complex_test_image():
    # Create a sky-like background
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 200  # Light gray for sky
    
    # Add some clouds (light gray patches)
    for _ in range(5):
        x = np.random.randint(0, 1180)
        y = np.random.randint(0, 200)
        cv2.circle(img, (x, y), 50, (240, 240, 240), -1)
    
    # Draw power lines (dark lines with perspective)
    # First pole
    cv2.line(img, (200, 100), (200, 600), (80, 80, 80), 4)  # Vertical pole
    # Second pole
    cv2.line(img, (1000, 150), (1000, 700), (80, 80, 80), 4)  # Vertical pole
    
    # Power lines with perspective
    for offset in [0, 30, 60]:  # Multiple lines
        cv2.line(img, (200, 200 + offset), (1000, 250 + offset), (60, 60, 60), 2)
    
    # Add some "foreign objects"
    # Bird near power line
    cv2.ellipse(img, (500, 230), (20, 15), 0, 0, 360, (30, 30, 30), -1)
    
    # Drone near power line
    drone_pts = np.array([[700, 280], [730, 280], [715, 300]], np.int32)
    cv2.fillPoly(img, [drone_pts], (0, 0, 255))
    
    # Tree branch coming close to power line
    pts = np.array([[300, 350], [350, 270], [400, 350]], np.int32)
    cv2.fillPoly(img, [pts], (0, 100, 0))
    
    # Save both original and a version with annotations
    cv2.imwrite('test_image_complex.jpg', img)
    
    # Add annotations for ground truth
    cv2.rectangle(img, (480, 210), (520, 250), (0, 255, 0), 2)  # Bird
    cv2.rectangle(img, (695, 275), (735, 305), (0, 255, 0), 2)  # Drone
    cv2.rectangle(img, (290, 260), (410, 360), (0, 255, 0), 2)  # Tree branch
    cv2.imwrite('test_image_complex_annotated.jpg', img)
    
    print("Created test images: test_image_complex.jpg and test_image_complex_annotated.jpg")

if __name__ == "__main__":
    create_complex_test_image()
