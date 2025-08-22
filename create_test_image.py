import cv2
import numpy as np

# Create a blank image
img = np.zeros((480, 640, 3), dtype=np.uint8)
img.fill(255)  # Make it white

# Draw some simulated power lines
cv2.line(img, (100, 100), (540, 100), (100, 100, 100), 2)  # Horizontal line
cv2.line(img, (100, 200), (540, 200), (100, 100, 100), 2)  # Horizontal line

# Draw a "foreign object" near the line
cv2.rectangle(img, (300, 80), (350, 120), (0, 0, 255), -1)  # Red rectangle

# Save the test image
cv2.imwrite('test_image.jpg', img)
print("Created test image: test_image.jpg")
