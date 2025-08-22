# src/sim/indicators.py
import cv2
import math

class IndicatorSimulator:
    def __init__(self):
        self.is_hazard = False

    def set_hazard(self, state: bool):
        self.is_hazard = bool(state)

    def draw(self, frame, angle_deg=None):
        h, w = frame.shape[:2]
        # LED indicator (top-left)
        led_center = (30, 30)
        color = (0,0,255) if self.is_hazard else (0,255,0)   # BGR
        cv2.circle(frame, led_center, 12, color, -1)
        cv2.putText(frame, "HAZARD" if self.is_hazard else "SAFE", (50,36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Servo arrow (bottom-left)
        base = (60, h-60)
        length = 70
        if angle_deg is None:
            angle_deg = 0.0
        # convert degrees to radians; negative so up arrow corresponds to negative angle
        ang = -math.radians(angle_deg)
        end = (int(base[0] + length * math.cos(ang)), int(base[1] + length * math.sin(ang)))
        cv2.arrowedLine(frame, base, end, (255,0,0), 4, tipLength=0.3)
        cv2.putText(frame, f"{angle_deg:.0f}°", (base[0]-10, base[1]-80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
