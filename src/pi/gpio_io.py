"""Controls the robot's physical bits (LEDs and servos).
Works on a Pi but also has a fake mode for testing on regular computers.

!!! READ THIS BEFORE WIRING ANYTHING !!!
The servo needs its own power supply - don't try to power it from the Pi!
Just hook up the control wire and make sure the grounds are connected.
If you try to power it from the Pi:
1. The servo will twitch and be super flaky
2. You might fry your Pi
3. The Pi will probably keep crashing
Been there, fried that. Just use a proper power supply.
"""

import logging
from typing import Optional
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pin definitions
LED_PIN = 18    # GPIO18
SERVO_PIN = 13  # GPIO13

# Servo parameters
SERVO_FREQ = 50  # 50Hz PWM frequency
DUTY_MIN = 2.5  # Duty cycle for 0 degrees
DUTY_MAX = 12.5 # Duty cycle for 180 degrees

class GPIOSimulator:
    """Simulated GPIO for development on non-Pi systems."""
    
    def __init__(self):
        self.pins = {}
        self.pwm = {}
        logger.info("Initializing GPIO Simulator")
        
    def setup(self, pin: int, mode: str) -> None:
        self.pins[pin] = 0
        logger.info(f"Setup PIN{pin} as {mode}")
        
    def output(self, pin: int, state: int) -> None:
        self.pins[pin] = state
        logger.info(f"PIN{pin} set to {state}")
        
    def PWM(self, pin: int, freq: int) -> 'PWMSimulator':
        self.pwm[pin] = PWMSimulator(pin, freq)
        return self.pwm[pin]
        
    def cleanup(self) -> None:
        logger.info("Cleaning up GPIO")
        self.pins.clear()
        self.pwm.clear()

class PWMSimulator:
    """Simulated PWM output for servo control."""
    
    def __init__(self, pin: int, freq: int):
        self.pin = pin
        self.freq = freq
        self.duty = 0
        logger.info(f"PWM on PIN{pin} at {freq}Hz")
        
    def start(self, duty: float) -> None:
        self.duty = duty
        logger.info(f"PWM PIN{self.pin} started at {duty}% duty cycle")
        
    def ChangeDutyCycle(self, duty: float) -> None:
        self.duty = duty
        logger.info(f"PWM PIN{self.pin} duty cycle changed to {duty}%")
        
    def stop(self) -> None:
        logger.info(f"PWM PIN{self.pin} stopped")

# Try to import RPi.GPIO, fallback to simulator if not available
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
    logger.info("Using real RPi.GPIO")
except ImportError:
    GPIO = GPIOSimulator()
    GPIO_AVAILABLE = False
    logger.info("Using GPIO simulator")

# Global PWM object for servo
servo_pwm: Optional[PWMSimulator] = None

def angle_to_duty(angle_deg: float) -> float:
    """Convert servo angle to PWM duty cycle.
    
    Args:
        angle_deg: Angle in degrees (0-180)
        
    Returns:
        PWM duty cycle (2.5-12.5%)
    """
    # Clamp angle to valid range
    angle_deg = max(0, min(180, angle_deg))
    # Linear interpolation between duty cycle limits
    return DUTY_MIN + (angle_deg / 180.0) * (DUTY_MAX - DUTY_MIN)

def setup_gpio() -> None:
    """Initialize GPIO pins for LED and servo control."""
    global servo_pwm
    
    # Use BCM pin numbering
    GPIO.setmode(GPIO.BCM)
    
    # Setup LED pin as output
    GPIO.setup(LED_PIN, GPIO.OUT)
    GPIO.output(LED_PIN, GPIO.LOW)
    
    # Setup servo pin as PWM output
    GPIO.setup(SERVO_PIN, GPIO.OUT)
    servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQ)
    servo_pwm.start(angle_to_duty(90))  # Start at center position
    
    logger.info("GPIO initialized")

def set_led_state(is_hazard: bool) -> None:
    """Set the state of the hazard indicator LED.
    
    Args:
        is_hazard: True to turn LED on, False to turn off
    """
    GPIO.output(LED_PIN, GPIO.HIGH if is_hazard else GPIO.LOW)

def set_servo_angle(angle_deg: float) -> None:
    """Set servo position by angle in degrees.
    
    Args:
        angle_deg: Target angle (0-180 degrees)
    """
    if servo_pwm is None:
        logger.error("Servo not initialized! Call setup_gpio() first")
        return
        
    duty = angle_to_duty(angle_deg)
    servo_pwm.ChangeDutyCycle(duty)
    
    # Small delay to allow servo to move
    time.sleep(0.1)

def cleanup_gpio() -> None:
    """Cleanup GPIO state - should be called on program exit."""
    if servo_pwm is not None:
        servo_pwm.stop()
    GPIO.cleanup()
    logger.info("GPIO cleaned up")
