"""
Video input handler that supports both USB/builtin cameras and the Pi Camera.
Falls back to OpenCV if Pi Camera isn't available. Tries to keep things simple
while being robust enough for real deployments.
"""

import logging
from typing import Union, Tuple, Optional
import cv2
import numpy as np
import platform

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if we're on a Raspberry Pi
IS_RASPBERRY_PI = platform.machine().startswith('arm') or platform.machine().startswith('aarch')

# Try to import picamera2, but don't raise error if not available
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    logger.info("picamera2 module not available - this is normal on non-Raspberry Pi systems")

class VideoSource:
    """
    Unified video source interface with fallback options.
    Supports OpenCV VideoCapture and Raspberry Pi Camera (picamera2).
    """
    
    def __init__(self, source: Union[int, str], use_picamera: bool = False, 
                 width: int = 640, height: int = 480):
        """
        Initialize video source with specified parameters.
        
        Args:
            source: Camera index (int) or video file path (str)
            use_picamera: Try to use Raspberry Pi camera if True
            width: Desired frame width
            height: Desired frame height
        """
        self.source = source
        self.width = width
        self.height = height
        self.capture = None
        self._picam = None
        self._is_running = False
        
        # Try picamera2 if requested and available on Raspberry Pi
        if use_picamera and IS_RASPBERRY_PI and PICAMERA_AVAILABLE:
            logger.info("Initializing Raspberry Pi Camera...")
            try:
                self._picam = Picamera2()
                
                # Configure camera
                config = self._picam.create_preview_configuration(
                    main={"size": (width, height)},
                    buffer_count=2  # Double buffering for smoother capture
                )
                self._picam.configure(config)
                
                logger.info("Successfully initialized Raspberry Pi Camera")
                return
                
            except ImportError:
                logger.warning("picamera2 not available. Falling back to OpenCV...")
            except Exception as e:
                logger.error(f"Failed to initialize Raspberry Pi Camera: {str(e)}")
                logger.warning("Falling back to OpenCV...")
        
        # OpenCV fallback
        try:
            logger.info(f"Initializing OpenCV VideoCapture with source: {source}")
            self.capture = cv2.VideoCapture(source)
            
            # Set resolution if source is a camera
            if isinstance(source, int):
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
            if not self.capture.isOpened():
                raise RuntimeError("Failed to open video source")
                
            logger.info("Successfully initialized OpenCV VideoCapture")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize video source: {str(e)}")

    def start(self) -> None:
        """Start the video stream."""
        if self._is_running:
            return
            
        if self._picam:
            try:
                self._picam.start()
            except Exception as e:
                raise RuntimeError(f"Failed to start Raspberry Pi Camera: {str(e)}")
        
        self._is_running = True
        logger.info("Video stream started")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the video source.
        
        Returns:
            Tuple of (success, frame)
            - success: True if frame was successfully read
            - frame: BGR numpy array or None if read failed
        """
        if not self._is_running:
            logger.warning("Attempting to read from stopped video source")
            return False, None
            
        try:
            if self._picam:
                # Get frame from Raspberry Pi Camera
                frame = self._picam.capture_array()
                # Convert to BGR if needed (picamera2 returns RGB)
                if frame is not None and frame.ndim == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return True, frame
                
            elif self.capture:
                # Get frame from OpenCV
                ret, frame = self.capture.read()
                return ret, frame
                
        except Exception as e:
            logger.error(f"Error reading frame: {str(e)}")
        
        return False, None

    def release(self) -> None:
        """Release the video source and free resources."""
        self._is_running = False
        
        if self._picam:
            try:
                self._picam.stop()
                self._picam.close()
            except Exception as e:
                logger.error(f"Error stopping Raspberry Pi Camera: {str(e)}")
                
        if self.capture:
            try:
                self.capture.release()
            except Exception as e:
                logger.error(f"Error releasing OpenCV capture: {str(e)}")
                
        logger.info("Video source released")

    def __enter__(self) -> 'VideoSource':
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.release()

    def __repr__(self) -> str:
        """String representation."""
        source_type = "PiCamera" if self._picam else "OpenCV"
        status = "running" if self._is_running else "stopped"
        return f"VideoSource({source_type}, {self.width}x{self.height}, {status})"
