!<img width="611" height="429" alt="Screenshot 2025-08-22 at 3 18 09 AM" src="https://github.com/user-attachments/assets/544b09d0-6a04-400d-bdbf-266bc3fcf938" />

[powerline1](https://github.com/user-attachments/assets/5299cf93-67de-4f9e-8175-d2407985a7ff)

# Automated Power Line Inspection System

An innovative computer vision system designed to enhance power line infrastructure maintenance through automated detection of potential hazards and foreign objects. This project combines classical computer vision techniques with modern deep learning approaches to create a robust, real-time inspection solution.

## Project Overview

Power line infrastructure maintenance is critical for ensuring reliable electricity distribution and preventing outages. Manual inspection is time-consuming, expensive, and can be dangerous. This system provides:

- Real-time detection of foreign objects and hazards near power lines
- Automated monitoring and early warning system
- Cost-effective alternative to manual inspection
- Enhanced safety through remote monitoring capabilities

## Key Features

### Core Detection System
- **Classical Computer Vision Pipeline**
  - Advanced edge detection using adaptive Canny algorithms
  - Hough transform for power line identification
  - Contour analysis for foreign object detection
  - Robust against varying lighting conditions

- **Deep Learning Integration**
  - YOLOv8-based object detection
  - Custom-trained models for specific hazard types
  - Real-time inference optimization
  - ONNX export support for deployment

### System Features
- Multi-source video input support (cameras, video files)
- Real-time processing and hazard detection
- Comprehensive logging and monitoring system
- Flexible deployment options (Desktop/Raspberry Pi)
- Remote monitoring capabilities

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/Juhayer24/Automated-Power-Line-Inspection-Bot.git
cd Automated-Power-Line-Inspection-Bot
```

2. Setup the environment (creates Python virtual environment and installs dependencies):
```bash
./setup_env.sh
source .venv/bin/activate
```

3. Run the demo:
```bash
./run.sh
```

## System Implementation

### Desktop Implementation

1. Run with webcam using classical CV:
```bash
python3 src/app.py \
    --source 0 \
    --detector classic \
    --record \
    --output runs/demo.mp4
```

2. Process a video file using YOLO:
```bash
python3 src/app.py \
    --source data/videos/test.mp4 \
    --detector yolo \
    --yolo-weights data/models/yolo/yolov8n.pt \
    --yolo-every 3  # Run YOLO every 3rd frame for speed
```

3. Extract frames from a video:
```bash
python3 scripts/extract_frames.py \
    --input data/videos/inspection.mp4 \
    --out_dir data/frames \
    --step 5 \
    --resize 640x480
```

### Field Deployment (Raspberry Pi)

The system can be deployed on Raspberry Pi hardware for field installations:

1. Standard deployment with classical CV processing:
```bash
python3 src/app.py --use-pi --detector classic
```

2. Advanced deployment with deep learning and telemetry:
```bash
python3 src/app.py \
    --use-pi \
    --detector yolo \
    --yolo-weights data/models/yolo/yolov8n.pt \
    --log-file logs/inspection.csv \
    --record
```

## Deep Learning Implementation

This project supports YOLOv8 models for power line detection. To train your own model:

1. Prepare your dataset in YOLOv8 format
2. Follow the [Ultralytics YOLOv8 Training Guide](https://docs.ultralytics.com/tasks/detect/#train)
3. Export to ONNX for optimized inference:
```bash
python3 scripts/export_yolo_onnx.py \
    --weights runs/train/weights/best.pt \
    --output data/models/yolo/model.onnx \
    --image data/images/test.jpg
```

## Hardware Implementation

### System Requirements
- Raspberry Pi 4 (recommended) or 3B+
- Pi Camera Module or USB webcam
- Servo motors for movement control
- Power supply (5V, 2.5A minimum)

### Safety Considerations for GPIO Implementation
1. Always power off the Pi before connecting/disconnecting components
2. Use appropriate voltage levels (3.3V for GPIO)
3. Include resistors for LED indicators
4. Double-check servo motor power requirements
5. Refer to `src/pi/gpio_io.py` for pin mappings

### Performance Tips
- Use `--yolo-every N` to run YOLO every N frames
- Consider using ONNX exported models
- Enable Pi camera hardware encoding for recording
- Monitor CPU temperature during extended operation

## Data Collection and Analysis

The system generates detailed CSV logs including:
- Timestamps
- Detection states
- Bounding boxes
- Angles
- Custom metrics

Example log analysis:
```python
from src.logging.log_writer import LogWriter
from src.state.state_machine import DebounceState

# Create logger
logger = LogWriter('logs/inspection.csv')

# Log events with state tracking
state = DebounceState(safe_to_hazard_frames=3)
logger.write_event(
    state=state.current_state,
    bbox=(100, 100, 50, 30),
    angle=45.2,
    extra={'confidence': 0.95}
)
```

## Development Architecture

### Project Structure
```
Automated-Power-Line-Inspection-Bot/
├── data/               # Data storage
│   ├── images/        # Test images
│   ├── videos/        # Test videos
│   └── models/        # YOLO weights
├── scripts/           # Utility scripts
├── src/               # Main source code
│   ├── detectors/     # CV algorithms
│   ├── io/           # Camera handling
│   ├── pi/           # Pi specific code
│   ├── state/        # State management
│   └── logging/      # Event logging
└── logs/             # Output logs
```

### Adding New Features
1. Follow the existing module structure
2. Add appropriate error handling
3. Update tests if applicable
4. Document new parameters in docstrings

## Contributing

We welcome contributions to enhance the system's capabilities. Please review our Contributing Guidelines before submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Special thanks to:

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV community
- Raspberry Pi Foundation
