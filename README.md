# Automated Power Line Inspection Bot 🤖 🔌

An intelligent robotic system for automated power line inspection using computer vision and robotics. This project combines classical CV techniques and modern deep learning (YOLO) to detect power lines and potential hazards, with optional Raspberry Pi support for field deployments.

## 🌟 Features

- Real-time power line detection using classical CV or YOLO
- Support for multiple input sources (webcam, video files, Pi camera)
- State machine for reliable hazard detection
- Detailed CSV logging with timestamps and metrics
- Video recording and frame extraction utilities
- Raspberry Pi GPIO integration for robotic control
- ONNX export support for optimized inference

## 🚀 Quickstart

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

## 📚 Usage Examples

### Laptop/Desktop Mode

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

### Raspberry Pi Mode

1. Basic Pi setup with classical CV:
```bash
python3 src/app.py --use-pi --detector classic
```

2. Full Pi deployment with YOLO and logging:
```bash
python3 src/app.py \
    --use-pi \
    --detector yolo \
    --yolo-weights data/models/yolo/yolov8n.pt \
    --log-file logs/inspection.csv \
    --record
```

## 🛠️ YOLO Training

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

## 🤖 Raspberry Pi Setup

### Hardware Requirements
- Raspberry Pi 4 (recommended) or 3B+
- Pi Camera Module or USB webcam
- Servo motors for movement control
- Power supply (5V, 2.5A minimum)

### GPIO Wiring Safety ⚠️
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

## 📊 Logging and Analysis

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

## 🔬 Development

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

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV community
- Raspberry Pi Foundation