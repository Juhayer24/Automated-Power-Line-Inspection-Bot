import argparse
import sys
from pathlib import Path
import numpy as np
import cv2
import onnxruntime as ort
from ultralytics import YOLO

def export_yolo_to_onnx(weights_path: str, onnx_path: str, image_size: int = 640) -> bool:
    """
    Converts our YOLO model to something that runs faster.
    ONNX models are optimized and run better on most hardware.
    
    Args:
        weights_path: Where the trained model is saved
        onnx_path: Where to save the converted model
        image_size: Size of images it expects (usually leave at 640)
        
    Returns:
        True if it worked, False if something went wrong
    """
    try:
        # Load YOLO model
        model = YOLO(weights_path)
        
        # Try different export methods based on ultralytics version
        try:
            # Newer versions of ultralytics
            success = model.export(format="onnx", 
                                 imgsz=[image_size, image_size],
                                 simplify=True,
                                 opset=12,
                                 filepath=onnx_path)
        except (AttributeError, TypeError):
            try:
                # Older versions might use model.model.export
                success = model.model.export(format="onnx",
                                          imgsz=[image_size, image_size],
                                          simplify=True,
                                          opset=12,
                                          file=onnx_path)
            except Exception as e:
                print(f"Error during model.model.export: {str(e)}")
                return False
                
        return True if success or Path(onnx_path).exists() else False
        
    except Exception as e:
        print(f"Error during export: {str(e)}")
        return False

def preprocess_image(image_path: str, image_size: int = 640) -> np.ndarray:
    """
    Preprocess image for ONNX inference.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
        
    # Resize and convert to float32
    img = cv2.resize(img, (image_size, image_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1] and transpose to NCHW format
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    return img

def validate_onnx_model(onnx_path: str, image_path: str, image_size: int = 640) -> bool:
    """
    Validate ONNX model by running inference on a sample image.
    """
    try:
        # Create inference session
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        # Get model metadata
        input_name = session.get_inputs()[0].name
        
        # Preprocess image
        img = preprocess_image(image_path, image_size)
        
        # Run inference
        outputs = session.run(None, {input_name: img})
        
        # Check output shape and content
        if len(outputs) == 0:
            print("Error: No output from model")
            return False
            
        # Process and print example detections
        detections = outputs[0]  # Assuming first output is detection tensor
        print(f"\nExample output tensor shape: {detections.shape}")
        
        if len(detections.shape) >= 2:
            num_detections = min(3, len(detections))  # Show up to 3 detections
            for i in range(num_detections):
                # Format varies by model version, but typically:
                # [x1, y1, x2, y2, conf, class_id] or [x, y, w, h, conf, class_id]
                detection = detections[i]
                print(f"Detection {i+1}: {detection}")
                
        return True
        
    except Exception as e:
        print(f"Error during validation: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Export YOLO model to ONNX and validate')
    parser.add_argument('--weights', required=True, help='Path to YOLO weights file')
    parser.add_argument('--output', required=True, help='Output path for ONNX model')
    parser.add_argument('--image', required=True, help='Path to sample image for validation')
    parser.add_argument('--size', type=int, default=640, help='Input image size (default: 640)')
    
    args = parser.parse_args()
    
    # Export model
    print(f"Exporting {args.weights} to ONNX format...")
    if not export_yolo_to_onnx(args.weights, args.output, args.size):
        print("Export failed!")
        return 1
    
    print(f"Successfully exported to: {args.output}")
    
    # Validate model
    print(f"\nValidating ONNX model with sample image...")
    if not validate_onnx_model(args.output, args.image, args.size):
        print("Validation failed!")
        return 1
        
    print("\nValidation successful!")
    return 0

if __name__ == '__main__':
    sys.exit(main())
