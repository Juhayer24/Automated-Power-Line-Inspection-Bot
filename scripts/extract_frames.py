import argparse
import cv2
import numpy as np
import os
from datetime import datetime, timedelta
from pathlib import Path

def calculate_entropy(image):
    """Figures out how much 'stuff' is going on in the image.
    Helps us spot frames that are basically the same or just empty."""
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    histogram_norm = histogram.ravel() / histogram.sum()
    non_zero_probs = histogram_norm[histogram_norm > 0]
    entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs))
    return entropy

def calculate_variance(image):
    """Calculate image variance as an alternative measure."""
    return cv2.meanStdDev(image)[1][0][0]

def is_similar_to_previous(current_frame, prev_metrics, threshold=0.1):
    """Check if frame is too similar to previous frame."""
    if prev_metrics is None:
        return False
        
    curr_entropy = calculate_entropy(current_frame)
    curr_variance = calculate_variance(current_frame)
    
    entropy_diff = abs(curr_entropy - prev_metrics['entropy'])
    variance_diff = abs(curr_variance - prev_metrics['variance'])
    
    return entropy_diff < threshold and variance_diff < threshold

def extract_frames(input_path, out_dir, step=1, resize=None, deduplicate=True):
    """Extract frames from video with optional deduplication and resizing."""
    video = cv2.VideoCapture(input_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    start_time = datetime.now() - timedelta(seconds=duration)
    
    # Create output directory
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Process frames
    frame_number = 0
    saved_count = 0
    prev_metrics = None
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
            
        if frame_number % step == 0:
            # Resize if specified
            if resize:
                width, height = map(int, resize.split('x'))
                frame = cv2.resize(frame, (width, height))
            
            # Check for duplicates if enabled
            if deduplicate and is_similar_to_previous(frame, prev_metrics):
                frame_number += 1
                continue
            
            # Calculate frame timestamp
            timestamp = start_time + timedelta(seconds=frame_number/fps)
            
            # Save frame
            filename = f"frame_{frame_number:06d}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.png"
            cv2.imwrite(str(out_dir / filename), frame)
            saved_count += 1
            
            # Update metrics for next comparison
            prev_metrics = {
                'entropy': calculate_entropy(frame),
                'variance': calculate_variance(frame)
            }
            
        frame_number += 1
        
    video.release()
    return frame_number, saved_count

def main():
    parser = argparse.ArgumentParser(description='Extract frames from video with deduplication')
    parser.add_argument('--input', required=True, help='Path to input video file')
    parser.add_argument('--out_dir', required=True, help='Output directory for frames')
    parser.add_argument('--step', type=int, default=1, help='Extract every Nth frame')
    parser.add_argument('--resize', help='Resize frames to WxH (e.g., 640x480)')
    parser.add_argument('--no-deduplicate', action='store_true', help='Disable deduplication')
    
    args = parser.parse_args()
    
    try:
        total_frames, saved_frames = extract_frames(
            args.input,
            args.out_dir,
            args.step,
            args.resize,
            not args.no_deduplicate
        )
        
        print(f"\nFrame Extraction Summary:")
        print(f"Total frames processed: {total_frames}")
        print(f"Frames saved: {saved_frames}")
        print(f"Frames skipped: {total_frames - saved_frames}")
        print(f"Output directory: {args.out_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
