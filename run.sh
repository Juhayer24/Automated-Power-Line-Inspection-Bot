#!/bin/bash

# Ensure we're in the virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    else
        echo "Virtual environment not found. Please run setup_env.sh first."
        exit 1
    fi
fi

# Default arguments
VIDEO_SOURCE="0"  # Default to webcam
DETECTOR="classic"  # Default to classic CV detector

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --video)
            VIDEO_SOURCE="$2"
            shift
            shift
            ;;
        --detector)
            DETECTOR="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Run the application
python3 src/app.py --source "$VIDEO_SOURCE" --detector "$DETECTOR"
