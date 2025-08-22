#!/bin/bash

# Ensure script is not run as root
if [ "$EUID" -eq 0 ]; then
    echo "Please do not run this script as root or with sudo."
    echo "The script will ask for sudo permissions when needed."
    exit 1
fi

echo "Setting up environment for Power Line Inspection Bot..."

# Check Python version (require >= 3.10)
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [ $(echo "$PYTHON_VERSION >= 3.10" | bc -l) -eq 0 ]; then
    echo "Error: Python version $PYTHON_VERSION detected."
    echo "This project requires Python 3.10 or higher."
    echo "Please install Python 3.10+ and try again."
    exit 1
fi

# Detect OS and install system dependencies
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Installing system dependencies..."
    # Check if we can use sudo
    if ! command -v sudo &> /dev/null; then
        echo "Error: 'sudo' is required but not installed."
        exit 1
    fi
    
    # Update package lists
    echo "Updating package lists..."
    sudo apt-get update || { echo "Failed to update package lists"; exit 1; }
    
    # Install required packages
    echo "Installing required system packages..."
    sudo apt-get install -y \
        python3-venv \
        python3-pip \
        build-essential \
        libatlas-base-dev \
        ffmpeg || { echo "Failed to install system packages"; exit 1; }
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "macOS detected. Please ensure you have Homebrew installed and the following packages:"
    echo "- python3"
    echo "- ffmpeg"
    echo "You can install them using: brew install python3 ffmpeg"
else
    echo "Warning: Unsupported operating system. You may need to install dependencies manually:"
    echo "- Python 3.10 or higher"
    echo "- ffmpeg"
    echo "- build-essential tools"
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv || { echo "Failed to create virtual environment"; exit 1; }
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip || { echo "Failed to upgrade pip"; exit 1; }

# Install requirements
echo "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt || { echo "Failed to install Python dependencies"; exit 1; }
else
    echo "Error: requirements.txt not found!"
    exit 1
fi

# Print PyTorch information
echo "
=== Important Note about PyTorch ===
If you need GPU support, please visit:
https://pytorch.org/get-started/locally/

Select your system configuration to get the correct installation command:
- Operating System
- Package manager (pip)
- CUDA version (if using GPU)

Install the appropriate PyTorch package in your virtual environment.
=================================="

echo "
Setup complete! 🎉

To use the Power Line Inspection Bot:

1. Activate the virtual environment (if not already activated):
   source .venv/bin/activate

2. Run the application:
   ./run.sh

Note: Make sure run.sh is executable (chmod +x run.sh if needed)
"

# Make run.sh executable if it exists
if [ -f "run.sh" ]; then
    chmod +x run.sh
fi

To run the application:
./run.sh
"
