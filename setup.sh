#!/bin/bash
# Setup script for Webcam Depth Detection project

set -e

echo "Setting up Webcam Depth Detection project..."

# Install Python dependencies (including depth-anything-v2)
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create checkpoints directory
echo "Creating checkpoints directory..."
mkdir -p checkpoints

# Download small model (recommended for real-time)
echo "Downloading Depth-Anything-V2 Small model..."
if command -v wget &> /dev/null; then
    wget -P checkpoints https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
elif command -v curl &> /dev/null; then
    curl -L -o checkpoints/depth_anything_v2_vits.pth https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
else
    echo "Error: Neither wget nor curl found. Please install one of them."
    exit 1
fi

echo ""
echo "Setup complete!"
echo ""
echo "To run the application:"
echo "  python webcam_depth.py"
echo ""
echo "For more options:"
echo "  python webcam_depth.py --help"
