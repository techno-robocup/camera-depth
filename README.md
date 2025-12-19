# Webcam Depth Detection using Depth-Anything-V2

Real-time depth estimation from webcam feed using Depth-Anything-V2 model.

## Features

- Real-time depth estimation from webcam
- Multiple model sizes (small, base, large)
- Side-by-side visualization of original frame and depth map
- Save snapshots with 's' key
- GPU acceleration support

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, but recommended for real-time performance)
- Webcam

## Installation

### Quick Setup (Automated)

Run the setup script:
```bash
./setup.sh
```

This will install all dependencies (including depth-anything-v2 from PyPI) and download the small model.

### Manual Setup

1. Install dependencies (includes depth-anything-v2 package):
```bash
pip install -r requirements.txt
```

2. Create checkpoints directory:
```bash
mkdir -p checkpoints
```

3. Download model weights:

**Small model (recommended for real-time):**
```bash
wget -P checkpoints https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
```

**Base model:**
```bash
wget -P checkpoints https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth
```

**Large model:**
```bash
wget -P checkpoints https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
```

## Usage

Basic usage with default settings (small model, camera 0):
```bash
python webcam_depth.py
```

With custom options:
```bash
python webcam_depth.py --model-size base --camera 0 --width 640 --height 480
```

### Command-line Arguments

- `--model-size`: Model size to use (`small`, `base`, or `large`). Default: `small`
- `--device`: Device to use (`cuda` or `cpu`). Default: auto-detect
- `--camera`: Camera device index. Default: `0`
- `--width`: Camera capture width. Default: `640`
- `--height`: Camera capture height. Default: `480`

### Controls

- **q**: Quit the application
- **s**: Save current frame and depth map

## Project Structure

```
camera-depth/
├── webcam_depth.py       # Main application script
├── requirements.txt      # Python dependencies
├── setup.sh             # Automated setup script
├── README.md            # This file
└── checkpoints/         # Model weights directory
    └── depth_anything_v2_*.pth
```

## Performance Tips

- Use the `small` model for best real-time performance
- Reduce camera resolution if experiencing lag
- Use CUDA if available for significant speed improvement
- Close other applications to free up GPU memory

## Troubleshooting

**ImportError: No module named 'depth_anything_v2'**
- Make sure you installed all dependencies: `pip install -r requirements.txt`
- Or install the package directly: `pip install depth-anything-v2`

**FileNotFoundError: checkpoint not found**
- Download the model weights as described in the Installation section
- Ensure weights are in the `checkpoints/` directory

**Camera not opening**
- Check if your camera is accessible: `ls /dev/video*`
- Try a different camera index with `--camera 1`
- Ensure no other application is using the camera

**Slow performance**
- Use the `small` model instead of `base` or `large`
- Reduce camera resolution
- Use GPU if available

## References

- [Depth-Anything-V2 GitHub](https://github.com/DepthAnything/Depth-Anything-V2)
- [Depth-Anything-V2 Paper](https://arxiv.org/abs/2406.09414)
