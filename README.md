# Webcam Depth Detection using Depth-Anything-V2

Real-time depth estimation from webcam feed using Depth-Anything-V2 model.

## Features

- Real-time depth estimation from webcam
- Multiple model sizes (small, base, large)
- Side-by-side visualization of original frame and depth map
- Depth discontinuity detection - identify objects and boundaries based on depth changes
- Contour detection for depth edges
- Adjustable edge detection threshold
- Toggle between depth view and contour view
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

### Webcam Mode (Real-time)

Basic usage with default settings (small model, camera 0):
```bash
python webcam_depth.py
```

With custom options:
```bash
python webcam_depth.py --model-size base --camera 0 --width 640 --height 480 --edge-threshold 0.15
```

### Image Mode (Process single image)

Process a single image and save all outputs:
```bash
python webcam_depth.py --image path/to/image.jpg --output results/
```

This will create the following files in the output directory:
- `{filename}_original.jpg` - Original input image
- `{filename}_depth.jpg` - Colorized depth map
- `{filename}_contours.jpg` - Image with depth contours drawn
- `{filename}_edges.jpg` - Depth edge map
- `{filename}_combined_depth.jpg` - Side-by-side original and depth
- `{filename}_combined_contours.jpg` - Side-by-side contours and edges

With custom settings:
```bash
python webcam_depth.py --image photo.jpg --output my_results/ --edge-threshold 0.2 --min-contour-area 200
```

### Command-line Arguments

**General:**
- `--model-size`: Model size to use (`small`, `base`, or `large`). Default: `small`
- `--device`: Device to use (`cuda` or `cpu`). Default: auto-detect
- `--edge-threshold`: Threshold for depth edge detection (0-1). Default: `0.1`
- `--min-contour-area`: Minimum contour area to display in pixels. Default: `100`

**Image Mode:**
- `--image`: Path to input image. If specified, processes single image instead of webcam
- `--output`: Output directory for processed images. Default: `output/`

**Webcam Mode:**
- `--camera`: Camera device index. Default: `0`
- `--width`: Camera capture width. Default: `640`
- `--height`: Camera capture height. Default: `480`

### Controls (Webcam Mode Only)

- **q**: Quit the application
- **s**: Save current frame and depth map (also saves contours and edges in contour mode)
- **c**: Toggle between depth view and contour view
- **+/=**: Increase edge detection threshold (makes contours less sensitive)
- **-/_**: Decrease edge detection threshold (makes contours more sensitive)

### Display Modes

**Depth Mode (default)**: Shows original frame on left, colorized depth map on right

**Contour Mode**: Shows original frame with depth contours on left, edge map on right
- Green contours highlight areas with significant depth changes
- Useful for detecting object boundaries, edges, and depth discontinuities

## Project Structure

```
camera-depth/
├── webcam_depth.py       # Main application script
├── requirements.txt      # Python dependencies
├── setup.sh             # Automated setup script
├── README.md            # This file
├── checkpoints/         # Model weights directory
│   └── depth_anything_v2_*.pth
└── output/              # Default output directory for processed images
    └── (generated files)
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

## How Depth Contour Detection Works

The contour detection feature identifies areas where depth changes significantly:

1. **Depth Gradients**: Calculates the gradient (rate of change) of depth values using Sobel operators
2. **Edge Detection**: Applies threshold to find significant depth discontinuities
3. **Contour Finding**: Identifies closed contours around regions with depth edges
4. **Filtering**: Removes small noise contours based on minimum area

This is useful for:
- Detecting object boundaries in 3D space
- Identifying obstacles and edges
- Segmenting objects based on depth
- Finding transitions between foreground and background

## References

- [Depth-Anything-V2 GitHub](https://github.com/DepthAnything/Depth-Anything-V2)
- [Depth-Anything-V2 Paper](https://arxiv.org/abs/2406.09414)
- [depth-anything-v2 on PyPI](https://pypi.org/project/depth-anything-v2/)
