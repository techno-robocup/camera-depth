#!/usr/bin/env python3
"""
Webcam Depth Detection using Depth-Anything-V2
Real-time depth estimation from webcam feed
"""

import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2
import argparse


def setup_model(model_size='small', device='cuda'):
    """
    Initialize Depth-Anything-V2 model

    Args:
        model_size: 'small', 'base', or 'large'
        device: 'cuda' or 'cpu'

    Returns:
        Depth-Anything-V2 model
    """
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }

    model_map = {
        'small': 'vits',
        'base': 'vitb',
        'large': 'vitl'
    }

    encoder = model_map.get(model_size, 'vits')
    model = DepthAnythingV2(**model_configs[encoder])

    # Load pretrained weights
    checkpoint = f'checkpoints/depth_anything_v2_{encoder}.pth'
    try:
        state_dict = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"Loaded model: {checkpoint}")
    except FileNotFoundError:
        print(f"Warning: Could not find checkpoint {checkpoint}")
        print("Please download the model weights from:")
        print(f"https://huggingface.co/depth-anything/Depth-Anything-V2-{model_size.capitalize()}/resolve/main/depth_anything_v2_{encoder}.pth")
        print(f"and place it in the 'checkpoints' directory")
        raise

    model = model.to(device).eval()
    return model


def colorize_depth(depth, cmap='inferno'):
    """
    Convert depth map to colorized visualization

    Args:
        depth: Depth map (numpy array)
        cmap: Colormap to use

    Returns:
        Colorized depth map (RGB)
    """
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_normalized = (depth_normalized * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
    return depth_colored


def process_frame(model, frame, device):
    """
    Process a single frame to generate depth map

    Args:
        model: Depth-Anything-V2 model
        frame: Input frame (BGR)
        device: torch device

    Returns:
        Depth map (numpy array)
    """
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Prepare input
    h, w = rgb_frame.shape[:2]

    # Inference
    with torch.no_grad():
        depth = model.infer_image(rgb_frame)

    return depth


def main():
    parser = argparse.ArgumentParser(description='Webcam Depth Detection using Depth-Anything-V2')
    parser.add_argument('--model-size', type=str, default='small',
                        choices=['small', 'base', 'large'],
                        help='Model size: small, base, or large')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use: cuda or cpu')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index')
    parser.add_argument('--width', type=int, default=640,
                        help='Camera capture width')
    parser.add_argument('--height', type=int, default=480,
                        help='Camera capture height')

    args = parser.parse_args()

    print(f"Using device: {args.device}")
    print(f"Loading model: {args.model_size}")

    # Initialize model
    model = setup_model(args.model_size, args.device)

    # Initialize webcam
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return

    print("Starting webcam depth detection...")
    print("Press 'q' to quit, 's' to save current depth map")

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            # Process frame
            depth = process_frame(model, frame, args.device)

            # Colorize depth
            depth_colored = colorize_depth(depth)

            # Resize depth map to match frame size
            depth_colored = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))

            # Create side-by-side display
            combined = np.hstack([frame, depth_colored])

            # Add FPS counter
            frame_count += 1
            if frame_count % 30 == 0:
                cv2.putText(combined, f"Frame: {frame_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display
            cv2.imshow('Webcam Depth Detection - Original | Depth', combined)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame and depth map
                cv2.imwrite(f'frame_{frame_count}.jpg', frame)
                cv2.imwrite(f'depth_{frame_count}.jpg', depth_colored)
                print(f"Saved frame_{frame_count}.jpg and depth_{frame_count}.jpg")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam released and windows closed")


if __name__ == '__main__':
    main()
