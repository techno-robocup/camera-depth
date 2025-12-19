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


def detect_depth_edges(depth, threshold=0.1, blur_ksize=5):
    """
    Detect edges in depth map where there are significant depth discontinuities

    Args:
        depth: Depth map (numpy array)
        threshold: Threshold for edge detection (0-1, relative to depth range)
        blur_ksize: Kernel size for Gaussian blur preprocessing

    Returns:
        Binary edge map
    """
    # Normalize depth
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_normalized = (depth_normalized * 255).astype(np.uint8)

    # Apply Gaussian blur to reduce noise
    depth_blurred = cv2.GaussianBlur(depth_normalized, (blur_ksize, blur_ksize), 0)

    # Calculate gradients using Sobel
    grad_x = cv2.Sobel(depth_blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize gradient
    gradient_magnitude = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)

    # Apply threshold
    threshold_value = int(threshold * 255)
    _, edges = cv2.threshold(gradient_magnitude, threshold_value, 255, cv2.THRESH_BINARY)

    return edges


def find_depth_contours(edges, min_area=100):
    """
    Find contours in the edge map

    Args:
        edges: Binary edge map
        min_area: Minimum contour area to keep

    Returns:
        List of contours
    """
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by area
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    return filtered_contours


def draw_depth_contours(frame, contours, color=(0, 255, 0), thickness=2):
    """
    Draw contours on frame

    Args:
        frame: Frame to draw on
        contours: List of contours
        color: BGR color for contours
        thickness: Line thickness

    Returns:
        Frame with contours drawn
    """
    result = frame.copy()
    cv2.drawContours(result, contours, -1, color, thickness)
    return result


def process_image_file(image_path, model, device, output_dir, edge_threshold=0.1, min_contour_area=100):
    """
    Process a single image file and save all outputs

    Args:
        image_path: Path to input image
        model: Depth-Anything-V2 model
        device: torch device
        output_dir: Directory to save outputs
        edge_threshold: Threshold for edge detection
        min_contour_area: Minimum contour area
    """
    import os
    from pathlib import Path

    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image from {image_path}")
        return

    print(f"Processing image: {image_path}")
    print(f"Image size: {frame.shape[1]}x{frame.shape[0]}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get base filename
    base_name = Path(image_path).stem

    # Process frame
    print("Generating depth map...")
    depth = process_frame(model, frame, device)

    # Colorize depth
    depth_colored = colorize_depth(depth)
    depth_colored = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))

    # Detect edges and contours
    print("Detecting depth edges and contours...")
    edges = detect_depth_edges(depth, threshold=edge_threshold)
    edges_resized = cv2.resize(edges, (frame.shape[1], frame.shape[0]))
    contours = find_depth_contours(edges_resized, min_area=min_contour_area)

    # Draw contours
    frame_with_contours = draw_depth_contours(frame, contours, color=(0, 255, 0), thickness=2)

    # Save outputs
    print(f"Saving outputs to {output_dir}/")
    cv2.imwrite(str(output_path / f"{base_name}_original.jpg"), frame)
    cv2.imwrite(str(output_path / f"{base_name}_depth.jpg"), depth_colored)
    cv2.imwrite(str(output_path / f"{base_name}_contours.jpg"), frame_with_contours)
    cv2.imwrite(str(output_path / f"{base_name}_edges.jpg"), edges_resized)

    # Create side-by-side visualizations
    combined_depth = np.hstack([frame, depth_colored])
    edges_bgr = cv2.cvtColor(edges_resized, cv2.COLOR_GRAY2BGR)
    combined_contours = np.hstack([frame_with_contours, edges_bgr])

    cv2.imwrite(str(output_path / f"{base_name}_combined_depth.jpg"), combined_depth)
    cv2.imwrite(str(output_path / f"{base_name}_combined_contours.jpg"), combined_contours)

    print(f"\nSaved files:")
    print(f"  - {base_name}_original.jpg")
    print(f"  - {base_name}_depth.jpg")
    print(f"  - {base_name}_contours.jpg (found {len(contours)} contours)")
    print(f"  - {base_name}_edges.jpg")
    print(f"  - {base_name}_combined_depth.jpg")
    print(f"  - {base_name}_combined_contours.jpg")
    print("\nProcessing complete!")


def main():
    parser = argparse.ArgumentParser(description='Webcam Depth Detection using Depth-Anything-V2')
    parser.add_argument('--model-size', type=str, default='small',
                        choices=['small', 'base', 'large'],
                        help='Model size: small, base, or large')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use: cuda or cpu')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to input image (if not specified, uses webcam)')
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory for processed images (default: output/)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index')
    parser.add_argument('--width', type=int, default=640,
                        help='Camera capture width')
    parser.add_argument('--height', type=int, default=480,
                        help='Camera capture height')
    parser.add_argument('--edge-threshold', type=float, default=0.1,
                        help='Threshold for depth edge detection (0-1)')
    parser.add_argument('--min-contour-area', type=int, default=100,
                        help='Minimum contour area to display')

    args = parser.parse_args()

    print(f"Using device: {args.device}")
    print(f"Loading model: {args.model_size}")

    # Initialize model
    model = setup_model(args.model_size, args.device)

    # Check if processing a single image or webcam
    if args.image:
        # Process single image
        process_image_file(args.image, model, args.device, args.output,
                          args.edge_threshold, args.min_contour_area)
        return

    # Initialize webcam
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return

    print("Starting webcam depth detection...")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save current frame and depth map")
    print("  'c' - Toggle contour display mode")
    print("  '+' - Increase edge threshold")
    print("  '-' - Decrease edge threshold")

    frame_count = 0
    show_contours = False
    edge_threshold = args.edge_threshold

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

            # Detect depth edges and contours
            edges = detect_depth_edges(depth, threshold=edge_threshold)
            edges_resized = cv2.resize(edges, (frame.shape[1], frame.shape[0]))
            contours = find_depth_contours(edges_resized, min_area=args.min_contour_area)

            # Create display based on mode
            if show_contours:
                # Draw contours on original frame
                frame_with_contours = draw_depth_contours(frame, contours, color=(0, 255, 0), thickness=2)

                # Convert edges to BGR for display
                edges_bgr = cv2.cvtColor(edges_resized, cv2.COLOR_GRAY2BGR)

                # Create side-by-side display: Original with contours | Edge map
                combined = np.hstack([frame_with_contours, edges_bgr])
                window_title = 'Depth Contours - Frame with Contours | Edge Map'
            else:
                # Create side-by-side display: Original | Depth
                combined = np.hstack([frame, depth_colored])
                window_title = 'Webcam Depth Detection - Original | Depth'

            # Add info text
            frame_count += 1
            info_text = f"Frame: {frame_count} | Mode: {'Contours' if show_contours else 'Depth'}"
            if show_contours:
                info_text += f" | Threshold: {edge_threshold:.2f} | Contours: {len(contours)}"

            cv2.putText(combined, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Display
            cv2.imshow(window_title, combined)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame and depth map
                cv2.imwrite(f'frame_{frame_count}.jpg', frame)
                cv2.imwrite(f'depth_{frame_count}.jpg', depth_colored)
                if show_contours:
                    frame_with_contours_final = draw_depth_contours(frame, contours, color=(0, 255, 0), thickness=2)
                    cv2.imwrite(f'contours_{frame_count}.jpg', frame_with_contours_final)
                    cv2.imwrite(f'edges_{frame_count}.jpg', edges_resized)
                    print(f"Saved frame, depth, contours, and edges for frame {frame_count}")
                else:
                    print(f"Saved frame_{frame_count}.jpg and depth_{frame_count}.jpg")
            elif key == ord('c'):
                # Toggle contour mode
                show_contours = not show_contours
                cv2.destroyAllWindows()
                print(f"Contour mode: {'ON' if show_contours else 'OFF'}")
            elif key == ord('+') or key == ord('='):
                # Increase threshold
                edge_threshold = min(1.0, edge_threshold + 0.01)
                print(f"Edge threshold: {edge_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):
                # Decrease threshold
                edge_threshold = max(0.01, edge_threshold - 0.01)
                print(f"Edge threshold: {edge_threshold:.2f}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam released and windows closed")


if __name__ == '__main__':
    main()
