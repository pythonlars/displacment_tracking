import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ==================== CONFIGURATION ====================
FRAME1_PATH = 'frames2/78.png'
FRAME2_PATH = 'frames2/79.png'

# CLAHE Parameters
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)

# Processing Options
USE_CLAHE = False  # Set to True to use CLAHE preprocessing


# ==================== FUNCTIONS ====================

def load_images(frame1_path, frame2_path):
    """Load and convert frames to grayscale."""
    if not os.path.exists(frame1_path):
        raise FileNotFoundError(f"Frame 1 not found: {frame1_path}")
    if not os.path.exists(frame2_path):
        raise FileNotFoundError(f"Frame 2 not found: {frame2_path}")

    gray1 = cv2.imread(frame1_path, 0)
    gray2 = cv2.imread(frame2_path, 0)

    if gray1 is None:
        raise ValueError(f"Could not read frame 1: {frame1_path}")
    if gray2 is None:
        raise ValueError(f"Could not read frame 2: {frame2_path}")

    # Validate dimensions match
    if gray1.shape != gray2.shape:
        raise ValueError(f"Frame dimensions don't match: {gray1.shape} vs {gray2.shape}")
    return gray1, gray2


def preprocess_with_clahe(gray_image, clip_limit=CLAHE_CLIP_LIMIT, tile_size=CLAHE_TILE_SIZE):
    """Apply CLAHE preprocessing to enhance contrast."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    enhanced = clahe.apply(gray_image)
    return enhanced


def calculate_phase_correlation(frame1, frame2):
    """Calculate global displacement using phase correlation."""
    # Convert to float32 for phase correlation
    frame1_f = np.float32(frame1)
    frame2_f = np.float32(frame2)

    # Calculate phase correlation
    shift, confidence = cv2.phaseCorrelate(frame1_f, frame2_f)

    return shift, confidence


def interpret_displacement(shift, confidence):
    """
    Interpret phase correlation results and return formatted information.

    Args:
        shift: (dx, dy) tuple from phaseCorrelate
        confidence: confidence value from phaseCorrelate

    Returns:
        dict with interpretation details
    """
    dx, dy = shift
    magnitude = np.sqrt(dx**2 + dy**2)

    # Determine direction labels for frame shift
    x_direction = "right" if dx > 0 else "left" if dx < 0 else "none"
    y_direction = "down" if dy > 0 else "up" if dy < 0 else "none"

    # Camera movement is opposite to frame displacement
    camera_x = "left" if dx > 0 else "right" if dx < 0 else "stationary"
    camera_y = "up" if dy > 0 else "down" if dy < 0 else "stationary"

    return {
        'dx': dx,
        'dy': dy,
        'magnitude': magnitude,
        'confidence': confidence,
        'x_direction': x_direction,
        'y_direction': y_direction,
        'camera_x': camera_x,
        'camera_y': camera_y
    }


def print_results(frame1_path, frame2_path, result_dict):
    """Print displacement results in human-readable format."""
    print("\n" + "="*50)
    print("DISPLACEMENT TRACKING RESULTS")
    print("="*50)
    print(f"Frame 1: {frame1_path}")
    print(f"Frame 2: {frame2_path}")
    print()

    print("Displacement Vector (frame shift):")
    print(f"  X-axis: {result_dict['dx']:+.2f} pixels ({result_dict['x_direction']})")
    print(f"  Y-axis: {result_dict['dy']:+.2f} pixels ({result_dict['y_direction']})")
    print()

    print("Total Displacement:")
    print(f"  Magnitude: {result_dict['magnitude']:.2f} pixels")
    print()

    print(f"Confidence: {result_dict['confidence']:.4f} ({result_dict['confidence']*100:.1f}%)")
    print()

    # Camera movement interpretation
    if result_dict['camera_x'] != 'stationary' or result_dict['camera_y'] != 'stationary':
        camera_movement = []
        if result_dict['camera_x'] != 'stationary':
            camera_movement.append(result_dict['camera_x'].upper())
        if result_dict['camera_y'] != 'stationary':
            camera_movement.append(result_dict['camera_y'].upper())
        print(f"Camera Movement: {' and '.join(camera_movement)}")
    else:
        print("Camera Movement: STATIONARY")

    print("="*50 + "\n")


def visualize_displacement(frame1_path, frame2_path, shift, output_path='displacement_visualization.png'):
    """
    Create a visualization showing displacement with overlay and arrows.

    Args:
        frame1_path: Path to first frame
        frame2_path: Path to second frame
        shift: (dx, dy) displacement tuple
        output_path: Where to save the visualization
    """
    # Load frames in color
    frame1 = cv2.imread(frame1_path)
    frame2 = cv2.imread(frame2_path)

    dx, dy = shift
    dx_int, dy_int = int(round(dx)), int(round(dy))

    height, width = frame1.shape[:2]

    # Create canvas for visualization
    # Make it larger to accommodate the shift
    canvas_height = height + abs(dy_int)
    canvas_width = width + abs(dx_int)

    # Create two canvases for overlaying
    canvas1 = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas2 = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Position frame1 at origin position
    y1_start = max(0, -dy_int)
    x1_start = max(0, -dx_int)
    canvas1[y1_start:y1_start+height, x1_start:x1_start+width] = frame1

    # Position frame2 shifted by displacement
    y2_start = max(0, dy_int)
    x2_start = max(0, dx_int)
    canvas2[y2_start:y2_start+height, x2_start:x2_start+width] = frame2

    # Create overlay with transparency (50% each frame)
    overlay = cv2.addWeighted(canvas1, 0.5, canvas2, 0.5, 0)

    # Draw white borders showing non-overlapping regions
    # These are the areas where only one frame exists
    mask1 = cv2.cvtColor(canvas1, cv2.COLOR_BGR2GRAY) > 0
    mask2 = cv2.cvtColor(canvas2, cv2.COLOR_BGR2GRAY) > 0

    # Non-overlapping regions (XOR operation)
    non_overlap = np.logical_xor(mask1, mask2)

    # Draw white border around non-overlapping regions
    overlay[non_overlap] = [255, 255, 255]  # White color

    # Add a few arrows showing displacement direction
    # Place arrows at bottom of the image
    arrow_color = (255, 255, 255)  # White
    arrow_thickness = 3

    # Calculate arrow positions (3 arrows at bottom)
    arrow_y = canvas_height - 50  # 50 pixels from bottom
    arrow_positions = [
        canvas_width // 4,
        canvas_width // 2,
        3 * canvas_width // 4
    ]

    # Draw arrows
    magnitude = np.sqrt(dx**2 + dy**2)
    for arrow_x in arrow_positions:
        start_point = (arrow_x, arrow_y)
        # Scale the arrow to make it visible (multiply by factor)
        # Use a minimum arrow length of 30 pixels for visibility
        if magnitude > 0.1:
            arrow_scale = max(30 / magnitude, 3)  # Ensure minimum visible length
            end_point = (arrow_x + int(dx * arrow_scale),
                         arrow_y + int(dy * arrow_scale))
        else:
            # Displacement too small to show direction, draw a dot/small marker
            end_point = start_point

        # Draw arrow (or point if no movement)
        if start_point != end_point:
            cv2.arrowedLine(overlay, start_point, end_point,
                           arrow_color, arrow_thickness, tipLength=0.3)
        else:
            cv2.circle(overlay, start_point, 5, arrow_color, -1)

    # Add text showing displacement values
    text = f"Displacement: dx={dx:.1f}px, dy={dy:.1f}px"
    cv2.putText(overlay, text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Save the visualization
    cv2.imwrite(output_path, overlay)
    print(f"Visualization saved to: {output_path}")

    # Display the visualization
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(14, 8))
    plt.imshow(overlay_rgb)
    plt.title(f"Displacement: dx={dx:.2f}px, dy={dy:.2f}px")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return overlay


def process_two_frames(frame1_path, frame2_path, use_clahe=USE_CLAHE):
    """
    Process two frames and calculate displacement.

    Args:
        frame1_path: Path to first frame
        frame2_path: Path to second frame
        use_clahe: Whether to apply CLAHE preprocessing

    Returns:
        result_dict: Dictionary with displacement information
    """
    # Load images
    print(f"Loading frames...")
    gray1, gray2 = load_images(frame1_path, frame2_path)
    print(f"  Frame dimensions: {gray1.shape}")

    # Optionally apply CLAHE
    if use_clahe:
        print(f"Applying CLAHE preprocessing...")
        processed1 = preprocess_with_clahe(gray1)
        processed2 = preprocess_with_clahe(gray2)
    else:
        print(f"Using raw grayscale (CLAHE disabled)...")
        processed1 = gray1
        processed2 = gray2

    # Calculate phase correlation
    print(f"Calculating phase correlation...")
    shift, confidence = calculate_phase_correlation(processed1, processed2)

    # Interpret results
    result_dict = interpret_displacement(shift, confidence)

    return result_dict


# ==================== MAIN EXECUTION ====================

if __name__ == '__main__':
    try:
        # Process the two frames
        results = process_two_frames(FRAME1_PATH, FRAME2_PATH)

        # Print results
        print_results(FRAME1_PATH, FRAME2_PATH, results)

        # Create visualization
        print("\nCreating displacement visualization...")
        shift = (results['dx'], results['dy'])
        visualize_displacement(FRAME1_PATH, FRAME2_PATH, shift)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
