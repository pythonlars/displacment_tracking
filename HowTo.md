# How to Complete Your Displacement Tracking Code

This guide will walk you through completing the `displacment_tracking.py` code to calculate displacement between two video frames.

## What You're Building

Your code will:
- Load two consecutive frames (78.png and 79.png)
- Calculate how far the camera moved between frames
- Show the displacement in pixels (X and Y direction)
- Display the total distance and confidence score

## Step-by-Step Instructions

### Step 1: Add Configuration Flag

Find line 13 in your code (right after `CLAHE_TILE_SIZE = (8, 8)`), and add:

```python
# Processing Options
USE_CLAHE = False  # Set to True to use CLAHE preprocessing
```

**Why?** This lets you toggle between using CLAHE (contrast enhancement) or raw grayscale. Phase correlation usually works better without CLAHE.

---

### Step 2: Add Result Interpretation Function

Go to the end of your functions section (after the `calculate_phase_correlation` function, around line 57), and add this function:

```python
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
```

**What this does:** Takes the raw phase correlation results and calculates:
- dx, dy: displacement in pixels
- magnitude: total distance moved
- Direction labels (human-readable)
- Camera movement (opposite of how the image shifted)

---

### Step 3: Add Output Formatting Function

Right after the function you just added, insert this:

```python
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
```

**What this does:** Formats and prints all the displacement information in a nice, readable way.

---

### Step 4: Add Main Processing Function

Add this function next:

```python
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
```

**What this does:** This is the main workflow that:
1. Loads your two frames
2. Optionally applies CLAHE
3. Calculates phase correlation
4. Interprets the results

---

### Step 5: Add Main Execution Block

Finally, at the very end of the file, add:

```python
# ==================== MAIN EXECUTION ====================

if __name__ == '__main__':
    try:
        # Process the two frames
        results = process_two_frames(FRAME1_PATH, FRAME2_PATH)

        # Print results
        print_results(FRAME1_PATH, FRAME2_PATH, results)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
```

**What this does:** Runs the code when you execute the script, with error handling.

---

## How to Run It

1. **Save your file** after adding all the code above

2. **Open terminal** in your project directory:
   ```bash
   cd /Users/larsanderegg/Documents/displacement-tracking
   ```

3. **Run the script:**
   ```bash
   python displacment_tracking.py
   ```

---

## Expected Output

You should see something like this:

```
Loading frames...
  Frame dimensions: (720, 1280)
Using raw grayscale (CLAHE disabled)...
Calculating phase correlation...

==================================================
DISPLACEMENT TRACKING RESULTS
==================================================
Frame 1: frames2/78.png
Frame 2: frames2/79.png

Displacement Vector (frame shift):
  X-axis: -12.34 pixels (left)
  Y-axis: +5.67 pixels (down)

Total Displacement:
  Magnitude: 13.58 pixels

Confidence: 0.7823 (78.2%)

Camera Movement: RIGHT and UP
==================================================
```

---

## Understanding the Results

### Direction Convention

**Frame displacement** (what the code directly calculates):
- **Positive X** = frame shifted right
- **Negative X** = frame shifted left
- **Positive Y** = frame shifted down
- **Negative Y** = frame shifted up

**Camera movement** (the inverse):
- If frame shifted right → camera moved left
- If frame shifted left → camera moved right
- If frame shifted down → camera moved up
- If frame shifted up → camera moved down

### Confidence Score

- **0.0 - 0.5**: Poor match, results unreliable
- **0.5 - 0.7**: Moderate match, results likely correct
- **0.7 - 1.0**: Good match, results very reliable

---

## Testing Different Settings

### Try With CLAHE

Change line 14 to:
```python
USE_CLAHE = True
```

Then run again and compare the confidence scores. Usually raw grayscale works better, but CLAHE can help with low-contrast images.

---

## Troubleshooting

### Problem: "No module named cv2"
**Solution:** Install OpenCV:
```bash
pip install opencv-python
```

### Problem: "No module named matplotlib"
**Solution:** Install matplotlib:
```bash
pip install matplotlib
```

### Problem: Low confidence scores (< 0.5)
**Possible causes:**
- Too much motion blur
- Not enough texture/detail in the images
- Camera rotated between frames (phase correlation only handles translation)

**Try:** Enable CLAHE to see if it improves

### Problem: "File not found"
**Solution:** Make sure frames2/78.png and frames2/79.png exist in your project folder

---

## Next Steps: Processing Entire Videos

Once this works, you can extend it to process entire videos. Here's the concept:

```python
def process_video(video_path):
    """Process entire video frame by frame."""
    cap = cv2.VideoCapture(video_path)

    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_num = 0
    all_results = []

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Calculate displacement
        shift, confidence = calculate_phase_correlation(prev_gray, curr_gray)
        result = interpret_displacement(shift, confidence)

        # Store results
        all_results.append({
            'frame': frame_num,
            'dx': result['dx'],
            'dy': result['dy'],
            'magnitude': result['magnitude'],
            'confidence': result['confidence']
        })

        # Update for next iteration
        prev_gray = curr_gray
        frame_num += 1

    cap.release()
    return all_results
```

You can then save results to CSV or plot the camera trajectory.

---

## Visualizing Displacement with Overlay and Arrows

To create a visual representation like your example image (showing transparent overlay, white borders for displacement, and arrows), add this function to your code:

### Step 6: Add Visualization Function

Add this function after the `print_results()` function:

```python
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
    for arrow_x in arrow_positions:
        start_point = (arrow_x, arrow_y)
        # Scale the arrow to make it visible (multiply by factor)
        arrow_scale = 3
        end_point = (arrow_x + int(dx * arrow_scale),
                     arrow_y + int(dy * arrow_scale))

        # Draw arrow
        cv2.arrowedLine(overlay, start_point, end_point,
                       arrow_color, arrow_thickness, tipLength=0.3)

    # Add text showing displacement values
    text = f"Displacement: dx={dx:.1f}px, dy={dy:.1f}px"
    cv2.putText(overlay, text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Save the visualization
    cv2.imwrite(output_path, overlay)
    print(f"Visualization saved to: {output_path}")

    return overlay
```

### Step 7: Update Main Execution to Include Visualization

Modify your `if __name__ == '__main__':` block to create the visualization:

```python
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
```

### What the Visualization Shows

1. **Transparent Overlay**: Both frames overlaid with 50% transparency so you can see how they align
2. **White Borders**: The non-overlapping regions (areas that don't match between frames) are highlighted in white, showing how far the displacement is
3. **White Arrows**: Three arrows at the bottom showing the direction and magnitude of displacement
4. **Text Label**: Displacement values shown at the top

### Output File

After running the script, you'll get:
- Console output with displacement data
- A file called `displacement_visualization.png` showing the visual overlay

### Customization Options

You can modify these parameters in the `visualize_displacement()` function:

**Arrow appearance:**
```python
arrow_scale = 3  # Make arrows longer/shorter
arrow_thickness = 3  # Make arrows thicker/thinner
arrow_color = (255, 255, 255)  # Change color (B, G, R)
```

**Number of arrows:**
```python
# For 5 arrows instead of 3:
arrow_positions = [
    canvas_width // 6,
    2 * canvas_width // 6,
    3 * canvas_width // 6,
    4 * canvas_width // 6,
    5 * canvas_width // 6
]
```

**Transparency levels:**
```python
# In the addWeighted line, change the weights:
overlay = cv2.addWeighted(canvas1, 0.7, canvas2, 0.3, 0)  # Frame1 more visible
overlay = cv2.addWeighted(canvas1, 0.3, canvas2, 0.7, 0)  # Frame2 more visible
```

**Border thickness:**
To make the white borders thicker, you can dilate the non-overlap mask:
```python
# Add this before the line: overlay[non_overlap] = [255, 255, 255]
kernel = np.ones((5, 5), np.uint8)
non_overlap = cv2.dilate(non_overlap.astype(np.uint8), kernel, iterations=1).astype(bool)
```

### Viewing the Result

Open the `displacement_visualization.png` file to see:
- How much the frames overlap
- The direction of camera movement (shown by arrows)
- The displacement distance (shown by white non-overlapping areas)

---

## Summary Checklist

- [ ] Added `USE_CLAHE` configuration flag
- [ ] Added `interpret_displacement()` function
- [ ] Added `print_results()` function
- [ ] Added `process_two_frames()` function
- [ ] Added `if __name__ == '__main__':` block
- [ ] Added `visualize_displacement()` function (for visualization)
- [ ] Updated main execution block to create visualization
- [ ] Saved the file
- [ ] Ran `python displacment_tracking.py`
- [ ] Got output showing displacement results
- [ ] Got visualization image `displacement_visualization.png`

Good luck! Let me know if you run into any issues.
