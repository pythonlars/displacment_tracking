import cv2
import numpy as np
import os

# ==================== CONFIGURATION ====================
FRAME1_PATH = 'frames2/78.png'
FRAME2_PATH = 'frames2/79.png'

# CLAHE Parameters
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)


# ==================== FUNCTIONS ====================

def load_images(frame1_path, frame2_path):
    """Load and convert frames to grayscale."""
    if not os.path.exists(frame1_path):
        raise FileNotFoundError(f"Frame 1 not found: {frame1_path}")
    if not os.path.exists(frame2_path):
        raise FileNotFoundError(f"Frame 2 not found: {frame2_path}")

    frame1 = cv2.imread(frame1_path)
    frame2 = cv2.imread(frame2_path)

    if frame1 is None:
        raise ValueError(f"Could not read frame 1: {frame1_path}")
    if frame2 is None:
        raise ValueError(f"Could not read frame 2: {frame2_path}")

    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

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


