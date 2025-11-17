import sys
import time
import random
from pathlib import Path
from typing import Tuple

import cv2
import torch
import numpy as np
from ultralytics import YOLO


# =========================
# Configuration
# =========================

VIDEO_FILENAME = "100vs100_HMN1_MT15_TN23_CR15-12.MP4"
MODEL_ONE_NAME = "Model_One_fp16.pt"
CROP_SIZE = 1280  

# =========================
# Path & environment helpers
# =========================

def get_project_paths() -> Tuple[Path, Path, Path]:
    """
    Returns:
        base_dir: project root (one level above Codes/)
        input_video: full path to input video
        model_one_path: full path to Model_One.pt
    """
    base_dir = Path(__file__).resolve().parents[1]
    input_dir = base_dir / "Input"
    models_dir = base_dir / "Models"

    input_video = input_dir / VIDEO_FILENAME
    model_one_path = models_dir / MODEL_ONE_NAME

    return base_dir, input_video, model_one_path


def check_file_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


# =========================
# Device & model loading
# =========================

def get_device() -> str:
    """
    Choose the best available device:
    - 'mps' on Mac if available (Apple GPU)
    - 'cuda' on NVIDIA machines
    - otherwise 'cpu'
    """
    if torch.backends.mps.is_available():
        print("Using Apple MPS GPU ✅")
        return "mps"
    elif torch.cuda.is_available():
        print("Using NVIDIA CUDA GPU ✅")
        return "cuda"
    else:
        print("No GPU available, using CPU ⚠️")
        return "cpu"


def load_yolo_model(model_path: Path, device: str) -> YOLO:
    """
    Load a YOLO model from disk.
    """
    print(f"Loading model from: {model_path}")
    model = YOLO(str(model_path))
    return model


# =========================
# Video helpers
# =========================

def open_video_capture(video_path: Path) -> cv2.VideoCapture:
    print(f"\nOpening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    return cap


def get_video_properties(cap: cv2.VideoCapture) -> Tuple[float, int, int, int]:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25  # fallback

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties -> FPS: {fps}, Size: {width}x{height}, Frames: {total_frames}")
    return fps, width, height, total_frames


# =========================
# Detection & cropping
# =========================

def get_top_two_detections(result) -> Tuple[np.ndarray, np.ndarray]:
    """
    From a YOLO Result, return the 2 bounding boxes with highest confidence.

    Returns:
        boxes_xyxy: (2, 4) array with [x1, y1, x2, y2]
        confs: (2,) array of confidences

    Raises:
        ValueError if fewer than 2 detections are available.
    """
    boxes = result.boxes
    if boxes is None or boxes.xyxy is None or len(boxes) < 2:
        raise ValueError("Fewer than 2 detections found in this frame.")

    xyxy = boxes.xyxy.cpu().numpy()  # (N, 4)
    conf = boxes.conf.cpu().numpy()  # (N,)

    # Get indices of top 2 confidences (highest first)
    idx_sorted = np.argsort(-conf)
    top_two_idx = idx_sorted[:2]

    return xyxy[top_two_idx], conf[top_two_idx]


def compute_geometric_center_from_boxes(boxes_xyxy: np.ndarray) -> Tuple[float, float]:
    """
    Compute the geometric center between two fencer bounding boxes.

    For each box [x1, y1, x2, y2]:
        fencer center = ((x1 + x2)/2, (y1 + y2)/2)

    Geometric center = average of the two fencer centers.
    """
    centers = []
    for box in boxes_xyxy:
        x1, y1, x2, y2 = box
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        centers.append((cx, cy))

    # Average of the two centers
    cx = 0.5 * (centers[0][0] + centers[1][0])
    cy = 0.5 * (centers[0][1] + centers[1][1])
    return cx, cy


def crop_square_around_center(
    frame: np.ndarray,
    center_x: float,
    center_y: float,
    crop_size: int = 640,
) -> np.ndarray:
    """
    Crop a square region of `crop_size x crop_size` around (center_x, center_y),
    clamping to the frame boundaries.

    The *intended* center of the square is exactly (center_x, center_y).
    If the square would go out of bounds, we shift it just enough to stay inside.
    """
    h, w = frame.shape[:2]
    half = crop_size // 2

    cx = int(round(center_x))
    cy = int(round(center_y))

    x1 = cx - half
    x2 = cx + half
    y1 = cy - half
    y2 = cy + half

    # Adjust if out of bounds (while keeping the square size)
    if x1 < 0:
        x2 -= x1  # shift right
        x1 = 0
    if y1 < 0:
        y2 -= y1  # shift down
        y1 = 0
    if x2 > w:
        shift = x2 - w
        x1 -= shift
        x2 = w
    if y2 > h:
        shift = y2 - h
        y1 -= shift
        y2 = h

    # Final clamps
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    cropped = frame[y1:y2, x1:x2].copy()
    return cropped


# =========================
# Display helper (scaled to fit window)
# =========================

def show_image(window_name: str, img: np.ndarray, max_width: int = 1280, max_height: int = 720):
    """
    Show an image in a resizable window, scaling it down if it's larger than
    max_width x max_height, while preserving aspect ratio.
    """
    h, w = img.shape[:2]

    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        img_to_show = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    else:
        img_to_show = img

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img_to_show)
    print(f"Showing window: {window_name} (press any key to close)")
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


# =========================
# Core logic
# =========================

def run_on_random_frame(video_path: Path, model_one: YOLO, device: str, crop_size: int = CROP_SIZE):
    cap = open_video_capture(video_path)
    fps, width, height, total_frames = get_video_properties(cap)

    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Video appears to have zero frames.")

    # Pick a random frame index
    random_frame_idx = random.randint(0, total_frames - 1)
    print(f"Randomly selected frame index: {random_frame_idx}")

    # Seek and read that frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Could not read frame at index {random_frame_idx}.")

    # Run Model_One on this single frame
    print("Running Model_One inference on selected frame...")
    results = model_one(frame, device=device, verbose=False)
    result = results[0]

    # Get top two detections (2 fencers with highest confidence)
    try:
        boxes_xyxy, confs = get_top_two_detections(result)
        print("Top 2 detection confidences:", confs)
    except ValueError as e:
        print(f"Warning: {e}")
        print("Showing the original frame instead (no crop).")
        show_image("Original frame (fewer than 2 detections)", frame)
        return

    # Compute geometric center between the 2 fencers (using their box centers)
    center_x, center_y = compute_geometric_center_from_boxes(boxes_xyxy)
    print(f"Geometric center between 2 fencers: ({center_x:.1f}, {center_y:.1f})")

    # Crop square 640x640 around that center
    cropped = crop_square_around_center(frame, center_x, center_y, crop_size=crop_size)

    # Draw the 2 boxes and the center on the original frame for visualization
    for (x1, y1, x2, y2) in boxes_xyxy:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.circle(frame, (int(round(center_x)), int(round(center_y))), 5, (0, 0, 255), -1)

    # Show both frames (scaled to fit the window)
    show_image("Original frame with 2 fencers and geometric center", frame)
    show_image(f"Cropped {crop_size}x{crop_size} square around center", cropped)


# =========================
# Entry point
# =========================

def main():
    try:
        base_dir, input_video, model_one_path = get_project_paths()

        # Sanity checks
        check_file_exists(input_video, "Input video")
        check_file_exists(model_one_path, "Model_One")

        device = get_device()

        # Load Model_One
        model_one = load_yolo_model(model_one_path, device)

        # Run logic on a single random frame
        run_on_random_frame(
            video_path=input_video,
            model_one=model_one,
            device=device,
            crop_size=CROP_SIZE,
        )

        print("All done ✅")

    except Exception as e:
        print("\n❌ An error occurred:")
        print(e)
        # raise  # Uncomment if you want full traceback


if __name__ == "__main__":
    main()
