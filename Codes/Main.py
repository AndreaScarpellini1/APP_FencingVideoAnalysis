import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results
import numpy as np


# =========================
# Configuration
# =========================

# VIDEO_FILENAME = "100vs100_HMN1_MT15_TN23_CR15-12.MP4"
VIDEO_FILENAME = "12vs5_HMN1_MT15_TN5_CR5-1.MP4"
MODEL_ONE_NAME = "Model_One_fp16.pt"
MODEL_SCOREBOARD = "Model_Scoreboard_fp16.pt"
MODEL_POINT_DETECTOR = "Model_PointDetector21_fp16.pt"
OUTPUT_VIDEO_NAME = "annotated_combined_optimized.mp4" # Changed name to reflect optimization
BATCH_SIZE = 32  # you can tweak this if needed


# =========================
# Path & environment helpers
# =========================

def get_project_paths() -> Tuple[Path, Path, Path, Path, Path, Path]:
    """
    Returns:
        base_dir: project root (one level above Codes/)
        input_video: full path to input video
        model_one_path: full path to Model_One.pt
        model_scoreboard_path: full path to Model_Scoreboard.pt
        model_point_detector_path: full path to Model_PointDetector.pt
        output_dir: Output folder path
    """
    base_dir = Path(__file__).resolve().parents[1]
    input_dir = base_dir / "Input"
    models_dir = base_dir / "Models"
    output_dir = base_dir / "Output"

    input_video = input_dir / VIDEO_FILENAME
    model_one_path = models_dir / MODEL_ONE_NAME
    model_scoreboard_path = models_dir / MODEL_SCOREBOARD
    model_point_detector_path = models_dir / MODEL_POINT_DETECTOR

    return (
        base_dir,
        input_video,
        model_one_path,
        model_scoreboard_path,
        model_point_detector_path,
        output_dir,
    )


def check_file_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


# =========================
# Device & model loading
# =========================

def get_device() -> str:
    """
    Choose the best available device.
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


def load_yolo_model(model_path: Path) -> YOLO:
    """
    Load a YOLO model from disk.
    """
    print(f"Loading model from: {model_path}")
    # Device setting is handled during inference call
    model = YOLO(str(model_path))
    return model


# =========================
# Video I/O helpers
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


def create_video_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer at: {output_path}")
    print(f"\nWriting output video to: {output_path}")
    return writer


def read_frame_batch(cap: cv2.VideoCapture, batch_size: int) -> List[np.ndarray]:
    """
    Read up to batch_size frames from the video.
    Returns:
        frames: list of BGR numpy arrays
    """
    frames = []
    for _ in range(batch_size):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames


# =========================
# Drawing & annotation
# =========================

def draw_detections_on_frame(
    frame: np.ndarray,
    result: Results,
    color: Tuple[int, int, int],
    label_prefix: str = "",
):
    """
    Draw bounding boxes and labels from a single YOLO Result on the given frame.
    """
    boxes = result.boxes
    if boxes is None or boxes.xyxy is None:
        return

    # Move tensor data to CPU and convert to numpy
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    names = result.names

    for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
        # Convert coordinates to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = f"{label_prefix}{names[c]} {p:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

def draw_static_box_on_frame(
    frame: np.ndarray,
    box_coords: List[int],
    class_name: str,
    color: Tuple[int, int, int],
    label_prefix: str = "",
):
    """
    Draw a fixed bounding box and label (like the scoreboard) on the frame.
    box_coords: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = [int(c) for c in box_coords]
    label = f"{label_prefix}{class_name}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 5, 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def annotate_batch_with_models(
    frames: List[np.ndarray],
    results_one: List[Results],
    scoreboard_box_coords: Optional[List[int]],
    results_point: Optional[List[Results]],
):
    """
    Draw detections from three models on each frame in-place.
    - Model_One: green, prefix 'M1:'
    - Model_Scoreboard: static cyan box (if available)
    - Model_PointDetector: red, prefix 'PD:'
    """
    for i, frame in enumerate(frames):
        # 1. Model One detections
        draw_detections_on_frame(frame, results_one[i], color=(0, 255, 0), label_prefix="M1:") # Green

        # 2. Scoreboard static box (used for cropping)
        if scoreboard_box_coords is not None:
            # Draw the static box
            draw_static_box_on_frame(
                frame,
                scoreboard_box_coords,
                "Scoreboard",
                color=(255, 255, 0), # Cyan/Yellow
                label_prefix="SB:",
            )

            # 3. Point Detector results (if available)
            if results_point is not None:
                # Need to offset the point detector results back to the original frame's coordinates
                x_offset, y_offset, _, _ = scoreboard_box_coords
                
                # Check if the result is valid for this frame (it should be, since we ran it batched)
                result_point = results_point[i]
                
                if result_point.boxes is not None and result_point.boxes.xyxy is not None:
                    # Create a temporary deep copy of the boxes to modify coordinates
                    point_boxes = result_point.boxes.cpu() 
                    
                    # Apply offset to the coordinates (x1, y1, x2, y2)
                    offset_tensor = torch.tensor([x_offset, y_offset, x_offset, y_offset], dtype=point_boxes.xyxy.dtype)
                    point_boxes.xyxy += offset_tensor.to(point_boxes.xyxy.device)
                    
                    # Create a temporary modified result object for drawing
                    # Note: We're modifying the boxes on the CPU copy, which is okay for drawing
                    temp_result = Results(
                        orig_img=result_point.orig_img,
                        path=result_point.path,
                        names=result_point.names,
                        boxes=point_boxes
                    )
                    
                    draw_detections_on_frame(frame, temp_result, color=(0, 0, 255), label_prefix="PD:") # Red


# =========================
# Core processing logic
# =========================

def process_video_with_three_models(
    video_path: Path,
    model_one: YOLO,
    model_scoreboard: YOLO,
    model_point_detector: YOLO,
    output_video_path: Path,
    device: str,
    batch_size: int,
) -> Tuple[int, float]:
    """
    Main loop with optimized logic for static scoreboard and cropped point detection.
    """
    cap = open_video_capture(video_path)
    fps, width, height, total_frames = get_video_properties(cap)
    writer = create_video_writer(output_video_path, fps, width, height)

    processed_frames = 0
    start_time = time.time()
    
    # State variables for optimization
    scoreboard_box_coords: Optional[List[int]] = None # [x1, y1, x2, y2]
    
    # --- 1. Find the static scoreboard box on the first frame ---
    print("\n--- Initializing: Detecting static scoreboard on first frame ---")
    
    # Read the first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        cap.release()
        writer.release()
        return 0, 0.0

    # Run the scoreboard model on the first frame
    # We set imgsz to a small value (e.g., 640) for a standard YOLO detection size
    # and batch=1 for single-frame inference.
    scoreboard_results = model_scoreboard(first_frame, device=device, verbose=False, imgsz=640)
    
    if scoreboard_results and scoreboard_results[0].boxes and len(scoreboard_results[0].boxes) > 0:
        # Get the first detection's bounding box coordinates (x1, y1, x2, y2)
        # Assuming the model only detects the single main scoreboard
        coords_tensor = scoreboard_results[0].boxes.xyxy[0].cpu().numpy()
        scoreboard_box_coords = [int(c) for c in coords_tensor]
        print(f"✅ Scoreboard detected: {scoreboard_box_coords}")
    else:
        print("❌ Scoreboard not detected in the first frame. Point detection optimization will be skipped.")

    # Reset video to the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print("Video capture reset to frame 0.")
    
    # --- 2. Main processing loop ---
    try:
        while True:
            frames = read_frame_batch(cap, batch_size)
            if not frames:
                break

            # 1. Run Model_One (full frame, batched)
            # Use the specified imgsz = 512 for Model_One
            results_one = model_one(frames, device=device, verbose=False, imgsz=512)
            
            # 2. Run Model_PointDetector (cropped frame, batched)
            results_point = None
            if scoreboard_box_coords is not None:
                x1, y1, x2, y2 = scoreboard_box_coords
                
                # Crop the batch of frames
                cropped_frames = [f[y1:y2, x1:x2] for f in frames]
                
                # Filter out any frames that resulted in an empty crop (shouldn't happen if coords are correct)
                # and skip the detection if no valid crops remain.
                valid_crops = [crop for crop in cropped_frames if crop.size > 0]
                
                if valid_crops:
                    # Run the point detector on the cropped frames.
                    # Use a small imgsz (e.g., 256 or 320) as the input is already small.
                    results_point = model_point_detector(valid_crops, device=device, verbose=False, imgsz=320)
                else:
                     print("Warning: Empty crop detected, skipping Point Detector for this batch.")


            # 3. Annotate frames in-place (handles coordinate offsetting for PD results)
            annotate_batch_with_models(frames, results_one, scoreboard_box_coords, results_point)

            # Write annotated frames
            for f in frames:
                writer.write(f)

            processed_frames += len(frames)
            print(f"Processed {processed_frames}/{total_frames} frames...", end="\r")

    finally:
        cap.release()
        writer.release()

    elapsed_time = time.time() - start_time

    print(f"\nDone! Combined annotated video saved at:\n{output_video_path}\n")

    return processed_frames, elapsed_time


# =========================
# Entry point
# =========================

def main():
    try:
        (
            base_dir,
            input_video,
            model_one_path,
            model_scoreboard_path,
            model_point_detector_path,
            output_dir,
        ) = get_project_paths()

        # Sanity checks
        check_file_exists(input_video, "Input video")
        check_file_exists(model_one_path, "Model_One")
        check_file_exists(model_scoreboard_path, "Model_Scoreboard")
        check_file_exists(model_point_detector_path, "Model_PointDetector21")

        device = get_device()

        # Load models
        model_one = load_yolo_model(model_one_path)
        model_scoreboard = load_yolo_model(model_scoreboard_path)
        model_point_detector = load_yolo_model(model_point_detector_path)

        # Output video path
        output_video_path = output_dir / OUTPUT_VIDEO_NAME

        # --- Run processing and measure stats ---
        processed_frames, elapsed_time = process_video_with_three_models(
            video_path=input_video,
            model_one=model_one,
            model_scoreboard=model_scoreboard,
            model_point_detector=model_point_detector,
            output_video_path=output_video_path,
            device=device,
            batch_size=BATCH_SIZE,
        )

        # Avoid division by zero
        effective_fps = processed_frames / elapsed_time if elapsed_time > 0 else 0.0

        print("========== SUMMARY ==========")
        print(f"Total frames processed: {processed_frames}")
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Effective processing speed: {effective_fps:.2f} frames/second")
        print("=============================")
        print("All done ✅")

    except Exception as e:
        print("\n❌ An error occurred:")
        print(e)
        # raise # Uncomment for full traceback during debugging


if __name__ == "__main__":
    main()