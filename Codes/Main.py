import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import torch
from ultralytics import YOLO


# =========================
# Configuration
# =========================

# You can switch the input video just by changing this filename.
# It is assumed to be located in PROJECT_ROOT / Input /
VIDEO_FILENAME = "100vs100_HMN1_MT15_TN23_CR15-12.MP4"
#VIDEO_FILENAME = "12vs5_HMN1_MT15_TN5_CR5-1.MP4"

# Model filenames (located in PROJECT_ROOT / Models /)
MODEL_ONE_NAME = "Model_One_fp16.pt"              # Main detection model (e.g., players, weapons, etc.)
MODEL_SCOREBOARD = "Model_Scoreboard_fp16.pt"     # Used ONLY on the first frame to locate the scoreboard
MODEL_POINT_DETECTOR = "Model_PointDetector21_fp16.pt"  # Applied on the cropped scoreboard area

# Output video filename (saved in PROJECT_ROOT / Output /)
OUTPUT_VIDEO_NAME = "annotated_combined.mp4"

# Number of frames to process at once. Higher = faster, but uses more GPU/CPU RAM.
BATCH_SIZE = 32


# =========================
# Path & environment helpers
# =========================

def get_project_paths() -> Tuple[Path, Path, Path, Path, Path, Path]:
    """
    Compute all important paths based on the location of this script.

    Returns:
        base_dir:             project root (one level above Codes/, if this script is in Codes/)
        input_video:          full path to the input video
        model_one_path:       full path to Model_One_fp16.pt
        model_pointdetector_path: full path to Model_PointDetector21_fp16.pt
        model_scoreboard_path: full path to Model_Scoreboard_fp16.pt
        output_dir:           folder where the output video will be saved
    """
    # __file__ = path of this .py file (e.g. PROJECT_ROOT/Codes/script.py)
    # parents[1] = go up two levels: script.py -> Codes -> PROJECT_ROOT
    base_dir = Path(__file__).resolve().parents[1]

    # Standard project structure:
    # PROJECT_ROOT / Input
    # PROJECT_ROOT / Models
    # PROJECT_ROOT / Output
    input_dir = base_dir / "Input"
    models_dir = base_dir / "Models"
    output_dir = base_dir / "Output"

    # Build full paths to video and models
    input_video = input_dir / VIDEO_FILENAME
    model_one_path = models_dir / MODEL_ONE_NAME
    model_pointdetector_path = models_dir / MODEL_POINT_DETECTOR
    model_scoreboard_path = models_dir / MODEL_SCOREBOARD

    return base_dir, input_video, model_one_path, model_pointdetector_path, model_scoreboard_path, output_dir


def check_file_exists(path: Path, description: str) -> None:
    """
    Small helper to fail early with a clear error if a file is missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


# =========================
# Device & model loading
# =========================

def get_device() -> str:
    """
    Choose the best available device for PyTorch / YOLO inference.

    Order of preference:
        1. Apple MPS (for newer MacBooks)
        2. NVIDIA CUDA GPU
        3. CPU (fallback, slower)
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

    Note: the 'device' argument is not used directly here;
    the device is specified at inference time when calling the model.
    """
    print(f"Loading model from: {model_path}")
    model = YOLO(str(model_path))
    return model


# =========================
# Video I/O helpers
# =========================

def open_video_capture(video_path: Path) -> cv2.VideoCapture:
    """
    Open a video file for reading with OpenCV and ensure it is valid.
    """
    print(f"\nOpening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    return cap


def get_video_properties(cap: cv2.VideoCapture) -> Tuple[float, int, int, int]:
    """
    Read basic properties from a cv2.VideoCapture object:
        - FPS
        - Width
        - Height
        - Total number of frames
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        # In some broken videos, FPS might be 0 or invalid; we use a fallback.
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties -> FPS: {fps}, Size: {width}x{height}, Frames: {total_frames}")
    return fps, width, height, total_frames


def create_video_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """
    Create a cv2.VideoWriter to save the annotated video.
    """
    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 'mp4v' codec is widely supported and works well with .mp4
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer at: {output_path}")

    print(f"\nWriting output video to: {output_path}")
    return writer


def read_frame_batch(cap: cv2.VideoCapture, batch_size: int) -> List:
    """
    Read up to 'batch_size' frames from the video.

    Returns:
        frames: list of BGR numpy arrays (OpenCV image format).
    """
    frames = []
    for _ in range(batch_size):
        ret, frame = cap.read()
        if not ret:
            # No more frames in the video
            break
        frames.append(frame)
    return frames


# =========================
# Drawing & annotation
# =========================

def draw_detections_on_frame(
    frame,
    result,
    color: Tuple[int, int, int],
    label_prefix: str = "",
):
    """
    Draw bounding boxes and labels from a single YOLO 'Result' on the given frame.

    Args:
        frame:       BGR image (numpy array) modified in-place.
        result:      YOLO Result object containing boxes, classes, confidences.
        color:       (B, G, R) tuple for box & text color.
        label_prefix: string prefix to prepend to the class name (e.g. "M1:", "P:").
    """
    boxes = result.boxes
    if boxes is None or boxes.xyxy is None:
        return

    # Extract bbox coordinates, class indices, and confidences to CPU numpy arrays.
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    names = result.names

    for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
        # Convert coordinates to integers for OpenCV
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Build label: "PREFIXclass_name 0.87"
        label = f"{label_prefix}{names[c]} {p:.2f}"

        # Draw rectangle and text
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 5, 10)),  # small offset above the bbox, not going off-screen
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_detections_on_frame_with_offset(
    frame,
    result,
    color: Tuple[int, int, int],
    label_prefix: str = "",
    offset: Tuple[int, int] = (0, 0),
    min_conf: float = 0.0,
):
    """
    Similar to draw_detections_on_frame, but:

    - Adds an (x, y) offset to the bounding box coordinates.
      This is useful when detections are computed on a cropped region,
      and we want to map them back onto the full frame.

    - Only draws detections with confidence >= min_conf.
    """
    boxes = result.boxes
    if boxes is None or boxes.xyxy is None:
        return

    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    names = result.names
    off_x, off_y = offset  # This offset will be added to each box coordinate

    for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
        # Optional extra filter on confidence (safety net).
        if p < min_conf:
            continue

        # Map from crop coordinates back to the full frame by adding the offset.
        x1, y1, x2, y2 = int(x1 + off_x), int(y1 + off_y), int(x2 + off_x), int(y2 + off_y)

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


# =========================
# Scoreboard detection (first frame)
# =========================

def detect_scoreboard_box(model_scoreboard: YOLO, frame, device: str):
    """
    Use 'model_scoreboard' ONLY on the first frame to find the scoreboard bbox.

    Args:
        model_scoreboard: YOLO model trained to detect the scoreboard.
        frame:            First frame of the video (BGR image).
        device:           Device string ("cpu", "cuda", "mps").

    Returns:
        (x1, y1, x2, y2, label_name, conf) if a detection is found,
        otherwise None.
    """
    print("\nDetecting scoreboard on first frame...")

    # YOLO can take a single image; we pass 'frame' directly
    results = model_scoreboard(frame, device=device, verbose=False)
    result = results[0]

    boxes = result.boxes
    if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
        print("⚠️ No scoreboard bbox found on the first frame.")
        return None

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    names = result.names

    # Choose the bbox with the highest confidence as the scoreboard
    best_idx = conf.argmax()
    x1, y1, x2, y2 = xyxy[best_idx]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    best_conf = float(conf[best_idx])
    label_name = names[cls[best_idx]]

    print(f"Scoreboard detected on first frame: ({x1}, {y1}, {x2}, {y2}) - {label_name} {best_conf:.2f}")
    return x1, y1, x2, y2, label_name, best_conf


# =========================
# Core processing logic
# =========================

def process_video_with_models(
    video_path: Path,
    model_one: YOLO,
    model_scoreboard: YOLO,
    model_pointdetector: YOLO,
    output_video_path: Path,
    device: str,
    batch_size: int,
) -> Tuple[int, float]:
    """
    Main processing pipeline for the video.

    Steps:
        1. Open the video and read FPS, width, height, total frames.
        2. Read the first frame and use 'model_scoreboard' to detect the scoreboard.
           - The scoreboard bbox is assumed to be fixed across the video.
        3. For the entire video (in batches of frames):
           - Run 'model_one' on each full frame and draw detections.
           - If the scoreboard bbox is known:
               a) Draw the same scoreboard bbox + label + confidence on each frame.
               b) Crop the scoreboard area from each frame.
               c) Run 'model_pointdetector' on these crops (batched).
               d) Draw the point-detector detections back onto the full frames,
                  translating coordinates by the scoreboard bbox offset.
           - Write annotated frames to the output video.

    Returns:
        processed_frames: number of frames that were processed.
        elapsed_time:     total processing time in seconds.
    """
    # Open input video and create output writer
    cap = open_video_capture(video_path)
    fps, width, height, total_frames = get_video_properties(cap)
    writer = create_video_writer(output_video_path, fps, width, height)

    # ---------- STEP 1: detect scoreboard on the FIRST frame ----------
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read the first frame of the video.")

    scoreboard_box = detect_scoreboard_box(model_scoreboard, first_frame, device)

    # Decide whether we can use the point detector or not
    if scoreboard_box is None:
        print("Proceeding without point detector (no scoreboard bbox found).")
        # If we didn't find the scoreboard, we revert the video to the beginning
        # and process only with 'model_one'.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        use_point_detector = False

        # Dummy values (won't be used)
        x1_clip = y1_clip = x2_clip = y2_clip = 0
        sb_label = ""
        sb_conf = 0.0
    else:
        # Reset the video to the beginning so we don't lose the first frame.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Unpack scoreboard box info
        x1_sb, y1_sb, x2_sb, y2_sb, sb_label, sb_conf = scoreboard_box

        # To be safe, we clip the scoreboard bbox once using the known width/height.
        # This avoids recomputing these clamps for each frame.
        x1_clip = max(0, min(x1_sb, width - 1))
        x2_clip = max(0, min(x2_sb, width))
        y1_clip = max(0, min(y1_sb, height - 1))
        y2_clip = max(0, min(y2_sb, height))

        # If the clipped box is valid (non-zero size), we can use the point detector.
        use_point_detector = (x2_clip > x1_clip) and (y2_clip > y1_clip)

        if not use_point_detector:
            print("⚠️ Scoreboard bbox is degenerate after clipping; disabling point detector.")

    processed_frames = 0
    start_time = time.time()

    try:
        # 'inference_mode' disables gradient tracking and some overhead, speeding up inference.
        with torch.inference_mode():
            while True:
                # ---------- STEP 2: read a batch of frames ----------
                frames = read_frame_batch(cap, batch_size)
                if not frames:
                    # No more frames to process
                    break

                # ---------- STEP 3: run model_one on full frames ----------
                # YOLO accepts a list of numpy arrays as a batch.
                results_one = model_one(
                    frames,
                    device=device,
                    verbose=False,
                    imgsz=512,   # you can tune this for speed/accuracy tradeoff
                )

                if use_point_detector:
                    # When we have a scoreboard bbox:
                    #   - draw 'model_one' detections
                    #   - draw scoreboard bbox + label
                    for frame, r1 in zip(frames, results_one):
                        # Draw detections from 'model_one' in GREEN
                        draw_detections_on_frame(frame, r1, color=(0, 255, 0), label_prefix="M1:")

                        # Draw the FIXED scoreboard bbox in YELLOW on every frame
                        cv2.rectangle(frame, (x1_clip, y1_clip), (x2_clip, y2_clip), (255, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"{sb_label} {sb_conf:.2f}",
                            (x1_clip, max(y1_clip - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                else:
                    # If no scoreboard box was found, we only draw 'model_one' detections
                    for frame, r1 in zip(frames, results_one):
                        draw_detections_on_frame(frame, r1, color=(0, 255, 0), label_prefix="M1:")

                # ---------- STEP 4: run point detector on scoreboard crops ----------
                if use_point_detector:
                    scoreboard_crops = []
                    valid_frames = []  # frames that correspond to each crop

                    # Build the list of crops for this batch.
                    # We reuse x1_clip, y1_clip, x2_clip, y2_clip which were pre-clipped.
                    for frame in frames:
                        crop = frame[y1_clip:y2_clip, x1_clip:x2_clip]

                        if crop.size == 0:
                            # Safety check in case something is off
                            continue

                        scoreboard_crops.append(crop)
                        valid_frames.append(frame)

                    if scoreboard_crops:
                        # Run the point detector on all crops in a single batch.
                        # We set conf=0.7 so YOLO will discard low-confidence detections
                        # on the GPU/CPU side, avoiding extra filtering in Python.
                        results_points = model_pointdetector(
                            scoreboard_crops,
                            device=device,
                            verbose=False,
                            conf=0.7,   # filter detections < 0.5 directly in YOLO
                        )

                        # Map the detections from each crop back to the full frames
                        # using the scoreboard bbox as an offset.
                        for frame, crop_result in zip(valid_frames, results_points):
                            draw_detections_on_frame_with_offset(
                                frame,
                                crop_result,
                                color=(255, 0, 0),   # RED for point detector
                                label_prefix="P:",   # prefix to identify these detections
                                offset=(x1_clip, y1_clip),
                                min_conf=0.0,       # already filtered by conf=0.5 above
                            )

                # ---------- STEP 5: write annotated frames to output video ----------
                for f in frames:
                    writer.write(f)

                processed_frames += len(frames)

                # Log progress (one line updated in-place)
                print(f"Processed {processed_frames}/{total_frames} frames...", end="\r")

    finally:
        # Always release resources, even if an exception occurs.
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
        # Get all relevant paths
        (
            base_dir,
            input_video,
            model_one_path,
            model_pointdetector_path,
            model_scoreboard_path,
            output_dir,
        ) = get_project_paths()

        # Sanity checks to avoid confusing runtime errors
        check_file_exists(input_video, "Input video")
        check_file_exists(model_one_path, "Model_One")
        check_file_exists(model_scoreboard_path, "Model_Scoreboard_fp16")
        check_file_exists(model_pointdetector_path, "Model_PointDetector21_fp16")

        # Choose device (CPU / CUDA / MPS)
        device = get_device()

        # Load all YOLO models
        model_one = load_yolo_model(model_one_path, device)
        model_scoreboard = load_yolo_model(model_scoreboard_path, device)
        model_pointdetector = load_yolo_model(model_pointdetector_path, device)

        # Full path for the output video file
        output_video_path = output_dir / OUTPUT_VIDEO_NAME

        # Run the full processing pipeline and measure total time
        processed_frames, elapsed_time = process_video_with_models(
            video_path=input_video,
            model_one=model_one,
            model_scoreboard=model_scoreboard,
            model_pointdetector=model_pointdetector,
            output_video_path=output_video_path,
            device=device,
            batch_size=BATCH_SIZE,
        )

        # Compute effective FPS of the entire processing (including I/O)
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
        # For debugging, you can uncomment the next line to see the full traceback:
        # raise


if __name__ == "__main__":
    main()