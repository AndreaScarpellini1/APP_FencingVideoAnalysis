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

VIDEO_FILENAME = "100vs100_HMN1_MT15_TN23_CR15-12.MP4"
MODEL_ONE_NAME = "Model_One_fp16.pt"
MODEL_THREE_NAME = "Model_Three_fp16.pt"
OUTPUT_VIDEO_NAME = "annotated_combined.mp4"
BATCH_SIZE = 32  # you can tweak this if needed


# =========================
# Path & environment helpers
# =========================

def get_project_paths() -> Tuple[Path, Path, Path, Path, Path]:
    """
    Returns:
        base_dir: project root (one level above Codes/)
        input_video: full path to input video
        model_one_path: full path to Model_One.pt
        model_three_path: full path to Model_Three.pt
        output_dir: Output folder path
    """
    base_dir = Path(__file__).resolve().parents[1]
    input_dir = base_dir / "Input"
    models_dir = base_dir / "Models"
    output_dir = base_dir / "Output"

    input_video = input_dir / VIDEO_FILENAME
    model_one_path = models_dir / MODEL_ONE_NAME
    model_three_path = models_dir / MODEL_THREE_NAME

    return base_dir, input_video, model_one_path, model_three_path, output_dir


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


def read_frame_batch(cap: cv2.VideoCapture, batch_size: int) -> List:
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
    frame,
    result,
    color: Tuple[int, int, int],
    label_prefix: str = "",
):
    """
    Draw bounding boxes and labels from a single YOLO Result on the given frame.
    """
    boxes = result.boxes
    if boxes is None or boxes.xyxy is None:
        return

    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    names = result.names

    for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
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


def annotate_batch_with_models(
    frames: List,
    results_one,
    results_three,
):
    """
    Draw detections from two models on each frame in-place.
    - Model_One: green, prefix 'M1:'
    - Model_Three: red, prefix 'M3:'
    """
    for frame, r1, r3 in zip(frames, results_one, results_three):
        draw_detections_on_frame(frame, r1, color=(0, 255, 0), label_prefix="M1:")
        draw_detections_on_frame(frame, r3, color=(0, 0, 255), label_prefix="M3:")


# =========================
# Core processing logic
# =========================

def process_video_with_two_models(
    video_path: Path,
    model_one: YOLO,
    model_three: YOLO,
    output_video_path: Path,
    device: str,
    batch_size: int,
) -> Tuple[int, float]:
    """
    Main loop:
    - open video
    - read frames in batches
    - run both models on each batch
    - draw detections from both models
    - write combined annotated frames to a single output video

    Returns:
        processed_frames: total number of frames processed
        elapsed_time: total execution time in seconds
    """
    cap = open_video_capture(video_path)
    fps, width, height, total_frames = get_video_properties(cap)
    writer = create_video_writer(output_video_path, fps, width, height)

    processed_frames = 0
    start_time = time.time()

    try:
        while True:
            frames = read_frame_batch(cap, batch_size)
            if not frames:
                break

            # YOLO accepts list of np.ndarray as batched input
            results_one = model_one(frames, device=device, verbose=False)
            results_three = model_three(frames, device=device, verbose=False)

            # Annotate frames in-place
            annotate_batch_with_models(frames, results_one, results_three)

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
            model_three_path,
            output_dir,
        ) = get_project_paths()

        # Sanity checks
        check_file_exists(input_video, "Input video")
        check_file_exists(model_one_path, "Model_One")
        check_file_exists(model_three_path, "Model_Three")

        device = get_device()

        # Load models
        model_one = load_yolo_model(model_one_path, device)
        model_three = load_yolo_model(model_three_path, device)

        # Output video path
        output_video_path = output_dir / OUTPUT_VIDEO_NAME

        # --- Run processing and measure stats ---
        processed_frames, elapsed_time = process_video_with_two_models(
            video_path=input_video,
            model_one=model_one,
            model_three=model_three,
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
        # Uncomment the next line if you want full traceback during debugging
        # raise


if __name__ == "__main__":
    main()
