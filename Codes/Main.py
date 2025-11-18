import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import csv  # for CSV writing

import cv2
import torch
from ultralytics import YOLO


# =========================
# Configuration
# =========================

VIDEO_FILENAME = "100vs100_HMN1_MT15_TN23_CR15-12.MP4"
# VIDEO_FILENAME = "12vs5_HMN1_MT15_TN5_CR5-1.MP4"

MODEL_ONE_NAME = "Model_One_fp16.pt"
MODEL_SCOREBOARD = "Model_Scoreboard_fp16.pt"
MODEL_POINT_DETECTOR = "Model_PointDetector21_fp16.pt"
MODEL_TWO_NAME = "Model_Two_fp16.pt"   # <--- NEW: pose / skeleton model

OUTPUT_VIDEO_NAME = "annotated_combined.mp4"

# ByteTrack tracker config (must be reachable from your working dir)
TRACKER_CONFIG = "bytetrack.yaml"

# Batch size for batched frame processing
BATCH_SIZE = 32

# Custom skeleton edges (using your indices)
skeleton_edges = [
    # top triangle
    (0, 1),  # mask-as
    (0, 7),  # mask-nas
    (1, 7),  # as-nas

    # central body square
    (1, 4),   # as-aa
    (4, 10),  # aa-naa
    (10, 7),  # naa-nas

    # armed arm
    (1, 2),  # as-ae
    (2, 3),  # ae-ah

    # not armed arm
    (7, 8),
    (8, 9),

    # armed leg
    (4, 5),
    (5, 6),

    # not armed leg
    (10, 11),
    (11, 12),
]


# =========================
# Path & environment helpers
# =========================

def get_project_paths() -> Tuple[Path, Path, Path, Path, Path, Path, Path]:
    """
    Returns:
        base_dir:             project root (one level above Codes/)
        input_video:          full path to input video
        model_one_path:       full path to Model_One_fp16.pt
        model_pointdetector_path: full path to Model_PointDetector21_fp16.pt
        model_scoreboard_path: full path to Model_Scoreboard_fp16.pt
        model_two_path:       full path to Model_Two_fp16.pt
        output_dir:           Output folder path
    """
    base_dir = Path(__file__).resolve().parents[1]
    input_dir = base_dir / "Input"
    models_dir = base_dir / "Models"
    output_dir = base_dir / "Output"

    input_video = input_dir / VIDEO_FILENAME
    model_one_path = models_dir / MODEL_ONE_NAME
    model_pointdetector_path = models_dir / MODEL_POINT_DETECTOR
    model_scoreboard_path = models_dir / MODEL_SCOREBOARD
    model_two_path = models_dir / MODEL_TWO_NAME

    return (
        base_dir,
        input_video,
        model_one_path,
        model_pointdetector_path,
        model_scoreboard_path,
        model_two_path,
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
    (device is used at inference time, not here.)
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
    id_color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
):
    """
    Draw bounding boxes and labels from a single YOLO Result on the given frame.
    If tracking IDs are present (from ByteTrack), they are appended to the label.
    Optionally color specific IDs using id_color_map (e.g. {1: (0,0,255), 2: (255,255,0)}).
    """
    boxes = result.boxes
    if boxes is None or boxes.xyxy is None:
        return

    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    names = result.names

    # IDs for tracking (may be None if no tracking)
    if hasattr(boxes, "id") and boxes.id is not None:
        ids = boxes.id.cpu().numpy().astype(int)
    else:
        ids = [None] * len(xyxy)

    for (x1, y1, x2, y2), c, p, tid in zip(xyxy, cls, conf, ids):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        base_label = f"{label_prefix}{names[c]} {p:.2f}"
        if tid is not None:
            label = f"{base_label} ID{tid}"
        else:
            label = base_label

        # Choose color: special per-ID color if provided, otherwise default
        if tid is not None and id_color_map is not None and tid in id_color_map:
            draw_color = id_color_map[tid]
        else:
            draw_color = color

        cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            draw_color,
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
    Like draw_detections_on_frame, but translates the boxes by an (x, y) offset.
    Useful when detections are computed on a crop and must be mapped back
    onto the full frame.

    Only detections with confidence >= min_conf are drawn.
    """
    boxes = result.boxes
    if boxes is None or boxes.xyxy is None:
        return

    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    names = result.names
    off_x, off_y = offset

    for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
        if p < min_conf:
            continue

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


def draw_skeleton_on_frame(
    frame,
    keypoints_xy,
    keypoints_conf=None,
    edges=None,
    color=(0, 255, 255),
    conf_th: float = 0.3,
):
    """
    Draws a custom skeleton on the frame given keypoints and edges.

    keypoints_xy:  (num_kpts, 2) array-like with (x, y) in image coords
    keypoints_conf: (num_kpts,) confidences or None
    edges: list of (i, j) pairs
    """
    if edges is None:
        return
    if keypoints_xy is None:
        return

    num_kpts = len(keypoints_xy)
    if num_kpts == 0:
        return

    for (i, j) in edges:
        if i >= num_kpts or j >= num_kpts:
            continue

        x_i, y_i = keypoints_xy[i]
        x_j, y_j = keypoints_xy[j]

        if keypoints_conf is not None:
            if i < len(keypoints_conf) and j < len(keypoints_conf):
                if keypoints_conf[i] < conf_th or keypoints_conf[j] < conf_th:
                    continue

        p1 = (int(x_i), int(y_i))
        p2 = (int(x_j), int(y_j))

        cv2.line(frame, p1, p2, color, 2)
        cv2.circle(frame, p1, 3, color, -1)
        cv2.circle(frame, p2, 3, color, -1)


# =========================
# Scoreboard detection (first frame)
# =========================

def detect_scoreboard_box(model_scoreboard: YOLO, frame, device: str):
    """
    Use model_scoreboard only on the first frame to find the scoreboard bbox.
    Returns:
        (x1, y1, x2, y2, label_name, conf) or None if nothing is found.
    """
    print("\nDetecting scoreboard on first frame...")
    # Relaxed confidence threshold 0.5 for robustness
    results = model_scoreboard(frame, device=device, verbose=False, conf=0.5)
    result = results[0]

    boxes = result.boxes
    if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
        print("⚠️ No scoreboard bbox found on the first frame.")
        return None

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    names = result.names

    # Take the bbox with the highest confidence
    best_idx = conf.argmax()
    x1, y1, x2, y2 = xyxy[best_idx]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    best_conf = float(conf[best_idx])
    label_name = names[cls[best_idx]]

    print(f"Scoreboard detected on first frame: ({x1}, {y1}, {x2}, {y2}) - {label_name} {best_conf:.2f}")
    return x1, y1, x2, y2, label_name, best_conf


# =========================
# Core processing logic (with ByteTrack for fencers, batched)
# =========================

def process_video_with_models(
    video_path: Path,
    model_one: YOLO,
    model_scoreboard: YOLO,
    model_pointdetector: YOLO,
    model_two: YOLO,
    output_video_path: Path,
    device: str,
    batch_size: int,
) -> Tuple[int, float]:
    """
    Batched processing + ByteTrack tracking for fencers (model_one).

    - model_one: uses .track(...) with tracker="bytetrack.yaml" and persist=True
                 on batched frames (list of np.ndarray) to track all fencers.
      * We specially track ID1 and ID2 for skeleton.
    - model_scoreboard: used on first frame only to find scoreboard bbox.
    - model_pointdetector: runs on cropped scoreboard area (batched crops).
    - model_two: pose/skeleton model, runs ONLY on expanded crops
                 of fencer ID1 and ID2.

    CSV: one row per frame with:
        Frame_Number, Time, TouchLeft, TouchRight,
        Box_FencerID1, Box_FencerID2,
        SkeletonID1, SkeletonID2

    Box_FencerID1 / Box_FencerID2:
      - bbox of track ID 1/2 in the format "x1,y1,x2,y2"
      - "NaN" if that ID is not detected in that frame.

    SkeletonID1 / SkeletonID2:
      - skeleton keypoints in the format "x0,y0;x1,y1;...;x12,y12"
      - "NaN" if skeleton not detected for that ID in that frame.

    Thresholds:
      - model_one (tracking, fencers): conf = 0.8
      - model_scoreboard: conf = 0.5
      - model_pointdetector: conf = 0.5 and p >= 0.5 in Python filter
      - model_two (pose): conf = 0.5 for detections, keypoints conf >= 0.3 for drawing
    """
    cap = open_video_capture(video_path)
    fps, width, height, total_frames = get_video_properties(cap)
    writer = create_video_writer(output_video_path, fps, width, height)

    # CSV path: same name as output video but with .csv
    csv_path = output_video_path.with_suffix(".csv")
    print(f"Writing touch CSV to: {csv_path}")

    # --- 1) Read first frame and detect scoreboard bbox ---
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read the first frame of the video.")

    scoreboard_box = detect_scoreboard_box(model_scoreboard, first_frame, device)

    if scoreboard_box is None:
        print("Proceeding without point detector (no scoreboard bbox found).")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        use_point_detector = False
        x1_clip = y1_clip = x2_clip = y2_clip = 0
        sb_label = ""
        sb_conf = 0.0
    else:
        # Reset the video to the beginning so we don't lose the first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        x1_sb, y1_sb, x2_sb, y2_sb, sb_label, sb_conf = scoreboard_box

        # Clip bbox to video dimensions once
        x1_clip = max(0, min(x1_sb, width - 1))
        x2_clip = max(0, min(x2_sb, width))
        y1_clip = max(0, min(y1_sb, height - 1))
        y2_clip = max(0, min(y2_sb, height))

        use_point_detector = (x2_clip > x1_clip) and (y2_clip > y1_clip)
        if not use_point_detector:
            print("⚠️ Scoreboard bbox is degenerate after clipping; disabling point detector.")

    processed_frames = 0
    frame_index = 0  # global frame counter used for CSV Frame_Number
    start_time = time.time()

    # Color map for tracked IDs: ID 1 and ID 2 special
    id_color_map = {
        1: (0, 0, 255),      # ID 1 -> red (BGR)
        2: (255, 255, 0),    # ID 2 -> cyan (BGR)
    }

    with open(csv_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            "Frame_Number",
            "Time",
            "TouchLeft",
            "TouchRight",
            "Box_FencerID1",
            "Box_FencerID2",
            "SkeletonID1",
            "SkeletonID2",
        ])

        try:
            with torch.inference_mode():
                while True:
                    frames = read_frame_batch(cap, batch_size)
                    if not frames:
                        break

                    n_batch = len(frames)

                    # Per-batch arrays
                    batch_touch_left = [0] * n_batch
                    batch_touch_right = [0] * n_batch
                    batch_box_fencerID1 = ["NaN"] * n_batch
                    batch_box_fencerID2 = ["NaN"] * n_batch
                    batch_box_coords_id1: List[Optional[Tuple[int, int, int, int]]] = [None] * n_batch
                    batch_box_coords_id2: List[Optional[Tuple[int, int, int, int]]] = [None] * n_batch

                    batch_skeleton_id1 = ["NaN"] * n_batch
                    batch_skeleton_id2 = ["NaN"] * n_batch

                    # --- 2) model_one + ByteTrack on full frames (batched) ---
                    results_one = model_one.track(
                        source=frames,
                        device=device,
                        verbose=False,
                        tracker=TRACKER_CONFIG,
                        persist=True,   # keep ByteTrack state across calls
                        conf=0.8,
                        imgsz=448,
                    )

                    # Draw fencers + optionally scoreboard, and at the same time
                    # extract boxes for track IDs 1 and 2.
                    if use_point_detector:
                        for idx, (frame, r1) in enumerate(zip(frames, results_one)):
                            # draw fencers (ID1/ID2 with special colors)
                            draw_detections_on_frame(
                                frame,
                                r1,
                                color=(0, 255, 0),  # default green
                                label_prefix="M1:",
                                id_color_map=id_color_map,
                            )

                            # draw scoreboard
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

                            # --- extract bbox for track IDs 1 and 2 ---
                            boxes = r1.boxes
                            if boxes is not None and boxes.xyxy is not None and boxes.id is not None:
                                xyxy = boxes.xyxy.cpu().numpy()
                                conf = boxes.conf.cpu().numpy()
                                tids = boxes.id.cpu().numpy().astype(int)

                                for target_id in (1, 2):
                                    idxs_id = [j for j, tid in enumerate(tids) if tid == target_id]
                                    if idxs_id:
                                        best_j = max(idxs_id, key=lambda j: conf[j])
                                        x1_, y1_, x2_, y2_ = xyxy[best_j]
                                        x1_i, y1_i, x2_i, y2_i = int(x1_), int(y1_), int(x2_), int(y2_)
                                        if target_id == 1:
                                            batch_box_fencerID1[idx] = f"{x1_i},{y1_i},{x2_i},{y2_i}"
                                            batch_box_coords_id1[idx] = (x1_i, y1_i, x2_i, y2_i)
                                        else:
                                            batch_box_fencerID2[idx] = f"{x1_i},{y1_i},{x2_i},{y2_i}"
                                            batch_box_coords_id2[idx] = (x1_i, y1_i, x2_i, y2_i)
                    else:
                        for idx, (frame, r1) in enumerate(zip(frames, results_one)):
                            draw_detections_on_frame(
                                frame,
                                r1,
                                color=(0, 255, 0),
                                label_prefix="M1:",
                                id_color_map=id_color_map,
                            )

                            # --- extract bbox for track IDs 1 and 2 ---
                            boxes = r1.boxes
                            if boxes is not None and boxes.xyxy is not None and boxes.id is not None:
                                xyxy = boxes.xyxy.cpu().numpy()
                                conf = boxes.conf.cpu().numpy()
                                tids = boxes.id.cpu().numpy().astype(int)

                                for target_id in (1, 2):
                                    idxs_id = [j for j, tid in enumerate(tids) if tid == target_id]
                                    if idxs_id:
                                        best_j = max(idxs_id, key=lambda j: conf[j])
                                        x1_, y1_, x2_, y2_ = xyxy[best_j]
                                        x1_i, y1_i, x2_i, y2_i = int(x1_), int(y1_), int(x2_), int(y2_)
                                        if target_id == 1:
                                            batch_box_fencerID1[idx] = f"{x1_i},{y1_i},{x2_i},{y2_i}"
                                            batch_box_coords_id1[idx] = (x1_i, y1_i, x2_i, y2_i)
                                        else:
                                            batch_box_fencerID2[idx] = f"{x1_i},{y1_i},{x2_i},{y2_i}"
                                            batch_box_coords_id2[idx] = (x1_i, y1_i, x2_i, y2_i)

                    # --- 3) Point detector on scoreboard crops (batched) ---
                    if use_point_detector:
                        scoreboard_crops = []
                        crop_indices = []  # indices of frames that produced a valid crop

                        for idx, frame in enumerate(frames):
                            crop = frame[y1_clip:y2_clip, x1_clip:x2_clip]
                            if crop.size == 0:
                                continue
                            scoreboard_crops.append(crop)
                            crop_indices.append(idx)

                        if scoreboard_crops:
                            results_points = model_pointdetector(
                                scoreboard_crops,
                                device=device,
                                verbose=False,
                                conf=0.5,   # more permissive
                                imgsz=256,  # scoreboard is already a crop
                            )

                            for idx_in_list, crop_result in zip(crop_indices, results_points):
                                frame = frames[idx_in_list]

                                # Draw detections from point detector on the full frame (red)
                                draw_detections_on_frame_with_offset(
                                    frame,
                                    crop_result,
                                    color=(255, 0, 0),
                                    label_prefix="P:",
                                    offset=(x1_clip, y1_clip),
                                    min_conf=0.5,  # enforce 0.5 threshold visually
                                )

                                # Touch logic using class index: 0 = TouchLeft, 1 = TouchRIght
                                boxes = crop_result.boxes
                                if boxes is not None and boxes.xyxy is not None and len(boxes.xyxy) > 0:
                                    cls = boxes.cls.cpu().numpy().astype(int)
                                    conf = boxes.conf.cpu().numpy()
                                    for c, p in zip(cls, conf):
                                        if p < 0.5:
                                            continue
                                        if c == 0:
                                            batch_touch_left[idx_in_list] = 1
                                        elif c == 1:
                                            batch_touch_right[idx_in_list] = 1

                    # --- 4) Model_Two (skeleton) on expanded crops for ID1 and ID2 ---
                    skeleton_crops = []
                    skeleton_meta = []  # list of (frame_idx_in_batch, fencer_id, x_offset, y_offset)

                    expand_factor = 0.15  # 15% expansion on each side in x

                    for idx in range(n_batch):
                        for fencer_id, box_coords in (
                            (1, batch_box_coords_id1[idx]),
                            (2, batch_box_coords_id2[idx]),
                        ):
                            if box_coords is None:
                                continue
                            x1_b, y1_b, x2_b, y2_b = box_coords
                            w_box = x2_b - x1_b
                            if w_box <= 0:
                                continue
                            expand = int(w_box * expand_factor)

                            x1_exp = max(0, x1_b - expand)
                            x2_exp = min(width - 1, x2_b + expand)
                            y1_exp = y1_b
                            y2_exp = y2_b

                            if x2_exp <= x1_exp or y2_exp <= y1_exp:
                                continue

                            crop = frames[idx][y1_exp:y2_exp, x1_exp:x2_exp]
                            if crop.size == 0:
                                continue

                            skeleton_crops.append(crop)
                            # offset to map keypoints back to full frame coords
                            skeleton_meta.append((idx, fencer_id, x1_exp, y1_exp))

                    if skeleton_crops:
                        results_two = model_two(
                            skeleton_crops,
                            device=device,
                            verbose=False,
                            conf=0.5,
                            imgsz=448,
                        )

                        for meta, res_pose in zip(skeleton_meta, results_two):
                            frame_idx_in_batch, fencer_id, x_off, y_off = meta
                            frame = frames[frame_idx_in_batch]

                            kps = getattr(res_pose, "keypoints", None)
                            if kps is None or kps.xy is None:
                                continue

                            kps_xy = kps.xy  # tensor [num_person, num_kpts, 2]
                            kps_conf = kps.conf  # tensor [num_person, num_kpts]

                            if kps_xy is None or kps_xy.numel() == 0:
                                continue

                            # Choose the first person (or main detection)
                            # You could also choose the one with highest mean keypoint conf.
                            if kps_conf is not None and kps_conf.numel() > 0:
                                # Select person index with highest mean conf
                                mean_conf = kps_conf.mean(dim=1)  # [num_person]
                                person_idx = int(torch.argmax(mean_conf).item())
                                k_xy = kps_xy[person_idx].cpu().numpy()
                                k_conf = kps_conf[person_idx].cpu().numpy()
                            else:
                                k_xy = kps_xy[0].cpu().numpy()
                                k_conf = None

                            # Map keypoints back to full-frame coordinates
                            full_xy = []
                            for (x_c, y_c) in k_xy:
                                full_xy.append((float(x_c) + x_off, float(y_c) + y_off))

                            # Draw skeleton (different color for ID1 vs ID2)
                            if fencer_id == 1:
                                skel_color = (0, 0, 255)  # red
                            else:
                                skel_color = (255, 255, 0)  # cyan

                            draw_skeleton_on_frame(
                                frame,
                                full_xy,
                                keypoints_conf=k_conf,
                                edges=skeleton_edges,
                                color=skel_color,
                                conf_th=0.3,
                            )

                            # Save skeleton coords in CSV string "x0,y0;...;x12,y12"
                            coords_str = ";".join(
                                f"{int(x)},{int(y)}" for (x, y) in full_xy
                            )

                            if fencer_id == 1:
                                batch_skeleton_id1[frame_idx_in_batch] = coords_str
                            else:
                                batch_skeleton_id2[frame_idx_in_batch] = coords_str

                    # --- 5) Write frames + CSV rows ---
                    for i, frame in enumerate(frames):
                        frame_number = frame_index + i
                        time_sec = frame_number / fps if fps > 0 else 0.0

                        writer.write(frame)
                        csv_writer.writerow([
                            frame_number,
                            time_sec,
                            batch_touch_left[i],
                            batch_touch_right[i],
                            batch_box_fencerID1[i],
                            batch_box_fencerID2[i],
                            batch_skeleton_id1[i],
                            batch_skeleton_id2[i],
                        ])

                    processed_frames += n_batch
                    frame_index += n_batch

                    print(f"Processed {processed_frames}/{total_frames} frames...", end="\r")

        finally:
            cap.release()
            writer.release()

    elapsed_time = time.time() - start_time

    print(f"\nDone! Combined annotated video saved at:\n{output_video_path}")
    print(f"CSV with touches + boxes + skeletons saved at:\n{csv_path}\n")

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
            model_pointdetector_path,
            model_scoreboard_path,
            model_two_path,
            output_dir,
        ) = get_project_paths()

        # Sanity checks
        check_file_exists(input_video, "Input video")
        check_file_exists(model_one_path, "Model_One")
        check_file_exists(model_scoreboard_path, "Model_Scoreboard_fp16")
        check_file_exists(model_pointdetector_path, "Model_PointDetector21_fp16")
        check_file_exists(model_two_path, "Model_Two_fp16")

        device = get_device()

        # Load models
        model_one = load_yolo_model(model_one_path, device)
        model_scoreboard = load_yolo_model(model_scoreboard_path, device)
        model_pointdetector = load_yolo_model(model_pointdetector_path, device)
        model_two = load_yolo_model(model_two_path, device)

        # Output video path
        output_video_path = output_dir / OUTPUT_VIDEO_NAME

        # Run processing and measure stats
        processed_frames, elapsed_time = process_video_with_models(
            video_path=input_video,
            model_one=model_one,
            model_scoreboard=model_scoreboard,
            model_pointdetector=model_pointdetector,
            model_two=model_two,
            output_video_path=output_video_path,
            device=device,
            batch_size=BATCH_SIZE,
        )

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
        # raise  # uncomment for full traceback during debugging


if __name__ == "__main__":
    main()