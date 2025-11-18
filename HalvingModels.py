from pathlib import Path
from ultralytics import YOLO
import torch


def print_model_dtype(tag: str, model: YOLO):
    """Print the dtype of the underlying PyTorch model."""
    try:
        dtype = next(model.model.parameters()).dtype
        print(f"  [{tag}] torch dtype: {dtype}")
    except Exception as e:
        print(f"  [{tag}] could not read dtype: {e}")


def convert_to_fp16(model_path: Path) -> None:
    print(f"\n=== Converting: {model_path.name} ===")

    if not model_path.exists():
        print(f"  -> SKIPPING, file not found: {model_path}")
        return

    # Load YOLO model (FP32 by default)
    model = YOLO(str(model_path))

    # Show original dtype
    print_model_dtype("before", model)

    # Convert underlying torch model to FP16
    model.model.half()

    # Show new dtype
    print_model_dtype("after", model)

    # Output path: same folder, name + "_fp16.pt"
    fp16_path = model_path.with_name(model_path.stem + "_fp16.pt")

    # Save using Ultralytics (writes a proper YOLO checkpoint)
    model.save(str(fp16_path))

    print(f"  -> Saved FP16 model to: {fp16_path}")


def main():
    # Project root = folder where this script lives
    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / "Models"

    print(f"Project root: {base_dir}")
    print(f"Models dir:   {models_dir}")

    # List the models you want to convert
    model_files = [
        models_dir / "Model_One.pt",
        models_dir / "Model_Two.pt",
        models_dir / "Model_Scoreboard.pt",
        models_dir / "Model_PointDetector21.pt",
    ]

    for mp in model_files:
        convert_to_fp16(mp)

    print("\nAll conversions done ✅")


if __name__ == "__main__":
    main()
