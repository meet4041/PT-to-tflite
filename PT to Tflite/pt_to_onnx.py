import os
from ultralytics import YOLO

PT_MODEL_PATH = "ball_detector_v3.pt"
IMG_SIZE = 224

def convert_model():
    if not os.path.exists(PT_MODEL_PATH):
        print(f"Error: {PT_MODEL_PATH} not found.")
        return

    print(f"Loading PyTorch model: {PT_MODEL_PATH}")
    model = YOLO(PT_MODEL_PATH)

    print(f"Exporting to ONNX (imgsz={IMG_SIZE})...")
    exported_path = model.export(
        format="onnx",
        imgsz=IMG_SIZE,
        simplify=True,
    )

    if exported_path:
        print(f"\nSUCCESS! ONNX model saved at: {exported_path}")
    else:
        print("\nExport failed. Check that the required export dependencies are installed.")


if __name__ == "__main__":
    convert_model()
