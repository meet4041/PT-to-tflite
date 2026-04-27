from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

EXPORT_PACKAGES = [
    "onnxslim>=0.1.71",
    "onnxruntime",
    "protobuf>=5",
    "tf_keras<=2.19.0",
    "sng4onnx>=1.0.1",
    "onnx_graphsurgeon>=0.3.26",
    "ai-edge-litert>=1.2.0,<1.4.0",
    "onnx2tf>=1.16.0",
]

REQUIRED_MODULES = (
    "onnxslim",
    "onnxruntime",
    "tf_keras",
    "sng4onnx",
    "onnx_graphsurgeon",
    "ai_edge_litert",
    "onnx2tf",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an Ultralytics YOLO .pt model to TensorFlow Lite."
    )
    parser.add_argument("weights", type=Path, help="Path to the .pt model")
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=[224],
        help="Inference image size. Use one value for square, or H W.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Export batch size",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help='Export device, e.g. "cpu" or "0"',
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export FP16 TFLite model",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Export INT8 quantized TFLite model",
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Dataset YAML for INT8 calibration",
    )
    parser.add_argument(
        "--nms",
        action="store_true",
        help="Include NMS in exported model if supported",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional destination directory for exported files",
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Do not auto-install missing export dependencies",
    )
    return parser.parse_args()


def normalize_imgsz(values: list[int]) -> int | tuple[int, int]:
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return values[0], values[1]
    raise SystemExit("--imgsz accepts either one value or two values")


def missing_modules() -> list[str]:
    return [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]


def install_export_dependencies() -> None:
    cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", *EXPORT_PACKAGES]
    extra_index = os.environ.get("PIP_EXTRA_INDEX_URL", "https://pypi.ngc.nvidia.com")
    if extra_index:
        cmd.extend(["--extra-index-url", extra_index])
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "Failed to install export dependencies.\n"
            f"Run:\n{sys.executable} -m pip install --no-cache-dir "
            + " ".join(f"'{pkg}'" for pkg in EXPORT_PACKAGES)
        ) from exc


def patch_ultralytics_requirement_check() -> None:
    import ultralytics.engine.exporter as exporter

    original = exporter.check_requirements

    def patched(requirements, *args, **kwargs):
        if isinstance(requirements, (list, tuple)):
            requirements = [
                "onnx2tf>=1.16.0" if str(req).startswith("onnx2tf>=") else req for req in requirements
            ]
        elif isinstance(requirements, str) and requirements.startswith("onnx2tf>="):
            requirements = "onnx2tf>=1.16.0"
        return original(requirements, *args, **kwargs)

    exporter.check_requirements = patched


def main() -> int:
    args = parse_args()

    if args.half and args.int8:
        raise SystemExit("Use either --half or --int8, not both")

    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        raise SystemExit(f"Model not found: {weights}")
    if weights.suffix.lower() != ".pt":
        raise SystemExit("Input model must be a .pt file")

    if args.int8 and not args.data:
        raise SystemExit("--data is required with --int8")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: ultralytics\n"
            f"Install with: {sys.executable} -m pip install ultralytics"
        ) from exc

    patch_ultralytics_requirement_check()

    missing = missing_modules()
    if missing:
        if args.no_install:
            raise SystemExit(
                "Missing export dependencies: "
                + ", ".join(missing)
                + "\nInstall with:\n"
                + f"{sys.executable} -m pip install --no-cache-dir "
                + " ".join(f"'{pkg}'" for pkg in EXPORT_PACKAGES)
            )
        install_export_dependencies()
        missing = missing_modules()
        if missing:
            raise SystemExit("Missing export dependencies after install: " + ", ".join(missing))

    imgsz = normalize_imgsz(args.imgsz)
    data = str(args.data.expanduser().resolve()) if args.data else None
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else None

    model = YOLO(str(weights))
    result = model.export(
        format="tflite",
        imgsz=imgsz,
        batch=args.batch,
        device=args.device,
        half=args.half,
        int8=args.int8,
        data=data,
        nms=args.nms,
    )

    exported = Path(result).resolve()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / exported.name
        exported.replace(target)
        exported = target.resolve()

    print(exported)
    return 0


if __name__ == "__main__":
    sys.exit(main())
