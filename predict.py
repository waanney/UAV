"""Run a single-image prediction, optionally with conformal outputs."""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from conformal import load_calibrator
from detect import annotate_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Trained .pt model")
    parser.add_argument("--source", required=True, help="Input image path")
    parser.add_argument("--output", default="output.png", help="Annotated output image")
    parser.add_argument("--calibrator", help="Optional JSON from calibrate_conformal.py")
    args = parser.parse_args()

    model = YOLO(args.model)
    calibration = load_calibrator(args.calibrator) if args.calibrator else None
    settings = (
        {
            "imgsz": int(calibration["imgsz"]),
            "conf": float(calibration["predict_conf"]),
            "iou": float(calibration["predict_iou"]),
        }
        if calibration
        else {"imgsz": 640, "conf": 0.25, "iou": 0.45}
    )
    result = model.predict(source=args.source, verbose=False, **settings)[0]
    annotated = annotate_result(result, calibration)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), annotated):
        raise IOError(f"Could not write output image: {output_path}")
    print(f"Saved prediction: {output_path}")


if __name__ == "__main__":
    main()
