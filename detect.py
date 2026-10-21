"""Detect objects in every image of a folder, optionally drawing conformal outputs.

Example:
    python detect.py --model best.pt --input images/ --output results/ \
        --calibrator runs/conformal/visdrone_q90.json
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from conformal import (
    calibrated_probabilities,
    correctness_sets,
    detector_arrays,
    load_calibrator,
    localization_regions,
    set_name,
)

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
COLORS = {  # BGR
    "correct": (0, 200, 0),
    "incorrect": (0, 0, 255),
    "uncertain": (0, 165, 255),
    "empty": (255, 255, 255),
}


def clamp_point(x, y, width, height):
    return max(0, min(width - 1, int(round(x)))), max(0, min(height - 1, int(round(y))))


def annotate_result(result, calibration=None, display_conf=0.25):
    """Return the annotated image.

    Without a calibrator this is the normal Ultralytics plot. With one, every
    box above display_conf is drawn in the colour of its conformal label set,
    inside its calibrated localization region.
    """
    if calibration is None:
        return result.plot()

    image = result.orig_img.copy()
    height, width = image.shape[:2]
    boxes, classes, conf = detector_arrays(result)
    if len(boxes) == 0:
        return image

    probabilities = calibrated_probabilities(calibration, boxes, conf)
    in_correct, in_incorrect = correctness_sets(calibration, probabilities)
    regions = localization_regions(calibration, boxes)

    for i in np.argsort(conf):  # draw low confidence first so strong boxes stay on top
        if conf[i] < display_conf:
            continue
        status = set_name(in_correct[i], in_incorrect[i])
        color = COLORS[status]
        x1, y1 = clamp_point(boxes[i, 0], boxes[i, 1], width, height)
        x2, y2 = clamp_point(boxes[i, 2], boxes[i, 3], width, height)
        rx1, ry1 = clamp_point(regions[i, 0], regions[i, 1], width, height)
        rx2, ry2 = clamp_point(regions[i, 2], regions[i, 3], width, height)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)      # detector box
        cv2.rectangle(image, (rx1, ry1), (rx2, ry2), color, 1)  # calibrated region
        name = result.names.get(int(classes[i]), str(classes[i])) if isinstance(result.names, dict) else str(classes[i])
        text = f"{name} p={probabilities[i]:.2f} {status}"
        cv2.putText(image, text, (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return image


def predict_settings(calibration):
    """Use the exact candidate-generation settings that qhat was calibrated with."""
    if calibration:
        return {
            "imgsz": int(calibration["imgsz"]),
            "conf": float(calibration["predict_conf"]),
            "iou": float(calibration["predict_iou"]),
        }
    return {"imgsz": 640, "conf": 0.25, "iou": 0.45}


def detect_images_in_folder(model_path, input_folder, output_folder, calibration_path=None, display_conf=0.25):
    """Run detection on every image in input_folder and save annotated copies."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    calibration = load_calibrator(calibration_path) if calibration_path else None
    settings = predict_settings(calibration)

    for image_path in sorted(Path(input_folder).iterdir()):
        if image_path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        result = model.predict(source=str(image_path), verbose=False, **settings)[0]
        annotated = annotate_result(result, calibration, display_conf)
        output_path = output_folder / f"detected_{image_path.name}"
        if not cv2.imwrite(str(output_path), annotated):
            raise IOError(f"Could not write output image: {output_path}")
        print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Trained .pt model")
    parser.add_argument("--input", required=True, help="Folder with images")
    parser.add_argument("--output", required=True, help="Folder for annotated images")
    parser.add_argument("--calibrator", help="JSON from calibrate_conformal.py (enables CP drawing)")
    parser.add_argument("--display-conf", type=float, default=0.25, help="Only draw boxes above this confidence")
    args = parser.parse_args()
    detect_images_in_folder(args.model, args.input, args.output, args.calibrator, args.display_conf)


if __name__ == "__main__":
    main()
