"""Run train -> validation -> CP calibration -> rerank -> independent test mAP.

Calibration and test folders must be separate from the train/validation files
listed in --data. Prefer sequence-level splits for video-derived datasets.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

from conformal import IMAGE_EXTENSIONS, image_files
from train import parse_device, train_model


def run_script(script, *args):
    command = [sys.executable, str(Path(__file__).with_name(script)), *map(str, args)]
    subprocess.run(command, check=True, cwd=Path(__file__).parent)


def paths_from_split_entry(entry, root):
    """Resolve image files for an Ultralytics directory/list-file split."""
    entries = entry if isinstance(entry, list) else [entry]
    found = set()
    for value in entries:
        path = Path(value)
        path = path if path.is_absolute() else root / path
        if path.is_dir():
            found.update(p.resolve() for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        elif path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    listed = Path(line.strip())
                    found.add((listed if listed.is_absolute() else root / listed).resolve())
        else:
            raise FileNotFoundError(f"Could not resolve dataset split path: {path}")
    return found


def check_split_disjointness(data_yaml, calibration_images, test_images):
    """Fail fast if calibration/test image paths overlap train or validation."""
    data_yaml = Path(data_yaml).resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(config.get("path") or data_yaml.parent)
    if not dataset_root.is_absolute():
        dataset_root = data_yaml.parent / dataset_root
    dataset_root = dataset_root.resolve()
    train_val = set()
    for split in ("train", "val"):
        if split not in config:
            raise ValueError(f"Dataset YAML must define a '{split}' split")
        train_val.update(paths_from_split_entry(config[split], dataset_root))

    calibration = {p.resolve() for p in image_files(calibration_images)}
    test = {p.resolve() for p in image_files(test_images)}
    if calibration & test:
        raise ValueError("Calibration and test image paths overlap")
    if calibration & train_val:
        raise ValueError("Calibration images overlap the train/validation split in --data")
    if test & train_val:
        raise ValueError("Test images overlap the train/validation split in --data")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Ultralytics data YAML; val is used for model selection")
    parser.add_argument("--cal-images", required=True, help="Separate CP-calibration image directory")
    parser.add_argument("--cal-labels", required=True, help="YOLO labels for calibration images")
    parser.add_argument("--test-images", required=True, help="Independent final test image directory")
    parser.add_argument("--test-labels", required=True, help="YOLO labels for final test images")
    parser.add_argument("--output-dir", default="runs/conformal_pipeline")
    parser.add_argument("--weights", help="Existing trained .pt file; if omitted, this script trains a model")
    parser.add_argument("--model-config", help="YOLO-World model YAML; required when --weights is omitted")
    parser.add_argument("--pretrained", help="Initial .pt weights; required when --weights is omitted")
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--device", default="0,1")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--predict-conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--match-iou", type=float, default=0.5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    check_split_disjointness(args.data, args.cal_images, args.test_images)
    if args.weights:
        best_weights = Path(args.weights).resolve()
        if not best_weights.is_file():
            raise FileNotFoundError(f"Weights not found: {best_weights}")
    else:
        if not args.model_config or not args.pretrained:
            parser.error("Provide --weights or both --model-config and --pretrained")
        _, best_weights = train_model(
            model_config=args.model_config,
            pretrained=args.pretrained,
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            device=args.device,
            batch=args.batch,
            project=str(output_dir / "training"),
            name="yolo_world",
        )
        best_weights = best_weights.resolve()

    # Score the selected checkpoint on validation. This set is for model
    # selection/diagnostics and must not be reused as the CP calibration set.
    model = YOLO(str(best_weights))
    val_metrics = model.val(
        data=args.data,
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=parse_device(args.device),
        conf=args.predict_conf,
        iou=args.nms_iou,
    )
    validation_summary = {
        "weights": str(best_weights),
        "precision": float(val_metrics.box.mp),
        "recall": float(val_metrics.box.mr),
        "map50": float(val_metrics.box.map50),
        "map50_95": float(val_metrics.box.map),
        "map75": float(val_metrics.box.map75),
    }
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(validation_summary, indent=2), encoding="utf-8"
    )
    print("Validation metrics:", json.dumps(validation_summary, indent=2))

    calibrator_path = output_dir / "conformal_calibrator.json"
    run_script(
        "calibrate_conformal.py",
        "--model", best_weights,
        "--images", Path(args.cal_images).resolve(),
        "--labels", Path(args.cal_labels).resolve(),
        "--output", calibrator_path,
        "--alpha", args.alpha,
        "--conf", args.predict_conf,
        "--iou", args.nms_iou,
        "--match-iou", args.match_iou,
        "--imgsz", args.imgsz,
    )
    run_script(
        "evaluate_conformal.py",
        "--model", best_weights,
        "--calibrator", calibrator_path,
        "--images", Path(args.test_images).resolve(),
        "--labels", Path(args.test_labels).resolve(),
        "--report", output_dir / "test_metrics.json",
    )
    print(f"Pipeline artifacts: {output_dir}")
    print(f"Use this calibration file in detect.py: {calibrator_path}")


if __name__ == "__main__":
    main()
