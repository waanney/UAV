"""Fit a split-conformal score calibrator from a YOLO validation split.

Example:
    python calibrate_conformal.py --weights runs/detect/train/weights/best.pt \
        --data ultralytics/data/VisDrone.yaml --output runs/conformal/visdrone.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ultralytics import YOLOWorld
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.mytools.conformal import SplitConformalDetector


class ConformalCalibrationValidator(DetectionValidator):
    """Detection validator that records post-NMS predictions and TP labels."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calibration_confidence: list[np.ndarray] = []
        self.calibration_classes: list[np.ndarray] = []
        self.calibration_tp: list[np.ndarray] = []

    def update_metrics(self, preds, batch):
        """Run regular validation bookkeeping and retain detections at IoU 0.5."""
        previous_count = len(self.stats["conf"])
        super().update_metrics(preds, batch)
        if len(self.stats["conf"]) == previous_count:
            return

        stat = self.stats
        self.calibration_confidence.append(stat["conf"][-1].detach().cpu().numpy())
        self.calibration_classes.append(stat["pred_cls"][-1].detach().cpu().numpy().astype(int))
        # iouv[0] is 0.50 in DetectionValidator.
        self.calibration_tp.append(stat["tp"][-1][:, 0].detach().cpu().numpy().astype(bool))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Trained YOLO/YOLOWorld checkpoint (.pt)")
    parser.add_argument("--data", required=True, help="Dataset YAML with a held-out calibration split")
    parser.add_argument("--split", default="val", help="Dataset split to use for calibration (default: val)")
    parser.add_argument("--output", required=True, help="Output JSON path for calibration scores")
    parser.add_argument("--alpha", type=float, default=0.1, help="Conformal significance level")
    parser.add_argument("--min-class-samples", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help='Device, e.g. "0", "0,1", or "cpu"')
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Low pre-NMS confidence floor; keep fixed for calibration and evaluation",
    )
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLOWorld(args.weights)

    # Build the full Ultralytics validator configuration, then override only
    # the values exposed by this script so evaluation is reproducible.
    validator_args = {
        **DEFAULT_CFG_DICT,
        **getattr(model.model, "args", {}),
        "model": args.weights,
        "data": args.data,
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "conf": args.conf,
        "iou": args.iou,
        "workers": args.workers,
        "plots": False,
        "save_json": False,
        "save_txt": False,
        "verbose": False,
        "mode": "val",
        "task": "detect",
    }
    validator = ConformalCalibrationValidator(args=validator_args)
    validator(model=model.model)

    if not validator.calibration_confidence:
        raise RuntimeError("No predictions were collected; check the checkpoint and calibration dataset")

    confidence = np.concatenate(validator.calibration_confidence)
    class_ids = np.concatenate(validator.calibration_classes)
    is_true_positive = np.concatenate(validator.calibration_tp)
    if not np.any(is_true_positive):
        raise RuntimeError("No matched true-positive detections found at IoU 0.50")

    calibrator = SplitConformalDetector(
        alpha=args.alpha,
        min_class_samples=args.min_class_samples,
    ).fit(confidence, is_true_positive, class_ids)
    output = Path(args.output)
    calibrator.save(output)

    print(f"Calibration detections: {len(confidence)}")
    print(f"Matched true positives: {int(is_true_positive.sum())}")
    print(f"Class-specific calibrators: {len(calibrator.class_scores)}")
    print(f"Global calibrator samples: {len(calibrator.global_scores)}")
    print(f"Saved conformal state: {output.resolve()}")
    print("Note: conformal p-values represent compatibility with TP calibration scores, not correctness probabilities.")


if __name__ == "__main__":
    main()

