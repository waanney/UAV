"""Validate YOLO/YOLOWorld with Conformal Uncertainty Calibration and Reranking.

This script runs validation on a dataset split (e.g. VisDrone val/test),
applies conformal calibration to update detection confidence scores and
reranks candidate boxes to evaluate the resulting mAP metrics.

Example:
    python val_conformal.py \
        --weights runs/detect/train/weights/best.pt \
        --data ultralytics/data/VisDrone.yaml \
        --conformal runs/conformal/visdrone.json \
        --split val \
        --method multiply \
        --alpha 0.1
"""

from __future__ import annotations

import argparse
from pathlib import Path
import torch

from ultralytics import YOLOWorld
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.mytools.conformal import SplitConformalDetector


class ConformalDetectionValidator(DetectionValidator):
    """Detection validator that applies Conformal Calibration and Reranking."""

    def __init__(
        self,
        calibrator: SplitConformalDetector,
        calibration_method: str = "multiply",
        alpha: float | None = None,
        gamma: float = 1.0,
        filter_rejected: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.calibrator = calibrator
        self.calibration_method = calibration_method
        self.alpha = alpha
        self.gamma = gamma
        self.filter_rejected = filter_rejected

    def update_metrics(self, preds, batch):
        """Calibrate confidence scores, rerank detections, and update metrics."""
        calibrated_preds = []
        for pred in preds:
            if pred is None or len(pred) == 0:
                calibrated_preds.append(pred)
                continue

            # pred shape: (N, 6) -> [x1, y1, x2, y2, conf, cls]
            conf = pred[:, 4].detach().cpu().numpy()
            cls_ids = pred[:, 5].detach().cpu().numpy().astype(int)

            calib_conf, p_vals = self.calibrator.calibrate_confidence(
                confidence=conf,
                class_ids=cls_ids,
                method=self.calibration_method,
                alpha=self.alpha,
                gamma=self.gamma,
            )

            pred_calib = pred.clone()
            pred_calib[:, 4] = torch.from_numpy(calib_conf).to(pred.device, dtype=pred.dtype)

            if self.filter_rejected:
                threshold = self.calibrator.alpha if self.alpha is None else self.alpha
                mask = torch.from_numpy(p_vals > threshold).to(pred.device)
                pred_calib = pred_calib[mask]

            if len(pred_calib) > 0:
                # Rerank: sort detections descending by calibrated confidence
                sort_idx = torch.argsort(pred_calib[:, 4], descending=True)
                pred_calib = pred_calib[sort_idx]

            calibrated_preds.append(pred_calib)

        super().update_metrics(calibrated_preds, batch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Trained YOLO/YOLOWorld checkpoint (.pt)")
    parser.add_argument("--data", required=True, help="Dataset YAML (e.g., ultralytics/data/VisDrone.yaml)")
    parser.add_argument("--conformal", required=True, help="Path to saved conformal JSON state from calibrate_conformal.py")
    parser.add_argument("--split", default="val", help="Dataset split to evaluate (val or test)")
    parser.add_argument(
        "--method",
        default="multiply",
        choices=["multiply", "harmonic", "p_value", "soft_threshold", "hard_threshold"],
        help="Calibration method: 'multiply', 'harmonic', 'p_value', 'soft_threshold', 'hard_threshold'",
    )
    parser.add_argument("--alpha", type=float, default=None, help="Significance level override (optional)")
    parser.add_argument("--gamma", type=float, default=1.0, help="Exponent for 'multiply' calibration method")
    parser.add_argument("--filter-rejected", action="store_true", help="Filter out detections with p_value <= alpha")
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help='Device, e.g. "0", "0,1", or "cpu"')
    parser.add_argument("--conf", type=float, default=0.001, help="Pre-NMS confidence floor")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conformal_path = Path(args.conformal)
    if not conformal_path.exists():
        raise FileNotFoundError(f"Conformal calibration file not found: {conformal_path}")

    print(f"Loading conformal calibrator from: {conformal_path}")
    calibrator = SplitConformalDetector.load(conformal_path)

    print(f"Loading model checkpoint: {args.weights}")
    model = YOLOWorld(args.weights)

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
        "verbose": True,
        "mode": "val",
        "task": "detect",
    }

    print("\n--- Running Validation with Conformal Calibration & Rerank ---")
    print(f"Method: {args.method} | Alpha: {args.alpha or calibrator.alpha} | Gamma: {args.gamma} | Filter Rejected: {args.filter_rejected}")

    validator = ConformalDetectionValidator(
        calibrator=calibrator,
        calibration_method=args.method,
        alpha=args.alpha,
        gamma=args.gamma,
        filter_rejected=args.filter_rejected,
        args=validator_args,
    )
    validator(model=model.model)

    metrics = validator.metrics
    print("\n================ Validation Results (Conformal Calibrated) ================")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"mAP@0.75     : {metrics.box.map75:.4f}")
    if hasattr(metrics.box, "maps") and metrics.box.maps is not None:
        print(f"APs per category: {metrics.box.maps}")
    print("==========================================================================")


if __name__ == "__main__":
    main()
