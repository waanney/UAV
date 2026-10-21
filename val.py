"""Score a trained detector on a named Ultralytics dataset split."""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from train import parse_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Trained .pt weights")
    parser.add_argument("--data", required=True, help="Ultralytics dataset YAML")
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0,1")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--report", help="Optional JSON output path")
    args = parser.parse_args()

    model = YOLO(args.model)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=parse_device(args.device),
        conf=args.conf,
        iou=args.iou,
    )
    summary = {
        "split": args.split,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map75": float(metrics.box.map75),
        "map50_95": float(metrics.box.map),
        "per_class_map50_95": [float(x) for x in metrics.box.maps],
    }
    print(json.dumps(summary, indent=2))
    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved validation metrics: {report}")


if __name__ == "__main__":
    main()
