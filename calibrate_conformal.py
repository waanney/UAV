"""Fit the size-conditional conformal calibrator for a YOLO / YOLO-World model.

Example:
    python calibrate_conformal.py --model path/to/best.pt \
        --images path/to/calibration/images \
        --labels path/to/calibration/labels \
        --output runs/conformal/visdrone_q90.json --alpha 0.1

Use images that were NOT used to train, early-stop or tune the model, and that
are also not your final test images. The images are split in two halves:
  * fit half  -> size buckets + isotonic probability model
  * calib half -> conformal quantiles (never seen by the fitting step)
"""

import argparse
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from conformal import (
    collect_record,
    fit_calibrator,
    image_files,
    save_calibrator,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Trained YOLO/YOLO-World .pt weights")
    parser.add_argument("--images", required=True, help="Held-out calibration image directory")
    parser.add_argument("--labels", required=True, help="YOLO-format labels directory")
    parser.add_argument("--output", default="runs/conformal/calibrator.json")
    parser.add_argument("--alpha", type=float, default=0.1, help="Miscoverage rate; 0.1 targets 90%%")
    parser.add_argument("--conf", type=float, default=0.001, help="Candidate-generation confidence cutoff")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold")
    parser.add_argument("--match-iou", type=float, default=0.5, help="GT matching IoU threshold")
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--n-buckets", type=int, default=3, help="Object-size buckets (1 = no size conditioning)")
    parser.add_argument("--min-bucket-samples", type=int, default=200,
                        help="Minimum fit detections before a bucket gets its own probability model")
    parser.add_argument("--fit-fraction", type=float, default=0.5, help="Share of images used for the fit half")
    parser.add_argument("--score-unit", choices=("object", "image"), default="object",
                        help="object: per-object coverage; image: joint per image (more conservative)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the fit/calibration image split")
    return parser.parse_args()


def check_args(args):
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must be between 0 and 1")
    if not 0 <= args.conf < 1 or not 0 <= args.iou <= 1 or not 0 <= args.match_iou <= 1:
        raise ValueError("--conf, --iou, and --match-iou must be within [0, 1]")
    if not 0 < args.fit_fraction < 1:
        raise ValueError("--fit-fraction must be between 0 and 1")
    if args.n_buckets < 1:
        raise ValueError("--n-buckets must be at least 1")


def split_images(images, fit_fraction, seed):
    """Deterministically split the image list into (fit_images, calib_images)."""
    order = np.random.default_rng(seed).permutation(len(images))
    n_fit = int(round(len(images) * fit_fraction))
    if n_fit < 1 or n_fit >= len(images):
        raise ValueError("Need at least 2 images so both halves are non-empty")
    fit = [images[i] for i in sorted(order[:n_fit])]
    calib = [images[i] for i in sorted(order[n_fit:])]
    return fit, calib


def main():
    args = parse_args()
    check_args(args)

    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    model = YOLO(str(model_path))

    fit_images, calib_images = split_images(image_files(args.images), args.fit_fraction, args.seed)

    def records_for(image_list, name):
        records = []
        for number, image_path in enumerate(image_list, start=1):
            records.append(collect_record(
                model, image_path, args.labels,
                conf=args.conf, iou=args.iou, imgsz=args.imgsz, match_iou=args.match_iou,
            ))
            if number % 50 == 0:
                print(f"  {name}: {number}/{len(image_list)} images")
        return records

    print(f"Collecting detections: {len(fit_images)} fit images, {len(calib_images)} calibration images")
    fit_records = records_for(fit_images, "fit")
    calib_records = records_for(calib_images, "calib")

    calibrator = fit_calibrator(
        fit_records, calib_records,
        alpha=args.alpha,
        n_buckets=args.n_buckets,
        min_bucket_samples=args.min_bucket_samples,
        score_unit=args.score_unit,
    )
    payload = save_calibrator(
        args.output, calibrator,
        model=model_path, conf=args.conf, iou=args.iou, imgsz=args.imgsz, match_iou=args.match_iou,
    )

    print(f"\nSaved conformal calibrator: {args.output}")
    print(f"Target coverage: {payload['target_coverage']:.1%} (score unit: {payload['score_unit']})")
    print(f"Size bucket edges (sqrt(w*h) px): {[round(e, 1) for e in payload['size_edges']]}")
    for bucket, qhat in enumerate(payload["qhat_box"]):
        note = " (pooled fallback)" if bucket in payload["qhat_box_fallback_buckets"] else ""
        print(f"  bucket {bucket}: qhat={qhat:.4f} from {payload['bucket_score_counts'][bucket]} scores{note}")
    own_models = sum(m is not None for m in payload["isotonic_by_bucket"])
    print(f"Buckets with their own probability model: {own_models}/{len(payload['isotonic_by_bucket'])}")
    print(f"Correctness qhat: correct={payload['qhat_correct']:.4f}, incorrect={payload['qhat_incorrect']:.4f}")
    print(f"Matched objects: {payload['matched_objects']}; missed ground-truth objects: {payload['missed_ground_truth_objects']}")
    if payload["missed_ground_truth_objects"]:
        print("Note: CP regions cover localization of matched objects; misses remain a recall error.")


if __name__ == "__main__":
    main()
