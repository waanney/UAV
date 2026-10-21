"""Fit split-conformal box and detection-correctness calibrators for YOLO.

Example:
    python calibrate_conformal.py --model path/to/best.pt \
        --images path/to/VisDrone2019-DET-val/images \
        --labels path/to/VisDrone2019-DET-val/labels \
        --output runs/conformal/visdrone_q90.json --alpha 0.1

Use images that were not used to train, early-stop, or tune the selected model.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from conformal import (
    box_score,
    finite_sample_quantile,
    greedy_class_aware_matches,
    image_files,
    read_yolo_labels,
    save_calibrator,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Trained YOLO/YOLO-World .pt weights")
    parser.add_argument("--images", required=True, help="Held-out calibration image directory")
    parser.add_argument("--labels", required=True, help="YOLO-format labels directory")
    parser.add_argument("--output", default="runs/conformal/calibrator.json")
    parser.add_argument("--alpha", type=float, default=0.1, help="Miscoverage rate; 0.1 targets 90%%")
    parser.add_argument("--conf", type=float, default=0.001, help="Candidate-generation confidence cutoff")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold")
    parser.add_argument("--match-iou", type=float, default=0.5, help="GT matching IoU threshold")
    parser.add_argument("--imgsz", type=int, default=1536)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must be between 0 and 1")
    if not 0 <= args.conf < 1 or not 0 <= args.iou <= 1 or not 0 <= args.match_iou <= 1:
        raise ValueError("--conf, --iou, and --match-iou must be within [0, 1]")

    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    model = YOLO(str(model_path))
    scores_by_image = []
    correctness_scores_by_image = []
    matched_objects = 0
    missed_objects = 0
    labeled_images = 0

    for image_path in image_files(args.images):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        height, width = image.shape[:2]
        gt_classes, gt_boxes = read_yolo_labels(Path(args.labels) / f"{image_path.stem}.txt", width, height)
        labeled_images += 1

        result = model.predict(
            source=image,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            pred_boxes = np.zeros((0, 4), dtype=float)
            pred_classes = np.zeros((0,), dtype=int)
            pred_confidences = np.zeros((0,), dtype=float)
        else:
            pred_boxes = result.boxes.xyxy.cpu().numpy()
            pred_classes = result.boxes.cls.cpu().numpy().astype(int)
            pred_confidences = result.boxes.conf.cpu().numpy()

        matches = greedy_class_aware_matches(
            pred_boxes, pred_classes, gt_boxes, gt_classes, args.match_iou
        )
        missed_objects += len(gt_boxes) - len(matches)
        matched_objects += len(matches)
        # Calibrate the set containing the true correctness label for every
        # candidate detection. One max score per image handles within-image
        # dependence; empty candidate sets have a vacuous score of zero.
        matched_prediction_ids = {pred_idx for pred_idx, _ in matches}
        correctness_scores = []
        for pred_idx, confidence in enumerate(pred_confidences):
            true_probability = (
                float(confidence) if pred_idx in matched_prediction_ids else 1.0 - float(confidence)
            )
            correctness_scores.append(1.0 - true_probability)
        correctness_scores_by_image.append(max(correctness_scores, default=0.0))

        if matches:
            image_scores = [box_score(pred_boxes[p], gt_boxes[g]) for p, g in matches]
            # One score per image accounts for dependence between objects in that image.
            scores_by_image.append(max(image_scores))

    if not scores_by_image:
        raise RuntimeError("No ground-truth objects matched; cannot calibrate localization regions")

    qhat, rank = finite_sample_quantile(scores_by_image, args.alpha)
    qhat_correctness, correctness_rank = finite_sample_quantile(
        correctness_scores_by_image, args.alpha
    )
    payload = save_calibrator(
        args.output,
        alpha=args.alpha,
        qhat=qhat,
        rank=rank,
        image_count=len(scores_by_image),
        matched_count=matched_objects,
        missed_count=missed_objects,
        qhat_correctness=qhat_correctness,
        correctness_rank=correctness_rank,
        correctness_images=len(correctness_scores_by_image),
        model=model_path,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        match_iou=args.match_iou,
    )

    print(f"Saved conformal calibrator: {args.output}")
    print(f"Target marginal coverage: {payload['target_coverage']:.1%}")
    print(f"Calibration images with matches: {len(scores_by_image)} / {labeled_images}")
    print(f"Matched objects: {matched_objects}; missed ground-truth objects: {missed_objects}")
    print(f"qhat: {qhat:.6g} (finite-sample rank {rank})")
    print(
        f"Correctness-label qhat: {qhat_correctness:.6g} "
        f"(finite-sample rank {correctness_rank})"
    )
    if missed_objects:
        print("Note: CP intervals cover localization of matched objects; misses remain a recall error.")


if __name__ == "__main__":
    main()
