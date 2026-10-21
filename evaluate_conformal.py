"""Evaluate held-out coverage and efficiency of a conformal box calibrator."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class

from conformal import (
    box_score,
    correctness_prediction_set,
    correctness_priority,
    detection_true_positive_matrix,
    expanded_box,
    greedy_class_aware_matches,
    image_files,
    load_calibrator,
    read_yolo_labels,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Same frozen weights used for calibration")
    parser.add_argument("--calibrator", required=True, help="JSON from calibrate_conformal.py")
    parser.add_argument("--images", required=True, help="Independent labeled test image directory")
    parser.add_argument("--labels", required=True, help="YOLO-format test labels directory")
    parser.add_argument("--report", help="Optional JSON path for machine-readable test metrics")
    args = parser.parse_args()

    calibration = load_calibrator(args.calibrator)
    model = YOLO(args.model)
    qhat = float(calibration["qhat"])
    qhat_correctness = float(calibration["qhat_correctness"])
    match_iou = float(calibration["match_iou"])

    object_scores = []
    correctness_predictions = 0
    correctness_labels_covered = 0
    correctness_set_sizes = []
    correctness_images_covered = 0
    tp_parts = []
    raw_conf_parts = []
    rerank_score_parts = []
    pred_class_parts = []
    target_class_parts = []
    all_matched_covered_images = 0
    images_with_matches = 0
    covered_ground_truth = 0
    ground_truth_total = 0
    enlargement_ratios = []
    missed_ground_truth = 0
    labeled_images = 0

    for image_path in image_files(args.images):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        height, width = image.shape[:2]
        gt_classes, gt_boxes = read_yolo_labels(Path(args.labels) / f"{image_path.stem}.txt", width, height)
        ground_truth_total += len(gt_boxes)
        labeled_images += 1

        result = model.predict(
            source=image,
            conf=float(calibration["predict_conf"]),
            iou=float(calibration["predict_iou"]),
            imgsz=int(calibration["imgsz"]),
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
            pred_boxes, pred_classes, gt_boxes, gt_classes, match_iou
        )
        tp_parts.append(
            detection_true_positive_matrix(pred_boxes, pred_classes, gt_boxes, gt_classes)
        )
        raw_conf_parts.append(pred_confidences)
        rerank_score_parts.append(
            np.asarray(
                [correctness_priority(conf, qhat_correctness) for conf in pred_confidences],
                dtype=float,
            )
        )
        pred_class_parts.append(pred_classes)
        target_class_parts.append(gt_classes)
        matched_prediction_ids = {pred_idx for pred_idx, _ in matches}
        image_correctness_covered = True
        for pred_idx, confidence in enumerate(pred_confidences):
            true_label = "correct" if pred_idx in matched_prediction_ids else "incorrect"
            prediction_set = correctness_prediction_set(confidence, qhat_correctness)
            correctness_predictions += 1
            correctness_set_sizes.append(len(prediction_set))
            label_covered = true_label in prediction_set
            correctness_labels_covered += int(label_covered)
            image_correctness_covered &= label_covered
        correctness_images_covered += int(image_correctness_covered)
        missed_ground_truth += len(gt_boxes) - len(matches)
        image_scores = []
        for pred_idx, gt_idx in matches:
            score = box_score(pred_boxes[pred_idx], gt_boxes[gt_idx])
            object_scores.append(score)
            image_scores.append(score)
            if score <= qhat:
                covered_ground_truth += 1

            base_width = max(pred_boxes[pred_idx, 2] - pred_boxes[pred_idx, 0], 1.0)
            base_height = max(pred_boxes[pred_idx, 3] - pred_boxes[pred_idx, 1], 1.0)
            region = expanded_box(pred_boxes[pred_idx], qhat)
            region_area = max(region[2] - region[0], 0) * max(region[3] - region[1], 0)
            base_area = base_width * base_height
            enlargement_ratios.append(region_area / base_area)

        if image_scores:
            images_with_matches += 1
            all_matched_covered_images += int(max(image_scores) <= qhat)

    box_coverage = float(np.mean(np.asarray(object_scores) <= qhat)) if object_scores else float("nan")
    image_coverage = (
        all_matched_covered_images / images_with_matches if images_with_matches else float("nan")
    )
    all_gt_success = covered_ground_truth / ground_truth_total if ground_truth_total else float("nan")
    mean_enlargement = float(np.mean(enlargement_ratios))
    tp_all = np.concatenate(tp_parts, axis=0)
    raw_ap = ap_per_class(
        tp_all,
        np.concatenate(raw_conf_parts),
        np.concatenate(pred_class_parts),
        np.concatenate(target_class_parts),
    )[5]
    reranked_ap = ap_per_class(
        tp_all,
        np.concatenate(rerank_score_parts),
        np.concatenate(pred_class_parts),
        np.concatenate(target_class_parts),
    )[5]
    report = {
        "alpha": float(calibration["alpha"]),
        "nominal_coverage": 1.0 - float(calibration["alpha"]),
        "correctness_image_coverage": correctness_images_covered / labeled_images if labeled_images else None,
        "correctness_box_label_coverage": (
            correctness_labels_covered / correctness_predictions if correctness_predictions else None
        ),
        "correctness_mean_set_size": float(np.mean(correctness_set_sizes)) if correctness_set_sizes else None,
        "correctness_ambiguous_fraction": (
            float(np.mean(np.asarray(correctness_set_sizes) == 2)) if correctness_set_sizes else None
        ),
        "matched_box_localization_coverage": box_coverage if np.isfinite(box_coverage) else None,
        "matched_image_joint_localization_coverage": image_coverage if np.isfinite(image_coverage) else None,
        "mean_region_to_box_area_ratio": mean_enlargement if enlargement_ratios else None,
        "ground_truth_detection_and_localization_success": all_gt_success if np.isfinite(all_gt_success) else None,
        "ground_truth_total": int(ground_truth_total),
        "ground_truth_missed": int(missed_ground_truth),
        "raw_confidence_map50": float(raw_ap[:, 0].mean()) if raw_ap.size else None,
        "raw_confidence_map50_95": float(raw_ap.mean()) if raw_ap.size else None,
        "cp_reranked_map50": float(reranked_ap[:, 0].mean()) if reranked_ap.size else None,
        "cp_reranked_map50_95": float(reranked_ap.mean()) if reranked_ap.size else None,
        "map_candidate_conf": float(calibration["predict_conf"]),
        "map_nms_iou": float(calibration["predict_iou"]),
    }
    print(f"Nominal matched-object coverage: {1 - float(calibration['alpha']):.1%}")
    print("Empirical per-image correctness-label coverage: "
          f"{correctness_images_covered / labeled_images:.1%} ({labeled_images} labeled images)")
    if correctness_predictions:
        print("Empirical candidate-box correctness-label coverage: "
              f"{correctness_labels_covered / correctness_predictions:.1%} "
              f"({correctness_predictions} boxes)")
        print(f"Mean correctness prediction-set size: {np.mean(correctness_set_sizes):.3f} (0-2)")
        print("Ambiguous correctness sets: "
              f"{np.mean(np.asarray(correctness_set_sizes) == 2):.1%} of candidate boxes")
    else:
        print("No candidate detections; correctness-set metrics are vacuous.")
    print(f"Empirical matched-box localization coverage: {box_coverage:.1%} ({len(object_scores)} boxes)")
    print(f"Empirical image coverage (all matched boxes covered): {image_coverage:.1%} ({images_with_matches} images)")
    print(f"Mean uncertainty-region/base-box area ratio: {mean_enlargement:.3f}x" if enlargement_ratios
          else "No matched detections; localization-region efficiency is unavailable.")
    print(f"Ground-truth objects detected and covered: {all_gt_success:.1%} ({covered_ground_truth}/{ground_truth_total})")
    print(f"Missed ground-truth objects: {missed_ground_truth}/{ground_truth_total}")
    if raw_ap.size:
        print("mAP on the same test detections and candidate cutoff:")
        print(f"  Raw YOLO confidence: mAP50={raw_ap[:, 0].mean():.4f}; mAP50-95={raw_ap.mean():.4f}")
        print(
            "  CP-tier rerank:     "
            f"mAP50={reranked_ap[:, 0].mean():.4f}; mAP50-95={reranked_ap.mean():.4f}"
        )
        print("  CP reranking is a heuristic; do not assume it improves mAP.")
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved test metrics: {report_path}")
    print(
        f"Labeled test images: {labeled_images}; images with at least one matched object: "
        f"{images_with_matches}"
    )
    print("The conformal coverage target applies to localization of matched objects, not missed objects.")


if __name__ == "__main__":
    main()
