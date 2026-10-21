"""Evaluate a conformal calibrator on an independent labeled test set.

Reports: localization coverage (overall, per size bucket, joint per image),
region enlargement, correctness-set coverage per true label, set sizes,
probability quality (Brier/ECE, raw confidence vs calibrated), and mAP with
raw confidence vs calibrated-probability ranking.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class

from conformal import (
    box_score,
    calibrated_probabilities,
    collect_record,
    correctness_sets,
    detection_true_positive_matrix,
    expanded_boxes,
    image_files,
    load_calibrator,
    localization_qhat,
    rank_scores,
    size_buckets,
)


def expected_calibration_error(probabilities, labels, bins=10):
    """Average gap between predicted probability and observed accuracy."""
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(probabilities, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            total += mask.mean() * abs(probabilities[mask].mean() - labels[mask].mean())
    return float(total)


def safe_mean(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()) if values.size else None


def mean_ap(tp, scores, pred_classes, target_classes):
    """mAP50 and mAP50-95 for a given ranking score (None if undefined)."""
    if not len(scores) or not len(target_classes):
        return None, None
    ap = ap_per_class(tp, scores, pred_classes, target_classes)[5]
    if not ap.size:
        return None, None
    return float(ap[:, 0].mean()), float(ap.mean())


def percent(value):
    return "n/a" if value is None else f"{value:.1%}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Same frozen weights used for calibration")
    parser.add_argument("--calibrator", required=True, help="JSON from calibrate_conformal.py")
    parser.add_argument("--images", required=True, help="Independent labeled test image directory")
    parser.add_argument("--labels", required=True, help="YOLO-format test labels directory")
    parser.add_argument("--report", help="Optional JSON path for machine-readable test metrics")
    args = parser.parse_args()

    calibrator = load_calibrator(args.calibrator)
    model = YOLO(args.model)
    qhat_box = np.asarray(calibrator["qhat_box"], dtype=float)
    bucket_count = len(qhat_box)

    # ---- collect everything per image
    loc_scores, loc_buckets, loc_covered = [], [], []
    ratios = []
    images_with_matches, images_all_covered = 0, 0
    gt_total, gt_missed, gt_covered = 0, 0, 0
    conf_parts, prob_parts, correct_parts, bucket_parts = [], [], [], []
    in_correct_parts, in_incorrect_parts = [], []
    tp_parts, class_parts, target_class_parts = [], [], []
    labeled_images = 0

    for image_path in image_files(args.images):
        record = collect_record(
            model, image_path, args.labels,
            conf=float(calibrator["predict_conf"]), iou=float(calibrator["predict_iou"]),
            imgsz=int(calibrator["imgsz"]), match_iou=float(calibrator["match_iou"]),
        )
        labeled_images += 1
        boxes, conf = record["boxes"], record["conf"]
        gt_total += len(record["gt_boxes"])
        gt_missed += len(record["gt_boxes"]) - len(record["matches"])

        tp_parts.append(detection_true_positive_matrix(boxes, record["classes"], record["gt_boxes"], record["gt_classes"]))
        class_parts.append(record["classes"])
        target_class_parts.append(record["gt_classes"])

        probabilities = calibrated_probabilities(calibrator, boxes, conf)
        in_correct, in_incorrect = correctness_sets(calibrator, probabilities)
        conf_parts.append(conf)
        prob_parts.append(probabilities)
        correct_parts.append(record["correct"])
        bucket_parts.append(size_buckets(boxes, calibrator["size_edges"]) if len(boxes) else np.zeros(0, dtype=int))
        in_correct_parts.append(in_correct)
        in_incorrect_parts.append(in_incorrect)

        image_covered = []
        for pred_idx, gt_idx in record["matches"]:
            score = box_score(boxes[pred_idx], record["gt_boxes"][gt_idx])
            bucket = int(size_buckets(boxes[pred_idx:pred_idx + 1], calibrator["size_edges"])[0])
            covered = score <= qhat_box[bucket]
            loc_scores.append(score)
            loc_buckets.append(bucket)
            loc_covered.append(covered)
            image_covered.append(covered)
            gt_covered += int(covered)
            region = expanded_boxes(boxes[pred_idx:pred_idx + 1], qhat_box[bucket])[0]
            base = max(boxes[pred_idx, 2] - boxes[pred_idx, 0], 1.0) * max(boxes[pred_idx, 3] - boxes[pred_idx, 1], 1.0)
            ratios.append((region[2] - region[0]) * (region[3] - region[1]) / base)
        if image_covered:
            images_with_matches += 1
            images_all_covered += int(all(image_covered))

    # ---- combine
    concat = lambda parts, dtype: np.concatenate(parts) if parts and sum(len(p) for p in parts) else np.zeros(0, dtype=dtype)
    conf_all = concat(conf_parts, float)
    prob_all = concat(prob_parts, float)
    correct_all = concat(correct_parts, bool)
    in_correct_all = concat(in_correct_parts, bool)
    in_incorrect_all = concat(in_incorrect_parts, bool)
    tp_all = np.concatenate(tp_parts, axis=0) if tp_parts else np.zeros((0, 10), dtype=bool)
    class_all = concat(class_parts, int)
    target_class_all = concat(target_class_parts, int)
    loc_covered = np.asarray(loc_covered, dtype=bool)
    loc_buckets = np.asarray(loc_buckets, dtype=int)

    # ---- localization
    per_bucket = []
    for bucket in range(bucket_count):
        mask = loc_buckets == bucket
        per_bucket.append({
            "bucket": bucket,
            "objects": int(mask.sum()),
            "coverage": safe_mean(loc_covered[mask]),
            "qhat": float(qhat_box[bucket]),
        })

    # ---- correctness
    set_sizes = in_correct_all.astype(int) + in_incorrect_all.astype(int)
    coverage_correct = safe_mean(in_correct_all[correct_all])
    coverage_incorrect = safe_mean(in_incorrect_all[~correct_all])

    # ---- ranking quality
    raw_map50, raw_map = mean_ap(tp_all, conf_all, class_all, target_class_all)
    rerank_map50, rerank_map = mean_ap(tp_all, rank_scores(prob_all, conf_all), class_all, target_class_all)

    report = {
        "alpha": float(calibrator["alpha"]),
        "nominal_coverage": 1.0 - float(calibrator["alpha"]),
        "score_unit": calibrator.get("score_unit"),
        "labeled_images": labeled_images,
        "localization_object_coverage": safe_mean(loc_covered),
        "localization_by_size_bucket": per_bucket,
        "localization_image_joint_coverage": (
            images_all_covered / images_with_matches if images_with_matches else None
        ),
        "mean_region_to_box_area_ratio": safe_mean(ratios),
        "ground_truth_total": int(gt_total),
        "ground_truth_missed": int(gt_missed),
        "ground_truth_detected_and_covered": gt_covered / gt_total if gt_total else None,
        "correctness_coverage_when_correct": coverage_correct,
        "correctness_coverage_when_incorrect": coverage_incorrect,
        "correctness_mean_set_size": safe_mean(set_sizes),
        "correctness_uncertain_fraction": safe_mean(set_sizes == 2),
        "correctness_empty_fraction": safe_mean(set_sizes == 0),
        "brier_raw_confidence": safe_mean((conf_all - correct_all) ** 2),
        "brier_calibrated": safe_mean((prob_all - correct_all) ** 2),
        "ece_raw_confidence": expected_calibration_error(conf_all, correct_all) if conf_all.size else None,
        "ece_calibrated": expected_calibration_error(prob_all, correct_all) if prob_all.size else None,
        "raw_confidence_map50": raw_map50,
        "raw_confidence_map50_95": raw_map,
        "calibrated_rerank_map50": rerank_map50,
        "calibrated_rerank_map50_95": rerank_map,
        "map_candidate_conf": float(calibrator["predict_conf"]),
        "map_nms_iou": float(calibrator["predict_iou"]),
    }

    # ---- print
    print(f"Nominal coverage target: {report['nominal_coverage']:.1%}")
    print(f"Localization coverage (matched objects): {percent(report['localization_object_coverage'])}")
    for row in per_bucket:
        print(f"  size bucket {row['bucket']}: {percent(row['coverage'])} over {row['objects']} objects (qhat={row['qhat']:.4f})")
    print(f"Image-joint localization coverage: {percent(report['localization_image_joint_coverage'])} ({images_with_matches} images)")
    if ratios:
        print(f"Mean uncertainty-region/base-box area ratio: {np.mean(ratios):.3f}x")
    print(f"Ground-truth detected and covered: {percent(report['ground_truth_detected_and_covered'])} ({gt_covered}/{gt_total}); missed: {gt_missed}")
    print(f"Correctness-set coverage: correct boxes {percent(coverage_correct)}, incorrect boxes {percent(coverage_incorrect)}")
    if set_sizes.size:
        print(f"Mean set size: {set_sizes.mean():.3f} (0-2); uncertain: {percent(report['correctness_uncertain_fraction'])}; empty: {percent(report['correctness_empty_fraction'])}")
        print(f"Brier score raw -> calibrated: {report['brier_raw_confidence']:.4f} -> {report['brier_calibrated']:.4f}")
        print(f"ECE raw -> calibrated:        {report['ece_raw_confidence']:.4f} -> {report['ece_calibrated']:.4f}")
    if raw_map is not None:
        print("mAP on the same detections and candidate cutoff:")
        print(f"  Raw YOLO confidence:     mAP50={raw_map50:.4f}; mAP50-95={raw_map:.4f}")
        print(f"  Calibrated-p ranking:    mAP50={rerank_map50:.4f}; mAP50-95={rerank_map:.4f}")
        print("  Reranking is a heuristic; keep it only if it beats the raw numbers.")

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved test metrics: {report_path}")
    print("Conformal coverage applies to localization of matched objects, not missed objects.")


if __name__ == "__main__":
    main()
