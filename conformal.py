"""Split-conformal uncertainty outputs for YOLO-style object detectors.

The calibration unit is an image. For each calibration image, the score is
the worst normalized coordinate error among class-aware, IoU-matched objects.
Consequently, the interval is calibrated for localization of matched objects;
it does not create predictions for objects the detector missed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def finite_sample_quantile(scores, alpha):
    """Return the split-conformal order statistic with finite-sample correction."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("At least one calibration score is required")
    if not np.all(np.isfinite(values)):
        raise ValueError("Calibration scores must be finite")

    rank = int(np.ceil((values.size + 1) * (1 - alpha)))
    if rank > values.size:
        return float("inf"), rank
    return float(np.partition(values, rank - 1)[rank - 1]), rank


def box_score(pred_xyxy, target_xyxy):
    """Max normalized coordinate residual for one predicted/true box pair."""
    pred = np.asarray(pred_xyxy, dtype=float)
    target = np.asarray(target_xyxy, dtype=float)
    if pred.shape != (4,) or target.shape != (4,):
        raise ValueError("Boxes must have shape (4,) in xyxy format")

    pred_width = max(pred[2] - pred[0], 1.0)
    pred_height = max(pred[3] - pred[1], 1.0)
    scale = np.array([pred_width, pred_height, pred_width, pred_height])
    return float(np.max(np.abs(target - pred) / scale))


def expanded_box(pred_xyxy, qhat):
    """Return an outer rectangle containing the calibrated coordinate region."""
    pred = np.asarray(pred_xyxy, dtype=float)
    if pred.shape != (4,):
        raise ValueError("Box must have shape (4,) in xyxy format")
    if not np.isfinite(qhat):
        raise ValueError("The conformal quantile is infinite; no finite region is available")

    width = max(pred[2] - pred[0], 1.0)
    height = max(pred[3] - pred[1], 1.0)
    dx, dy = qhat * width, qhat * height
    return np.array([pred[0] - dx, pred[1] - dy, pred[2] + dx, pred[3] + dy])


def correctness_prediction_set(confidence, qhat):
    """Return CP labels for whether one candidate detection is correct.

    The raw detector confidence is used as a binary score pair
    (p(correct)=confidence, p(incorrect)=1-confidence). CP supplies the set
    coverage statement; this does not turn confidence into a calibrated
    probability.
    """
    confidence = float(confidence)
    if not 0 <= confidence <= 1:
        raise ValueError("Detection confidence must be between 0 and 1")
    if not np.isfinite(qhat):
        raise ValueError("The correctness conformal quantile is not finite")
    labels = set()
    if 1.0 - confidence <= qhat:
        labels.add("correct")
    if confidence <= qhat:
        labels.add("incorrect")
    return labels


def correctness_priority(confidence, qhat):
    """Return an explicit rerank score based on the CP correctness set.

    Ordering is: singleton {correct}, ambiguous {correct, incorrect}, empty,
    singleton {incorrect}. Raw detector confidence breaks ties within a tier.
    This is a ranking heuristic, not a probability or a CP guarantee on AP.
    """
    labels = correctness_prediction_set(confidence, qhat)
    tier = {
        frozenset({"incorrect"}): 0,
        frozenset(): 1,
        frozenset({"correct", "incorrect"}): 2,
        frozenset({"correct"}): 3,
    }[frozenset(labels)]
    return float(tier * 2 + float(confidence))


def box_iou_matrix(boxes_a, boxes_b):
    """Pairwise IoU for xyxy boxes."""
    a = np.asarray(boxes_a, dtype=float).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)

    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.prod(np.maximum(bottom_right - top_left, 0), axis=2)
    area_a = np.prod(np.maximum(a[:, 2:] - a[:, :2], 0), axis=1)
    area_b = np.prod(np.maximum(b[:, 2:] - b[:, :2], 0), axis=1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def greedy_class_aware_matches(pred_boxes, pred_classes, target_boxes, target_classes, iou_threshold):
    """Greedily match highest-IoU class-consistent prediction/target pairs."""
    pred_boxes = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
    target_boxes = np.asarray(target_boxes, dtype=float).reshape(-1, 4)
    pred_classes = np.asarray(pred_classes, dtype=int).reshape(-1)
    target_classes = np.asarray(target_classes, dtype=int).reshape(-1)
    if len(pred_boxes) != len(pred_classes) or len(target_boxes) != len(target_classes):
        raise ValueError("Box and class counts do not match")

    ious = box_iou_matrix(pred_boxes, target_boxes)
    if len(pred_boxes) and len(target_boxes):
        ious[pred_classes[:, None] != target_classes[None, :]] = 0.0
    candidates = np.argwhere(ious >= iou_threshold)
    candidates = sorted(candidates, key=lambda pair: ious[pair[0], pair[1]], reverse=True)

    matched_predictions, matched_targets, matches = set(), set(), []
    for pred_idx, target_idx in candidates:
        pred_idx, target_idx = int(pred_idx), int(target_idx)
        if pred_idx not in matched_predictions and target_idx not in matched_targets:
            matches.append((pred_idx, target_idx))
            matched_predictions.add(pred_idx)
            matched_targets.add(target_idx)
    return matches


def detection_true_positive_matrix(pred_boxes, pred_classes, target_boxes, target_classes):
    """Match detections at IoU 0.50:0.95 using Ultralytics' greedy policy."""
    pred_boxes = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
    target_boxes = np.asarray(target_boxes, dtype=float).reshape(-1, 4)
    pred_classes = np.asarray(pred_classes, dtype=int).reshape(-1)
    target_classes = np.asarray(target_classes, dtype=int).reshape(-1)
    correct = np.zeros((len(pred_boxes), 10), dtype=bool)
    if not len(pred_boxes) or not len(target_boxes):
        return correct

    # Ultralytics matching uses rows=targets and columns=predictions.
    ious = box_iou_matrix(target_boxes, pred_boxes)
    ious[target_classes[:, None] != pred_classes[None, :]] = 0.0
    for threshold_idx, threshold in enumerate(np.linspace(0.5, 0.95, 10)):
        matches = np.argwhere(ious >= threshold)
        if not len(matches):
            continue
        matches = matches[np.argsort(ious[matches[:, 0], matches[:, 1]])[::-1]]
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int), threshold_idx] = True
    return correct


def read_yolo_labels(label_path, image_width, image_height):
    """Read normalized YOLO class,x_center,y_center,width,height labels."""
    label_path = Path(label_path)
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label file: {label_path}")

    rows = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) < 5:
            raise ValueError(f"Expected 5 YOLO label values at {label_path}:{line_number}")
        cls, xc, yc, width, height = map(float, values[:5])
        x1 = (xc - width / 2) * image_width
        y1 = (yc - height / 2) * image_height
        x2 = (xc + width / 2) * image_width
        y2 = (yc + height / 2) * image_height
        rows.append((int(cls), x1, y1, x2, y2))

    if not rows:
        return np.zeros((0,), dtype=int), np.zeros((0, 4), dtype=float)
    rows = np.asarray(rows, dtype=float)
    return rows[:, 0].astype(int), rows[:, 1:5]


def image_files(images_dir):
    """List supported image files in a directory in stable order."""
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {images_dir}")
    files = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise ValueError(f"No supported images found in {images_dir}")
    return files


def save_calibrator(path, *, alpha, qhat, rank, image_count, matched_count, missed_count,
                    qhat_correctness, correctness_rank, correctness_images,
                    model, conf, iou, imgsz, match_iou):
    """Save calibration metadata and the inference settings needed to reuse it."""
    if not np.isfinite(qhat):
        raise ValueError(
            "The finite-sample conformal quantile is infinite for this calibration size and alpha. "
            "Collect more independent calibration images or choose a larger alpha."
        )
    if not np.isfinite(qhat_correctness):
        raise ValueError(
            "The correctness conformal quantile is infinite. Collect more calibration images "
            "or choose a larger alpha."
        )
    payload = {
        "method": "image_max_split_conformal_box_and_detection_correctness",
        "alpha": float(alpha),
        "target_coverage": float(1 - alpha),
        "qhat": float(qhat),
        "quantile_rank": int(rank),
        "qhat_correctness": float(qhat_correctness),
        "correctness_quantile_rank": int(correctness_rank),
        "correctness_calibration_images": int(correctness_images),
        "calibration_images_with_matches": int(image_count),
        "matched_objects": int(matched_count),
        "missed_ground_truth_objects": int(missed_count),
        "score": "max(abs(gt_xyxy - pred_xyxy) / [pred_width,pred_height,pred_width,pred_height])",
        "correctness_score": "max over detections in an image of 1 - p(true correct/incorrect label)",
        "scope": (
            "localization: joint coverage of matched objects in images with at least one match; "
            "correctness: simultaneous label-set coverage of all candidate detections per image"
        ),
        "missed_objects_scope": "misses are excluded from localization scores; recall must be reported separately",
        "model": str(model),
        "predict_conf": float(conf),
        "predict_iou": float(iou),
        "imgsz": int(imgsz),
        "match_iou": float(match_iou),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_calibrator(path):
    """Load and validate a saved calibration file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "method", "alpha", "qhat", "qhat_correctness", "predict_conf",
        "predict_iou", "imgsz", "match_iou"
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Calibration file is missing fields: {sorted(missing)}")
    if payload["method"] != "image_max_split_conformal_box_and_detection_correctness":
        raise ValueError(f"Unsupported calibration method: {payload['method']}")
    if not np.isfinite(float(payload["qhat"])):
        raise ValueError("Calibration file has a non-finite qhat")
    return payload
