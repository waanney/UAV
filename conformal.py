from __future__ import annotations

import json
from pathlib import Path

import numpy as np

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
METHOD = "size_conditional_split_conformal_v2"
RANK_TIE_WEIGHT = 1e-6  # tiny confidence term that breaks ties between equal p


# --------------------------------------------------------------------------
# Quantile
# --------------------------------------------------------------------------
def finite_sample_quantile(scores, alpha):
    """Split-conformal quantile. Returns (value, rank); value is inf if too few scores."""
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


# --------------------------------------------------------------------------
# Boxes
# --------------------------------------------------------------------------
def box_size(boxes):
    """sqrt(width * height) in pixels for xyxy boxes (min side clamped to 1)."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    width = np.maximum(boxes[:, 2] - boxes[:, 0], 1.0)
    height = np.maximum(boxes[:, 3] - boxes[:, 1], 1.0)
    return np.sqrt(width * height)


def box_score(pred_xyxy, target_xyxy):
    """Largest coordinate error, divided by the predicted box width/height."""
    pred = np.asarray(pred_xyxy, dtype=float)
    target = np.asarray(target_xyxy, dtype=float)
    if pred.shape != (4,) or target.shape != (4,):
        raise ValueError("Boxes must have shape (4,) in xyxy format")
    width = max(pred[2] - pred[0], 1.0)
    height = max(pred[3] - pred[1], 1.0)
    scale = np.array([width, height, width, height])
    return float(np.max(np.abs(target - pred) / scale))


def expanded_boxes(boxes, qhat):
    """Enlarge each xyxy box by qhat * its own width/height. qhat may be per box."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    qhat = np.broadcast_to(np.asarray(qhat, dtype=float), (len(boxes),))
    if not np.all(np.isfinite(qhat)):
        raise ValueError("qhat must be finite")
    width = np.maximum(boxes[:, 2] - boxes[:, 0], 1.0)
    height = np.maximum(boxes[:, 3] - boxes[:, 1], 1.0)
    dx, dy = qhat * width, qhat * height
    return np.stack(
        [boxes[:, 0] - dx, boxes[:, 1] - dy, boxes[:, 2] + dx, boxes[:, 3] + dy], axis=1
    )


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

    used_predictions, used_targets, matches = set(), set(), []
    for pred_idx, target_idx in candidates:
        pred_idx, target_idx = int(pred_idx), int(target_idx)
        if pred_idx not in used_predictions and target_idx not in used_targets:
            matches.append((pred_idx, target_idx))
            used_predictions.add(pred_idx)
            used_targets.add(target_idx)
    return matches


def detection_true_positive_matrix(pred_boxes, pred_classes, target_boxes, target_classes):
    """Match detections at IoU 0.50:0.95 with the Ultralytics greedy policy."""
    pred_boxes = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
    target_boxes = np.asarray(target_boxes, dtype=float).reshape(-1, 4)
    pred_classes = np.asarray(pred_classes, dtype=int).reshape(-1)
    target_classes = np.asarray(target_classes, dtype=int).reshape(-1)
    correct = np.zeros((len(pred_boxes), 10), dtype=bool)
    if not len(pred_boxes) or not len(target_boxes):
        return correct

    ious = box_iou_matrix(target_boxes, pred_boxes)  # rows=targets, cols=predictions
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


# --------------------------------------------------------------------------
# Files / detector output
# --------------------------------------------------------------------------
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
        rows.append((
            int(cls),
            (xc - width / 2) * image_width,
            (yc - height / 2) * image_height,
            (xc + width / 2) * image_width,
            (yc + height / 2) * image_height,
        ))

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


def detector_arrays(result):
    """Turn one Ultralytics result into (boxes_xyxy, classes, confidences) numpy arrays."""
    if result.boxes is None or len(result.boxes) == 0:
        return np.zeros((0, 4), dtype=float), np.zeros((0,), dtype=int), np.zeros((0,), dtype=float)
    boxes = result.boxes.xyxy.cpu().numpy().astype(float)
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy().astype(float)
    return boxes, classes, confidences


def image_record(pred_boxes, pred_classes, pred_conf, gt_boxes, gt_classes, match_iou):
    """Bundle one image's detections, ground truth and matching into a dict."""
    matches = greedy_class_aware_matches(pred_boxes, pred_classes, gt_boxes, gt_classes, match_iou)
    correct = np.zeros(len(pred_boxes), dtype=bool)
    for pred_idx, _ in matches:
        correct[pred_idx] = True
    return {
        "boxes": np.asarray(pred_boxes, dtype=float).reshape(-1, 4),
        "classes": np.asarray(pred_classes, dtype=int).reshape(-1),
        "conf": np.asarray(pred_conf, dtype=float).reshape(-1),
        "gt_boxes": np.asarray(gt_boxes, dtype=float).reshape(-1, 4),
        "gt_classes": np.asarray(gt_classes, dtype=int).reshape(-1),
        "matches": matches,
        "correct": correct,
    }


def collect_record(model, image_path, labels_dir, *, conf, iou, imgsz, match_iou):
    """Run the detector on one image and return its image_record."""
    import cv2  # imported here so the maths in this file works without OpenCV

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    gt_classes, gt_boxes = read_yolo_labels(
        Path(labels_dir) / f"{Path(image_path).stem}.txt", width, height
    )
    result = model.predict(source=image, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
    boxes, classes, confidences = detector_arrays(result)
    return image_record(boxes, classes, confidences, gt_boxes, gt_classes, match_iou)


# --------------------------------------------------------------------------
# Size buckets
# --------------------------------------------------------------------------
def make_size_edges(sizes, n_buckets):
    """Bucket edges (quantiles of sizes). n_buckets=3 gives small/medium/large."""
    sizes = np.asarray(sizes, dtype=float)
    if n_buckets <= 1 or sizes.size == 0:
        return []
    quantile_points = np.linspace(0, 1, n_buckets + 1)[1:-1]
    return [float(e) for e in np.unique(np.quantile(sizes, quantile_points))]


def size_buckets(boxes, edges):
    """Bucket index (0 = smallest) for each xyxy box."""
    return np.searchsorted(np.asarray(edges, dtype=float), box_size(boxes), side="right")


# --------------------------------------------------------------------------
# Isotonic regression (pure numpy): calibrated P(correct | confidence)
# --------------------------------------------------------------------------
def fit_isotonic(x, y, max_points=2000):
    """Fit a non-decreasing step/line function y ~ f(x) with the PAV algorithm."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or x.size != y.size:
        raise ValueError("x and y must be non-empty and have equal length")

    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]

    # Pre-merge into equal-count groups so very large inputs stay fast.
    groups = np.array_split(np.arange(x.size), min(max_points, x.size))
    means, weights, x_sums = [], [], []
    for group in groups:
        means.append(float(y[group].mean()))
        weights.append(float(len(group)))
        x_sums.append(float(x[group].sum()))
        # Pool neighbouring blocks while the order is violated.
        while len(means) > 1 and means[-2] > means[-1]:
            w = weights[-2] + weights[-1]
            m = (means[-2] * weights[-2] + means[-1] * weights[-1]) / w
            s = x_sums[-2] + x_sums[-1]
            del means[-2:], weights[-2:], x_sums[-2:]
            means.append(m)
            weights.append(w)
            x_sums.append(s)

    block_x = [s / w for s, w in zip(x_sums, weights)]
    return {"x": block_x, "p": means}


def predict_isotonic(model, confidence):
    """Evaluate a fitted isotonic model (linear between block centres, flat outside)."""
    return np.interp(np.asarray(confidence, dtype=float), model["x"], model["p"])


# --------------------------------------------------------------------------
# Fitting the calibrator
# --------------------------------------------------------------------------
def _group_max(entries):
    """Max score per key for an iterable of (key, score)."""
    best = {}
    for key, score in entries:
        best[key] = max(score, best.get(key, -np.inf))
    return best


def _concat(parts, empty_shape, dtype):
    parts = [p for p in parts if len(p)]
    return np.concatenate(parts) if parts else np.zeros(empty_shape, dtype=dtype)


def fit_calibrator(fit_records, calib_records, *, alpha, n_buckets=3,
                   min_bucket_samples=200, score_unit="object"):
    """Build the calibrator dict.

    fit_records   : images used for size edges + isotonic probability model.
    calib_records : DIFFERENT images used only for the conformal quantiles.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if score_unit not in ("object", "image"):
        raise ValueError("score_unit must be 'object' or 'image'")

    # ---- 1. size edges from matched predicted boxes (fit images only)
    matched_boxes = _concat(
        [r["boxes"][[p for p, _ in r["matches"]]] for r in fit_records if r["matches"]],
        (0, 4), float,
    )
    if len(matched_boxes) == 0:
        raise RuntimeError("No matched detections in the fit images; cannot build size buckets")
    edges = make_size_edges(box_size(matched_boxes), n_buckets)
    bucket_count = len(edges) + 1

    # ---- 2. isotonic P(correct | conf), global and per size bucket
    fit_boxes = _concat([r["boxes"] for r in fit_records], (0, 4), float)
    fit_conf = _concat([r["conf"] for r in fit_records], (0,), float)
    fit_correct = _concat([r["correct"] for r in fit_records], (0,), bool)
    fit_bucket = size_buckets(fit_boxes, edges)
    if fit_correct.sum() == 0 or fit_correct.all():
        raise RuntimeError("Fit images need both correct and incorrect detections")

    isotonic_global = fit_isotonic(fit_conf, fit_correct.astype(float))
    isotonic_by_bucket = []
    for bucket in range(bucket_count):
        mask = fit_bucket == bucket
        positives = int(fit_correct[mask].sum())
        negatives = int(mask.sum()) - positives
        enough = mask.sum() >= min_bucket_samples and positives >= 10 and negatives >= 10
        isotonic_by_bucket.append(
            fit_isotonic(fit_conf[mask], fit_correct[mask].astype(float)) if enough else None
        )

    calibrator = {
        "method": METHOD,
        "alpha": float(alpha),
        "target_coverage": float(1 - alpha),
        "score_unit": score_unit,
        "size_edges": edges,
        "isotonic_global": isotonic_global,
        "isotonic_by_bucket": isotonic_by_bucket,
    }

    # ---- 3. conformal scores on the separate calibration images
    scores_correct, scores_incorrect = [], []   # label-conditional correctness scores
    localization = []                            # (image_index, bucket, score)
    matched_total, missed_total = 0, 0
    for image_index, record in enumerate(calib_records):
        matched_total += len(record["matches"])
        missed_total += len(record["gt_boxes"]) - len(record["matches"])
        if len(record["boxes"]):
            p = calibrated_probabilities(calibrator, record["boxes"], record["conf"])
            correct = record["correct"]
            scores_correct.extend((1.0 - p[correct]).tolist())   # true label: correct
            scores_incorrect.extend(p[~correct].tolist())        # true label: incorrect
        buckets = size_buckets(record["boxes"], edges) if len(record["boxes"]) else []
        for pred_idx, gt_idx in record["matches"]:
            score = box_score(record["boxes"][pred_idx], record["gt_boxes"][gt_idx])
            localization.append((image_index, int(buckets[pred_idx]), score))

    if not localization:
        raise RuntimeError("No matched objects in the calibration images; cannot calibrate localization")

    # ---- 4. localization quantiles (per bucket, pooled fallback)
    bucket_scores = [[] for _ in range(bucket_count)]
    if score_unit == "image":
        for (_, bucket), score in _group_max(((i, b), s) for i, b, s in localization).items():
            bucket_scores[bucket].append(score)
        pooled_scores = list(_group_max((i, s) for i, _, s in localization).values())
    else:
        for _, bucket, score in localization:
            bucket_scores[bucket].append(score)
        pooled_scores = [s for _, _, s in localization]

    pooled_qhat, pooled_rank = finite_sample_quantile(pooled_scores, alpha)
    if not np.isfinite(pooled_qhat):
        raise ValueError("Too few calibration images for this alpha; add images or raise alpha")

    qhat_box, box_ranks, fallback_buckets = [], [], []
    for bucket in range(bucket_count):
        if bucket_scores[bucket]:
            q, rank = finite_sample_quantile(bucket_scores[bucket], alpha)
        else:
            q, rank = float("inf"), 0
        if not np.isfinite(q):
            q, rank = pooled_qhat, pooled_rank
            fallback_buckets.append(bucket)
        qhat_box.append(float(q))
        box_ranks.append(int(rank))

    # ---- 5. correctness quantiles (one per true label)
    if not scores_correct or not scores_incorrect:
        raise RuntimeError("Calibration images need both correct and incorrect detections")
    qhat_correct, rank_correct = finite_sample_quantile(scores_correct, alpha)
    qhat_incorrect, rank_incorrect = finite_sample_quantile(scores_incorrect, alpha)
    if not (np.isfinite(qhat_correct) and np.isfinite(qhat_incorrect)):
        raise ValueError("Too few calibration detections for this alpha; add images or raise alpha")

    calibrator.update({
        "qhat_box": qhat_box,
        "qhat_box_ranks": box_ranks,
        "qhat_box_fallback_buckets": fallback_buckets,
        "qhat_box_pooled": float(pooled_qhat),
        "bucket_score_counts": [len(s) for s in bucket_scores],
        "qhat_correct": float(qhat_correct),
        "qhat_incorrect": float(qhat_incorrect),
        "correctness_counts": {"correct": len(scores_correct), "incorrect": len(scores_incorrect)},
        "fit_images": len(fit_records),
        "calibration_images": len(calib_records),
        "matched_objects": int(matched_total),
        "missed_ground_truth_objects": int(missed_total),
        "scope": (
            "localization: per-size-bucket coverage of matched objects "
            "(score_unit=image makes it joint per image and size bucket); "
            "correctness: label-conditional coverage (each true label separately) "
            "over candidate detections. Detections in one image are dependent, so "
            "treat these as approximate. Missed objects are excluded."
        ),
    })
    return calibrator


# --------------------------------------------------------------------------
# Using the calibrator
# --------------------------------------------------------------------------
def calibrated_probabilities(calibrator, boxes, confidence):
    """Calibrated P(detection is correct) given its size bucket and confidence."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    confidence = np.asarray(confidence, dtype=float).reshape(-1)
    if len(boxes) != len(confidence):
        raise ValueError("boxes and confidence must have the same length")
    probabilities = np.zeros(len(confidence), dtype=float)
    if len(confidence) == 0:
        return probabilities

    buckets = size_buckets(boxes, calibrator["size_edges"])
    for bucket in np.unique(buckets):
        model = calibrator["isotonic_by_bucket"][int(bucket)] or calibrator["isotonic_global"]
        mask = buckets == bucket
        probabilities[mask] = predict_isotonic(model, confidence[mask])
    return np.clip(probabilities, 0.0, 1.0)


def correctness_sets(calibrator, probabilities):
    """Return (label_correct_in_set, label_incorrect_in_set) boolean arrays."""
    p = np.asarray(probabilities, dtype=float)
    in_correct = (1.0 - p) <= float(calibrator["qhat_correct"])
    in_incorrect = p <= float(calibrator["qhat_incorrect"])
    return in_correct, in_incorrect


def set_name(in_correct, in_incorrect):
    """Human-readable name of one correctness set."""
    if in_correct and in_incorrect:
        return "uncertain"
    if in_correct:
        return "correct"
    if in_incorrect:
        return "incorrect"
    return "empty"


def localization_qhat(calibrator, boxes):
    """The calibrated qhat for each box, chosen by its size bucket."""
    buckets = size_buckets(boxes, calibrator["size_edges"])
    return np.asarray(calibrator["qhat_box"], dtype=float)[buckets]


def localization_regions(calibrator, boxes):
    """Enlarged uncertainty box for each detection."""
    return expanded_boxes(boxes, localization_qhat(calibrator, boxes))


def rank_scores(probabilities, confidence):
    """Score used for reranking: calibrated p, raw confidence only breaks ties."""
    return np.asarray(probabilities, dtype=float) + RANK_TIE_WEIGHT * np.asarray(confidence, dtype=float)


# --------------------------------------------------------------------------
# Save / load
# --------------------------------------------------------------------------
def save_calibrator(path, calibrator, *, model, conf, iou, imgsz, match_iou):
    """Write the calibrator plus the detector settings needed to reuse it."""
    payload = dict(calibrator)
    payload.update({
        "model": str(model),
        "predict_conf": float(conf),
        "predict_iou": float(iou),
        "imgsz": int(imgsz),
        "match_iou": float(match_iou),
    })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_calibrator(path):
    """Load and validate a saved calibrator."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "method", "alpha", "size_edges", "isotonic_global", "isotonic_by_bucket",
        "qhat_box", "qhat_correct", "qhat_incorrect",
        "predict_conf", "predict_iou", "imgsz", "match_iou",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Calibration file is missing fields: {sorted(missing)}")
    if payload["method"] != METHOD:
        raise ValueError(
            f"Unsupported calibration method '{payload['method']}'. "
            "Re-run calibrate_conformal.py with this version."
        )
    if len(payload["qhat_box"]) != len(payload["size_edges"]) + 1:
        raise ValueError("Calibration file is inconsistent: qhat_box vs size_edges")
    if not np.all(np.isfinite(payload["qhat_box"])):
        raise ValueError("Calibration file has a non-finite qhat")
    return payload
