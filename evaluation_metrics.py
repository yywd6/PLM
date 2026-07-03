"""Object and point metrics for the trainable baseline."""

import numpy as np
from sklearn.metrics import average_precision_score, auc, roc_auc_score


def safe_binary_metrics(labels, scores):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size == 0 or np.unique(labels).size < 2:
        return {"auroc": None, "ap": None}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
    }


def point_aupro(score_list, label_list, fpr_limit=0.3, num_thresholds=200):
    scores = [np.asarray(value, dtype=np.float64).reshape(-1) for value in score_list]
    labels = [np.asarray(value, dtype=bool).reshape(-1) for value in label_list]
    if not scores or not any(mask.any() for mask in labels):
        return None
    total_normal = sum(int((~mask).sum()) for mask in labels)
    if total_normal == 0:
        return None
    finite = np.concatenate([value[np.isfinite(value)] for value in scores])
    if finite.size == 0 or finite.min() == finite.max():
        return None
    thresholds = np.linspace(finite.max() + 1e-12, finite.min() - 1e-12, num_thresholds)
    fprs, pros = [], []
    for threshold in thresholds:
        false_positives = 0
        overlaps = []
        for sample_scores, sample_labels in zip(scores, labels):
            prediction = sample_scores >= threshold
            false_positives += int(np.logical_and(prediction, ~sample_labels).sum())
            if sample_labels.any():
                overlaps.append(float(prediction[sample_labels].mean()))
        fprs.append(false_positives / total_normal)
        pros.append(float(np.mean(overlaps)) if overlaps else 0.0)
    order = np.argsort(fprs, kind="mergesort")
    fprs = np.asarray(fprs)[order]
    pros = np.asarray(pros)[order]
    unique_fprs = np.unique(fprs)
    unique_pros = np.asarray([pros[fprs == value].max() for value in unique_fprs])
    fprs = np.concatenate(([0.0], unique_fprs))
    pros = np.concatenate(([0.0], unique_pros))
    keep = fprs <= fpr_limit
    clipped_fprs, clipped_pros = fprs[keep], pros[keep]
    if clipped_fprs.size == 0 or clipped_fprs[-1] < fpr_limit:
        clipped_fprs = np.append(clipped_fprs, fpr_limit)
        clipped_pros = np.append(clipped_pros, np.interp(fpr_limit, fprs, pros))
    return float(auc(clipped_fprs, clipped_pros) / fpr_limit)


def finite_mean(values):
    values = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None
