"""Core utilities for the frozen ULIP-2 zero-shot anomaly baseline."""

from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score


def anomaly_probability(point_embeddings, normal_embedding, anomaly_embedding, temperature=0.07):
    """Return anomaly probabilities from two cosine-similarity logits."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    point_embeddings = F.normalize(point_embeddings, dim=-1)
    normal_embedding = F.normalize(normal_embedding, dim=-1)
    anomaly_embedding = F.normalize(anomaly_embedding, dim=-1)
    normal_logits = point_embeddings @ normal_embedding.T / temperature
    anomaly_logits = point_embeddings @ anomaly_embedding.T / temperature
    return torch.softmax(torch.cat((normal_logits, anomaly_logits), dim=-1), dim=-1)[:, 1]


def f1_max(labels, scores):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels)
    precision = true_positives / np.arange(1, labels.size + 1)
    recall = true_positives / sorted_labels.sum()
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    return float(f1.max())


def binary_metrics(labels: Iterable[int], scores: Iterable[float]):
    labels = np.asarray(list(labels), dtype=np.int64)
    scores = np.asarray(list(scores), dtype=np.float64)
    if labels.size != scores.size:
        raise ValueError("labels and scores must contain the same number of values")
    if labels.size == 0 or np.unique(labels).size < 2:
        return {"auroc": float("nan"), "ap": float("nan"), "f1_max": float("nan")}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
        "f1_max": f1_max(labels, scores),
    }


def finite_mean(values):
    values = [value for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")
