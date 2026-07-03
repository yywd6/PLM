import numpy as np
import pytest
import torch

from baseline import anomaly_probability, binary_metrics


def test_anomaly_probability_prefers_aligned_prototype():
    points = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    normal = torch.tensor([[1.0, 0.0]])
    anomaly = torch.tensor([[0.0, 1.0]])
    scores = anomaly_probability(points, normal, anomaly, temperature=0.1)
    assert scores[0] < 0.5
    assert scores[1] > 0.5


def test_anomaly_probability_rejects_nonpositive_temperature():
    embedding = torch.tensor([[1.0, 0.0]])
    with pytest.raises(ValueError, match="temperature"):
        anomaly_probability(embedding, embedding, embedding, temperature=0.0)


def test_perfect_binary_metrics():
    assert binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == {
        "auroc": 1.0,
        "ap": 1.0,
        "f1_max": 1.0,
    }


def test_single_class_metrics_are_undefined():
    assert all(np.isnan(value) for value in binary_metrics([0, 0], [0.1, 0.2]).values())
