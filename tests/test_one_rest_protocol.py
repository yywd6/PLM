from argparse import Namespace

import pytest
import torch

from models.trainable_baseline import LinearPatchAdapter, patch_to_point, select_patch_tokens
from one_rest_protocol import assert_dataset_categories, resolve_categories, validate_one_rest_flags


def test_one_rest_excludes_source_category():
    train, test = resolve_categories("Real3D", "one_rest", "car")
    assert train == ["car"]
    assert "car" not in test
    assert len(test) == 11


def test_path_audit_rejects_target_leakage():
    dataset = Namespace(samples=["/data/car/test/a.npz", "/data/duck/test/b.npz"])
    with pytest.raises(RuntimeError, match="leakage"):
        assert_dataset_categories(dataset, ["car"], ["duck"])


def test_protocol_rejects_target_training_and_new_methods():
    base = dict(
        protocol="one_rest",
        exclude_train_category_from_test=True,
        zero_shot_target=True,
        save_per_category_metrics=True,
        save_mean_metrics=True,
        use_target_anomaly_for_training=False,
        use_geometric_cap=False,
        use_geometric_dap=False,
    )
    validate_one_rest_flags(Namespace(**base))
    base["use_target_anomaly_for_training"] = True
    with pytest.raises(ValueError, match="forbidden"):
        validate_one_rest_flags(Namespace(**base))


def test_linear_adapter_and_patch_mapping_shapes():
    adapter = LinearPatchAdapter(4, 6)
    layers = {11: torch.randn(2, 4, 4)}
    indices = torch.tensor([[[0, 1], [1, 2], [2, 3]]] * 2)
    tokens = select_patch_tokens(layers, indices, 11)
    assert adapter(tokens).shape == (2, 3, 6)
    points = patch_to_point(torch.randn(2, 3), indices, 4)
    assert points.shape == (2, 4)
