import torch

from models.trainable_baseline import (
    MultiLayerPatchAdapter,
    aggregate_object_probability,
    select_multi_layer_tokens,
)


def test_all_four_layers_are_projected_and_fused():
    feature_layers = [2, 5, 8, 11]
    features = {layer: torch.randn(2, 4, 6) for layer in feature_layers}
    indices = torch.tensor([[[0, 1], [1, 2], [2, 3]]] * 2)
    tokens = select_multi_layer_tokens(features, indices, feature_layers)
    adapter = MultiLayerPatchAdapter(6, 10, num_layers=4)
    output = adapter(tokens)
    assert output.shape == (2, 3, 10)
    assert torch.isfinite(output).all()
    assert torch.allclose(adapter.layer_weights(), torch.full((4,), 0.25))


def test_layer_weight_receives_gradient():
    adapter = MultiLayerPatchAdapter(4, 5, num_layers=4)
    output = adapter([torch.randn(1, 3, 4) for _ in range(4)])
    output[..., 0].sum().backward()
    assert adapter.layer_logits.grad is not None
    assert torch.isfinite(adapter.layer_logits.grad).all()


def test_object_probability_combines_global_and_top_patch_probabilities():
    global_logits = torch.tensor([0.0, 2.0])
    patch_logits = torch.tensor([[0.0, 1.0, 2.0, 3.0], [-1.0, 0.0, 1.0, 2.0]])
    actual = aggregate_object_probability(
        global_logits, patch_logits, global_alpha=0.5, top_percent=0.5
    )
    local = torch.topk(torch.sigmoid(patch_logits.double()), k=2, dim=1).values.mean(dim=1)
    expected = 0.5 * torch.sigmoid(global_logits.double()) + 0.5 * local
    assert actual.dtype == torch.float64
    assert torch.allclose(actual, expected)


def test_object_probability_avoids_float32_sigmoid_ties():
    global_logits = torch.zeros(2)
    patch_logits = torch.tensor([[20.0], [19.0]])
    scores = aggregate_object_probability(
        global_logits, patch_logits, global_alpha=0.0, top_percent=1.0
    )
    assert scores[0] > scores[1]
