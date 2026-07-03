import torch

from models.trainable_baseline import MultiLayerPatchAdapter, select_multi_layer_tokens


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
