import random

import numpy as np
import torch

from utils.reproducibility import dataloader_seed_kwargs, seed_everything


def _draw(seed):
    report = seed_everything(seed, num_workers=3)
    values = (random.random(), float(np.random.rand()), float(torch.rand(())))
    return report, values


def test_global_seed_is_repeatable_and_changes_between_experiments():
    first_report, first = _draw(111)
    second_report, second = _draw(111)
    _, different = _draw(222)

    assert first == second
    assert first != different
    assert first_report == second_report
    assert first_report["dataloader_worker_seeds"] == [111, 112, 113]


def test_dataloader_generator_uses_experiment_seed():
    first = dataloader_seed_kwargs(111)
    second = dataloader_seed_kwargs(222)

    assert first["generator"].initial_seed() == 111
    assert second["generator"].initial_seed() == 222
    assert first["worker_init_fn"].base_seed == 111
    assert second["worker_init_fn"].base_seed == 222
