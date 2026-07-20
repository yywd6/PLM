"""Small deterministic seed helpers shared by training and evaluation."""

from dataclasses import dataclass
import random

import numpy as np
import torch


@dataclass(frozen=True)
class WorkerSeedInitializer:
    """Pickle-safe DataLoader worker initializer with auditable seeds."""

    base_seed: int

    def __call__(self, worker_id):
        worker_seed = (int(self.base_seed) + int(worker_id)) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)


def seed_everything(seed, num_workers=0):
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return {
        "actual_seed": seed,
        "python_random_seed": seed,
        "numpy_seed": seed % (2**32),
        "torch_cpu_seed": int(torch.initial_seed()),
        "torch_cuda_seed": seed,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "dataloader_worker_base_seed": seed,
        "dataloader_worker_seeds": [
            (seed + worker_id) % (2**32)
            for worker_id in range(max(0, int(num_workers)))
        ],
    }


def dataloader_seed_kwargs(seed):
    seed = int(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return {
        "generator": generator,
        "worker_init_fn": WorkerSeedInitializer(seed),
    }
