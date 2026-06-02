from __future__ import annotations

import gc

import torch

from jradievo.merge._task_vector import TaskVector


@torch.no_grad()
def apply_dare_mask_(
    task_vector: TaskVector,
    mask_rate: float,
) -> None:
    for param in task_vector.params.values():
        torch.manual_seed(42)
        mask = (
            torch.bernoulli(torch.full_like(input=param, fill_value=mask_rate))
            .half()
            .to(param.device)
        )
        param *= 1 - mask
        if mask_rate < 1.0:
            param /= 1 - mask_rate
        del mask
        gc.collect()
        torch.cuda.empty_cache()
        del param
        torch.cuda.empty_cache()
