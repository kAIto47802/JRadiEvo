from __future__ import annotations

import gc
import sys

import torch

from jradievo.utils._pure import print_cpu_memory_usage


def apply_magnitude_mask_(
    params: torch.Tensor,
    model_names: list[str],
    mask_rates: float | dict[str, float],
) -> None:
    print(">masking smallest magnitude")
    print(f"device: {params.device}")
    print_cpu_memory_usage()
    sys.stdout.flush()

    for p, name in zip(params, model_names):
        mask_rate = mask_rates[name] if isinstance(mask_rates, dict) else mask_rates
        num_mask_params = int(params.shape[1] * mask_rate)

        apply_smallest_mask_(p, num_mask_params)


def apply_smallest_mask_(param: torch.Tensor, num_mask_params: int):
    print(">>masking smallest magnitude...")
    print_cpu_memory_usage()
    param_abs = param.abs()
    sys.stdout.flush()
    # kth_value = param_abs.kthvalue(k=num_mask_params, dim=0)[0]
    assert (param_abs >= 64.0).sum(dtype=torch.float32) == 0
    gc.collect()
    left = 0.0
    right = 64.0
    for _ in range(22):
        mid = (left + right) / 2
        if (param_abs < mid).sum(dtype=torch.float32) < num_mask_params:
            left = mid
        else:
            right = mid
        gc.collect()
        torch.cuda.empty_cache()
    kth_value = left

    print_cpu_memory_usage()
    sys.stdout.flush()
    mask = param_abs >= kth_value
    print_cpu_memory_usage()
    sys.stdout.flush()
    del param_abs
    gc.collect()
    print_cpu_memory_usage()
    sys.stdout.flush()
    param *= mask
    print_cpu_memory_usage()
    sys.stdout.flush()
    del param
    del mask
    gc.collect()
    print_cpu_memory_usage()
    sys.stdout.flush()


def get_param_signs(
    params: torch.Tensor | list[torch.Tensor],
) -> torch.Tensor:
    print(">getting param signs...")
    s = 0.0
    for param in params:
        s += param
        del param
        torch.cuda.empty_cache()
        gc.collect()
    assert isinstance(s, torch.Tensor)
    print_cpu_memory_usage()
    pos = (s > 0.0).half()
    neg = (s < 0.0).half()
    param_signs = pos - neg
    print_cpu_memory_usage()
    del s
    del pos
    del neg
    torch.cuda.empty_cache()
    gc.collect()
    print(">getting param signs done")
    print_cpu_memory_usage()
    majority_sign = torch.sign(param_signs.sum(dim=0))
    majority_sign = (param_signs == 0).half() * majority_sign
    param_signs += majority_sign
    print(">getting majority sign done")
    print_cpu_memory_usage()
    del majority_sign
    del params
    torch.cuda.empty_cache()
    gc.collect()
    return param_signs
