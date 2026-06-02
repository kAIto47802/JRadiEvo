from __future__ import annotations

from collections.abc import Callable
import os
from types import SimpleNamespace

import numpy as np
from PIL.Image import Image
import psutil
import torch
import torch.nn as nn

from jradievo.metrics._bleu import bleu_1
from jradievo.metrics._rouge import rouge_l_ja


def copy_params_to_model(model: nn.Module, params: dict[str, torch.Tensor]) -> None:
    for param_name, param_value in model.named_parameters():
        if param_name in params:
            param_value.data.copy_(params[param_name])


def generate_response(
    cfg: SimpleNamespace,
    model: nn.Module,
    tokenizer,
    image: Image,
    device: torch.device,
) -> str:
    if cfg.vlm.name == "BAAI/Bunny-v1_1-Llama-3-8B-V" or "bunny" in cfg.vlm.name:
        image = image.convert("RGB")
        text_chunks = [tokenizer(chunk).input_ids for chunk in cfg.vlm.bunny.text.split("<image>")]
        input_ids = (
            torch.tensor(text_chunks[0] + [-200] + text_chunks[1][1:], dtype=torch.long)
            .unsqueeze(0)
            .to(device)
        )
        image_tensor = model.process_images([image], model.config).to(
            dtype=model.dtype, device=device
        )
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            max_new_tokens=cfg.vlm.bunny.max_new_tokens,
            use_cache=True,
            repetition_penalty=cfg.vlm.bunny.repetition_penalty,
        )[0]
        return tokenizer.decode(output_ids[input_ids.shape[1] :], skip_special_tokens=True).strip()
    elif cfg.vlm.name == "openbmb/MiniCPM-Llama3-V-2_5":
        image = image.convert("RGB")
        res = model.chat(
            image=image,
            msgs=cfg.vlm.minicpm.msgs,
            tokenizer=tokenizer,
            sampling=cfg.vlm.minicpm.sampling,
            temperature=cfg.vlm.minicpm.temperature,
            system_prompt=cfg.vlm.minicpm.system_prompt,
            max_new_tokens=cfg.vlm.minicpm.max_new_tokens,
        )
        return res
    else:
        raise ValueError(f"Unsupported model name: {cfg.vlm.name}")


class SimpleDataLoder:
    def __init__(
        self, dataset: torch.utils.data.Dataset, indices: np.ndarray | None = None
    ) -> None:
        self.dataset = dataset
        self.indices = np.random.permutation(len(dataset)) if indices is None else indices

    def __iter__(self):
        for index in self.indices:
            yield self.dataset[index]

    def __len__(self):
        return len(self.indices)


def get_metrics(cfg: SimpleNamespace) -> dict[str, Callable[[str, str], float]]:
    res = {}
    for name in cfg.metrics:
        res[name] = {
            "rouge_l_ja": rouge_l_ja,
            "bleu_1": bleu_1,
        }[name]
    return res


def print_metrics(metrics: dict[str, float]) -> None:
    print(", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))


def print_gpu_memory_usage():
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
    free_memory = total_memory - reserved

    print(f"  |Allocated GPU memory: {allocated:.2f} MB")
    print(f"  |Reserved GPU memory: {reserved:.2f} MB")
    print(f"  |Free GPU memory: {free_memory:.2f} MB")


def print_cpu_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"  |CPU memory usage: {process.memory_info().rss / 1024**2:.2f} MB")
