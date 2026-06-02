from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from jradievo.utils._pure import generate_response, SimpleDataLoder


def run_inference(
    cfg: SimpleNamespace,
    model: nn.Module,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    dataloader: SimpleDataLoder,
    metrics: dict[str, Callable[[str, str], float]],
    device: torch.device,
) -> dict[str, float]:
    res = {k: 0.0 for k in metrics}
    skipped = 0
    for i, (text, image, dicom_id) in enumerate(dataloader):
        if not text:
            skipped += 1
            continue
        response = generate_response(cfg, model, tokenizer, image, device)
        if cfg.print_text:
            print(f"[Dicom ID {i}]\n{dicom_id}")
            print(f"[Text {i}]\n{text}")
            print(f"[Generated {i}]\n{response}")
        for metric_name, metric_fn in metrics.items():
            metric_value = metric_fn(response, text)
            res[metric_name] += metric_value
            if cfg.print_text:
                print(f"{metric_name}: {metric_value:.4f}")
    res = {k: v / (len(dataloader) - skipped) for k, v in res.items()}
    return res
