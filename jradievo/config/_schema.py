from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class DAREConfig:
    mask_rate_type: Literal["each", "same"]


@dataclass
class TiesConfig:
    param_mask_type: Literal["each", "same"]
    scaling_coefficient: float | None
    use_each_model_weight: bool


@dataclass
class BunnyConfig:
    text: str
    max_new_tokens: int = 100
    repetition_penalty: float = 1.0  # increase this to avoid chattering


@dataclass
class MiniCPMConfig:
    msgs: list[dict[str, str]]
    sampling: bool = False  # if sampling=False, beam_search will be used by default
    temperature: float = 0.7
    system_prompt: str = ""
    max_new_tokens: int = 128
