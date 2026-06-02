from __future__ import annotations

from collections.abc import Callable
import gc
import sys

import torch
import torch.nn as nn


if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


class TaskVector:
    def __init__(
        self,
        base_model: nn.Module | None = None,
        finetuned_model: nn.Module | None = None,
        finetuned_param_name_convert_fn: Callable[[str], str] | None = None,
    ) -> None:
        if base_model is None or finetuned_model is None:
            return
        pretrained_param_dict = {
            param_name: param_value for param_name, param_value in base_model.named_parameters()
        }
        self.base_model_param_names = [name for name, _ in base_model.named_parameters()]
        finetuned_param_dict = {
            (finetuned_param_name_convert_fn or (lambda s: s))(param_name): param_value
            for param_name, param_value in finetuned_model.named_parameters()
        }
        self.params = {}
        with torch.no_grad():
            for param_name in pretrained_param_dict:
                if param_name not in finetuned_param_dict:
                    print(f"param_name {param_name} is not contained in finetuned model!")
                    self.params[param_name] = torch.zeros_like(
                        input=pretrained_param_dict[param_name],
                        dtype=torch.float16,
                    )
                else:
                    self.params[param_name] = (
                        finetuned_param_dict[param_name] - pretrained_param_dict[param_name]
                    )

    @classmethod
    def from_param_dict(cls, params: dict[str, torch.Tensor], base_model: nn.Module) -> Self:
        instance = cls()
        instance.params = params
        instance.base_model_param_names = [name for name, _ in base_model.named_parameters()]
        return instance

    @torch.no_grad()
    def get_merged_params(
        self, base_model: nn.Module, coef: float = 1.0
    ) -> dict[str, torch.Tensor]:
        base_model_params = {
            param_name: param_value for param_name, param_value in base_model.named_parameters()
        }
        return {name: base_model_params[name] + coef * self.params[name] for name in self.params}

    def to_vector(self) -> torch.Tensor:
        return nn.utils.parameters_to_vector(
            [self.params[name].flatten() for name in self.base_model_param_names]
        )


def to_param_dict(param: torch.Tensor, base_model: nn.Module) -> dict[str, torch.Tensor]:
    base_model_parameters = base_model.parameters()
    nn.utils.vector_to_parameters(param, base_model_parameters)
    param_dic = {name: param for name, param in base_model.named_parameters()}

    del base_model_parameters
    del base_model
    del param
    torch.cuda.empty_cache()
    gc.collect()

    return param_dic
