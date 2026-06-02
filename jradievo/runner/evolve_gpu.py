from __future__ import annotations

from collections.abc import Callable
import copy
import gc
import sys
from types import SimpleNamespace

import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedTokenizerFast

from jradievo.merge._dare import apply_dare_mask_
from jradievo.merge._task_vector import TaskVector, to_param_dict
from jradievo.merge._ties import apply_smallest_mask_, get_param_signs
from jradievo.runner.inference import run_inference
from jradievo.utils._pure import (
    copy_params_to_model,
    print_cpu_memory_usage,
    print_gpu_memory_usage,
    print_metrics,
    SimpleDataLoder,
)


def run_evolve_gpu(
    cfg: SimpleNamespace,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    dataloader: SimpleDataLoder,
    metrics: dict[str, Callable[[str, str], float]],
    device: torch.device,
) -> None:
    def objective(trial):
        use_model_names = [*cfg.finetuned_models, "vlm"]
        # [DARE params] -------------------
        if cfg.dare.mask_rate_type == "each":
            mask_rates = {
                model_name: trial.suggest_float(f"dare_{model_name}_mask_rate", 0.0, 1.0)
                for model_name in use_model_names
            }
        elif cfg.dare.mask_rate_type == "same":
            mask_rates = trial.suggest_float("dare_mask_rate", 0.0, 1.0)
        else:
            raise NotImplementedError

        # [TIES params] -------------------
        if cfg.ties.param_mask_type == "each":
            param_value_mask_rate = {
                n: trial.suggest_float(f"ties_mask_rate_{n}", 0.0, 1.0) for n in use_model_names
            }
        elif cfg.ties.param_mask_type == "same":
            param_value_mask_rate = trial.suggest_float("ties_mask_rate", 0.0, 1.0)
        else:
            raise NotImplementedError

        scaling_coefficient = cfg.ties.scaling_coefficient or trial.suggest_float(
            "ties_scaling_coefficient", 0.0, 2.0
        )
        each_model_weight = (
            {n: trial.suggest_float(f"ties_weight_{n}", 0.0, 1.0) for n in use_model_names}
            if cfg.ties.use_each_model_weight
            else None
        )
        # ---------------------------------

        print("[[trial params]]")
        print(trial.params)

        with torch.no_grad():
            print("trial started")
            print_gpu_memory_usage()
            print_cpu_memory_usage()

            base_model = AutoModelForCausalLM.from_pretrained(
                cfg.base_model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            ).eval()
            print("base model loaded")
            print_gpu_memory_usage()
            print_cpu_memory_usage()

            task_vectors = []
            for model_name in use_model_names:
                print(f"[[{model_name}]]")
                print(f"loading {model_name}...")
                print_gpu_memory_usage()
                print_cpu_memory_usage()
                finetuned_model = (
                    AutoModelForCausalLM.from_pretrained(
                        cfg.finetuned_models[model_name] if model_name != "vlm" else cfg.vlm.name,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                    )
                    .eval()
                    .to(device)
                )
                print(f"Creating task vector for {model_name}...")
                task_vector = TaskVector(
                    base_model=copy.deepcopy(base_model).to(device),
                    finetuned_model=finetuned_model,
                    finetuned_param_name_convert_fn=cfg.finetuned_param_name_convert_fns.get(
                        model_name
                    ),
                )
                print_cpu_memory_usage()
                del finetuned_model
                torch.cuda.empty_cache()
                gc.collect()

                print(f"task vectors created: {model_name}")
                print_gpu_memory_usage()
                print_cpu_memory_usage()
                sys.stdout.flush()

                print(f"Masking {model_name}...")

                apply_dare_mask_(
                    task_vector,
                    mask_rate=(
                        mask_rates[model_name] if cfg.dare.mask_rate_type == "each" else mask_rates
                    ),
                )
                print("masking done")
                print_gpu_memory_usage()
                print_cpu_memory_usage()
                gc.collect()
                sys.stdout.flush()

                # >> ties merging ---------------------------------------------------------------------

                print(f"Making flattened model for {model_name}...")
                print_cpu_memory_usage()

                flattened_task_vector = task_vector.to_vector().half()
                print_cpu_memory_usage()
                del task_vector
                torch.cuda.empty_cache()
                gc.collect()
                print("making flattened models done")
                print(f"shape: {flattened_task_vector.shape}")
                print_gpu_memory_usage()
                print_cpu_memory_usage()
                sys.stdout.flush()

                mask_rate = (
                    param_value_mask_rate[model_name]
                    if isinstance(param_value_mask_rate, dict)
                    else param_value_mask_rate
                )
                num_mask_params = int(mask_rate * flattened_task_vector.shape[0])
                apply_smallest_mask_(flattened_task_vector, num_mask_params)
                print("masking smallest magnitude done")
                print_gpu_memory_usage()
                print_cpu_memory_usage()
                sys.stdout.flush()

                task_vectors.append(flattened_task_vector.to("cpu").half())
                print_cpu_memory_usage()
                del flattened_task_vector
                torch.cuda.empty_cache()
                gc.collect()
                print(f"flattened task vector all done: {model_name}")
                print_gpu_memory_usage()
                print_cpu_memory_usage()
                sys.stdout.flush()

            print("!!all task vectors done")
            print_gpu_memory_usage()
            print_cpu_memory_usage()

            param_signs = get_param_signs(task_vectors)
            print("getting param signs done")
            print_gpu_memory_usage()
            print_cpu_memory_usage()
            sys.stdout.flush()

            ## >> >> merge -----------------------------------------------------------------------
            num_preserved = 0.0
            merged_param = 0.0
            merge_models = {name: param for name, param in zip(use_model_names, task_vectors)}
            del task_vectors
            gc.collect()
            param_signs = param_signs.to(device)
            print("making flattened models dict done")
            print_cpu_memory_usage()
            for i, name in enumerate(use_model_names):
                print(f"Merging {name}...")
                print_cpu_memory_usage()
                param = merge_models[name]
                param = param.to(device)
                del merge_models[name]
                mask = ((param_signs > 0) & (param > 0)) | ((param_signs < 0) & (param < 0))
                print_cpu_memory_usage()
                sys.stdout.flush()
                mask = mask.half()
                gc.collect()
                param *= mask
                print_cpu_memory_usage()
                if each_model_weight is not None:
                    param *= each_model_weight[name]
                print("*")
                if isinstance(merged_param, float):
                    merged_param = param.to("cpu")
                else:
                    merged_param += param.to("cpu")
                print_cpu_memory_usage()
                del param
                torch.cuda.empty_cache()
                gc.collect()
                print("**")
                if isinstance(num_preserved, float):
                    num_preserved = mask.to("cpu")
                else:
                    num_preserved += mask.to("cpu")
                print("***")
                print_cpu_memory_usage()
                del mask
                torch.cuda.empty_cache()
                gc.collect()
                print_cpu_memory_usage()
                sys.stdout.flush()
            del param_signs
            del merge_models
            gc.collect()
            print("merging done")
            print_cpu_memory_usage()
            sys.stdout.flush()
            assert isinstance(merged_param, torch.Tensor)
            assert isinstance(num_preserved, torch.Tensor)
            merged_param = merged_param.to(device)
            merged_param /= torch.clamp(num_preserved.to(device), min=1.0)
            del num_preserved
            torch.cuda.empty_cache()
            gc.collect()
            print("rescaling done")
            print_gpu_memory_usage()
            print_cpu_memory_usage()
            sys.stdout.flush()
            ## << << merge -----------------------------------------------------------------------
            base_model = (
                AutoModelForCausalLM.from_pretrained(
                    cfg.base_model_name,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )
                .eval()
                .to(device)
            )
            print("base model loaded")
            print_cpu_memory_usage()

            merged_task_vector_param_dict = to_param_dict(
                param=merged_param,
                base_model=copy.deepcopy(base_model),
            )
            print_gpu_memory_usage()
            print_cpu_memory_usage()
            del merged_param
            gc.collect()
            print("making merged task vector param dict done")
            print_gpu_memory_usage()
            print_cpu_memory_usage()
            merged_task_vector = TaskVector.from_param_dict(
                merged_task_vector_param_dict,
                base_model,
            )
            del merged_task_vector_param_dict
            torch.cuda.empty_cache()
            gc.collect()
            print("making merged task vector done")
            print_cpu_memory_usage()

            merged_params = merged_task_vector.get_merged_params(
                base_model=base_model,
                coef=scaling_coefficient,
            )
            print("combining with pretrained model done")
            print_gpu_memory_usage()
            print_cpu_memory_usage()
            del merged_task_vector.params
            del merged_task_vector
            del base_model
            torch.cuda.empty_cache()
            gc.collect()
            print("merging with pretrained model done")
            print_gpu_memory_usage()
            print_cpu_memory_usage()
            sys.stdout.flush()
            ## << ties merging ---------------------------------------------------------------------

            vlm = (
                AutoModelForCausalLM.from_pretrained(
                    cfg.vlm.name,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )
                .eval()
                .to(device)
            )
            print("vlm loaded")
            print_gpu_memory_usage()
            print_cpu_memory_usage()
            copy_params_to_model(vlm, merged_params)
            print("copying params to model done")
            print_gpu_memory_usage()
            print_cpu_memory_usage()
            del merged_params
            torch.cuda.empty_cache()
            gc.collect()
            print_gpu_memory_usage()
            print_cpu_memory_usage()

            print("running inference...")
            val_metrics = run_inference(cfg, vlm, tokenizer, dataloader, metrics, device)
            print_metrics(val_metrics)

            del vlm
            torch.cuda.empty_cache()
            gc.collect()
            print("trial finished")
            print_gpu_memory_usage()
            print_cpu_memory_usage()
            sys.stdout.flush()

            return (
                tuple([val_metrics[met] for met in cfg.use_metrics])
                if len(cfg.use_metrics) > 1
                else val_metrics[cfg.use_metrics[0]]
            )

    study = optuna.create_study(
        study_name=cfg.config_name,
        storage=f"sqlite:///outputs/{cfg.config_name}.db",
        sampler={
            "cmaes": optuna.samplers.CmaEsSampler,
            "tpe": optuna.samplers.TPESampler,
            "gp": optuna.samplers.GPSampler,
        }[cfg.sampler](),
        direction=cfg.direction[0] if len(cfg.use_metrics) == 1 else None,
        directions=cfg.direction if len(cfg.use_metrics) > 1 else None,
        load_if_exists=True,
    )
    study.optimize(
        objective,
        n_trials=cfg.n_trials,
        callbacks=[MaxTrialsCallback(cfg.n_trials, states=(TrialState.COMPLETE,))],
    )
