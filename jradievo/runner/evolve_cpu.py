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
from jradievo.merge._ties import apply_magnitude_mask_, get_param_signs
from jradievo.runner.inference import run_inference
from jradievo.utils._pure import (
    copy_params_to_model,
    print_cpu_memory_usage,
    print_gpu_memory_usage,
    print_metrics,
    SimpleDataLoder,
)


def run_evolve_cpu(
    cfg: SimpleNamespace,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    dataloader: SimpleDataLoder,
    metrics: dict[str, Callable[[str, str], float]],
    device: torch.device,
) -> None:
    def objective(trial):
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
            print_cpu_memory_usage()
            use_model_names = [*cfg.finetuned_models, "vlm"]

            if cfg.dare.mask_rate_type == "each":
                mask_rates = {
                    name: trial.suggest_float(f"dare_{name}_mask_rate", 0.0, 1.0)
                    for name in use_model_names
                }
            elif cfg.dare.mask_rate_type == "same":
                mask_rates = trial.suggest_float("dare_mask_rate", 0.0, 1.0)
            else:
                raise NotImplementedError

            task_vectors = {}
            for model_name in use_model_names:
                print(f"loading {model_name}...")
                print_cpu_memory_usage()
                finetuned_model = AutoModelForCausalLM.from_pretrained(
                    cfg.finetuned_models[model_name] if model_name != "vlm" else cfg.vlm.name,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    # device_map="auto",
                ).eval()
                sys.stdout.flush()
                print(f"Creating task vector for {model_name}...")
                print_cpu_memory_usage()
                task_vectors[model_name] = TaskVector(
                    base_model=base_model,
                    finetuned_model=finetuned_model,
                    finetuned_param_name_convert_fn=cfg.finetuned_param_name_convert_fns.get(
                        model_name
                    ),
                )
                print_cpu_memory_usage()
                del finetuned_model
                gc.collect()
                print_cpu_memory_usage()
                sys.stdout.flush()
            del base_model
            gc.collect()

            print("task vectors created")
            print_cpu_memory_usage()

            for finetuned_model_name, task_vector in task_vectors.items():
                print(f"Masking {finetuned_model_name}...")
                print_cpu_memory_usage()
                apply_dare_mask_(
                    task_vector,
                    mask_rate=(
                        mask_rates[finetuned_model_name]
                        if cfg.dare.mask_rate_type == "each"
                        else mask_rates
                    ),
                )
                print_cpu_memory_usage()
                del task_vector
                gc.collect()
                sys.stdout.flush()

            print("masking done")
            print_cpu_memory_usage()

            # >> ties merging ---------------------------------------------------------------------
            if cfg.ties.param_mask_type == "each":
                param_value_mask_rate = {
                    n: trial.suggest_float(f"ties_mask_rate_{n}", 0.0, 1.0)
                    for n in use_model_names
                }
            elif cfg.ties.param_mask_type == "same":
                param_value_mask_rate = trial.suggest_float("ties_mask_rate", 0.0, 1.0)
            else:
                raise NotImplementedError

            scaling_coefficient = cfg.ties.scaling_coefficient or trial.suggest_float(
                "ties_scaling_coefficient", 0.0, 2.0
            )
            each_model_weight = (
                {n: trial.suggest_float(f"ties_weight_{n}", 0.0, 2.0) for n in use_model_names}
                if cfg.ties.use_each_model_weight
                else None
            )

            merge_model_ls = []
            for name in use_model_names:
                print(f"Making flattened model for {name}...")
                print_cpu_memory_usage()
                merge_model_ls.append(task_vectors[name].to_vector())
                print_cpu_memory_usage()
                # del task_vectors[name].task_vector_param_dict
                del task_vectors[name]
                # torch.cuda.empty_cache()
                gc.collect()
                print_cpu_memory_usage()
            print("making flattened models done")
            # print_gpu_memory_usage()
            print_cpu_memory_usage()
            sys.stdout.flush()

            merge_models = torch.vstack(merge_model_ls)
            del merge_model_ls

            apply_magnitude_mask_(
                merge_models,
                use_model_names,
                param_value_mask_rate,
            )
            print("masking smallest magnitude done")
            print_cpu_memory_usage()
            sys.stdout.flush()

            param_signs = get_param_signs(merge_models)
            print("getting param signs done")
            print_cpu_memory_usage()
            sys.stdout.flush()

            print("[[trial params]]")
            print(trial.params)

            ## >> >> merge -----------------------------------------------------------------------
            num_preserved = 0.0
            merged_param = 0.0
            merge_models_dic = {name: param for name, param in zip(use_model_names, merge_models)}
            del merge_models
            gc.collect()
            print("making flattened models dict done")
            print_cpu_memory_usage()
            for i, name in enumerate(use_model_names):
                print(f"Merging {name}...")
                # print_gpu_memory_usage()
                print_cpu_memory_usage()
                param = merge_models_dic[name]
                del merge_models_dic[name]
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
                    merged_param = param
                else:
                    merged_param += param
                print_cpu_memory_usage()
                del param
                # torch.cuda.empty_cache()
                gc.collect()
                print("**")
                if isinstance(num_preserved, float):
                    num_preserved = mask
                else:
                    num_preserved += mask
                print("***")
                print_cpu_memory_usage()
                del mask
                # torch.cuda.empty_cache()
                gc.collect()
                print_cpu_memory_usage()
                sys.stdout.flush()
            del param_signs
            del merge_models_dic
            # torch.cuda.empty_cache()
            gc.collect()
            print("merging done")
            # print_gpu_memory_usage()
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
            "nsga": optuna.samplers.NSGAIISampler,
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
