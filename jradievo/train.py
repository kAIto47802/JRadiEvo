import argparse
import importlib
import os
from types import SimpleNamespace

from dotenv import load_dotenv
import huggingface_hub
import torch
from transformers import AutoTokenizer

from jradievo.runner.evolve_cpu import run_evolve_cpu
from jradievo.runner.evolve_cpu_offload import run_evolve_cpu_offload
from jradievo.runner.evolve_gpu import run_evolve_gpu
from jradievo.utils._pure import get_metrics, SimpleDataLoder
from jradievo.utils._task_dependents import get_data, get_dataset


def main(cfg: SimpleNamespace) -> None:
    data = get_data(cfg)
    print(data[0] if isinstance(data, tuple) else data)
    dataset = get_dataset(cfg, data)
    dataloader = SimpleDataLoder(dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.vlm.name, trust_remote_code=True)
    metrics = get_metrics(cfg)

    {
        "cpu": run_evolve_cpu,
        "gpu": run_evolve_gpu,
        "cpu_offload": run_evolve_cpu_offload,
    }[cfg.mode](
        cfg,
        tokenizer,
        dataloader,
        metrics,
        device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="final")
    parser.add_argument("--mode", type=str, choices=["cpu", "cpu_offload", "gpu"], default="gpu")
    args = parser.parse_args()
    cfg = importlib.import_module(f"jradievo.config.{args.config}")
    cfg.config_name = args.config  # type: ignore
    cfg.mode = args.mode  # type: ignore
    load_dotenv()
    huggingface_hub.login(os.getenv("HF_TOKEN"))

    main(cfg)  # type: ignore
