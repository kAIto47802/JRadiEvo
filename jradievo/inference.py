import argparse
import importlib
import os
from types import SimpleNamespace

from dotenv import load_dotenv
import huggingface_hub
import polars as pl
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jradievo.runner.inference import run_inference
from jradievo.utils._pure import get_metrics, SimpleDataLoder
from jradievo.utils._task_dependents import get_dataset


def main(cfg: SimpleNamespace) -> None:
    data = pl.read_csv(cfg.test_data_path)
    print(data[0] if isinstance(data, tuple) else data)
    dataset = get_dataset(cfg, data)
    dataloader = SimpleDataLoder(dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.vlm.name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, trust_remote_code=True).to(device)
    metrics = get_metrics(cfg)

    run_inference(
        cfg,
        model,
        tokenizer,
        dataloader,
        metrics,
        device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="final")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    args = parser.parse_args()
    cfg = importlib.import_module(f"jradievo.config.{args.config}")
    cfg.config_name = args.config  # type: ignore
    cfg.test_data_path = args.data_path  # type: ignore
    cfg.model_path = args.model_path  # type: ignore
    load_dotenv()
    huggingface_hub.login(os.getenv("HF_TOKEN"))

    main(cfg)  # type: ignore
