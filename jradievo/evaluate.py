from argparse import ArgumentParser, Namespace

import numpy as np
import polars as pl

from jradievo.metrics._bleu import bleu_1, bleu_2, bleu_3, bleu_4
from jradievo.metrics._meteor import meteor
from jradievo.metrics._rouge import rouge_l


def main(args: Namespace) -> None:
    df = pl.read_csv(args.input_path)
    reference_texts = df["ja_text"].to_list()
    generated_texts = df[args.pred_col].to_list()

    for metric in [bleu_1, bleu_2, bleu_3, bleu_4, rouge_l, meteor]:
        scores = [
            metric(generated, reference)
            for generated, reference in zip(generated_texts, reference_texts)
        ]
        mean, lower, upper = bootstrap_ci(
            scores,
            n_bootstrap=10000,
            seed=42,
        )
        print(f"{metric.__name__}: {mean:.3f}, 95% CI: [{lower:.3f}, {upper:.3f}]")


def bootstrap_ci(
    scores: list[float] | np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> tuple[float, float, float]:
    scores = np.asarray(scores, dtype=float)
    n = len(scores)

    rng = np.random.default_rng(seed)

    indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = scores[indices].mean(axis=1)

    lower, upper = np.percentile(boot_means, [2.5, 97.5])
    return scores.mean(), lower, upper


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--pred-col", type=str, default="preds")
    args = parser.parse_args()

    main(args)
