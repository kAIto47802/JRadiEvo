from __future__ import annotations

import MeCab
from rouge_score.rouge_scorer import RougeScorer
from rouge_score.tokenizers import Tokenizer


class _MeCabTokenizer(Tokenizer):
    def __init__(self, use_stemmer=False) -> None:
        self._tagger = MeCab.Tagger("-Owakati")

    def tokenize(self, text) -> list[str]:
        return self._tagger.parse(text).split()


def rouge_l(test: str, ref: str) -> float:
    mecab_tokenizer = _MeCabTokenizer()
    scorer = RougeScorer(["rougeL"], use_stemmer=False, tokenizer=mecab_tokenizer)
    return scorer.score(ref, test)["rougeL"].fmeasure
