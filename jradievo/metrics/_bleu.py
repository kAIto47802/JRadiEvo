import MeCab
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


_mecab = MeCab.Tagger("-Owakati")


def _bleu(test: str, ref: str, weights: tuple[float, float, float, float]) -> float:
    reference = _mecab.parse(ref)
    candidate = _mecab.parse(test)

    smooth_fn = SmoothingFunction().method1
    res = sentence_bleu([reference], candidate, weights=weights, smoothing_function=smooth_fn)
    return res


def bleu_1(test: str, ref: str) -> float:
    return _bleu(test, ref, weights=(1, 0, 0, 0))


def bleu_2(test: str, ref: str) -> float:
    return _bleu(test, ref, weights=(0.5, 0.5, 0, 0))


def bleu_3(test: str, ref: str) -> float:
    return _bleu(test, ref, weights=(0.33, 0.33, 0.33, 0))


def bleu_4(test: str, ref: str) -> float:
    return _bleu(test, ref, weights=(0.25, 0.25, 0.25, 0.25))
