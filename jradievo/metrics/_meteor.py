import MeCab
import nltk
from nltk.translate.meteor_score import meteor_score


_mecab = MeCab.Tagger("-Owakati")

try:
    nltk.data.find("wordnet")
except LookupError:
    nltk.download("wordnet")


def meteor(test: str, ref: str) -> float:
    reference = _mecab.parse(ref).split()
    candidate = _mecab.parse(test).split()

    res = meteor_score([reference], candidate)
    return res
