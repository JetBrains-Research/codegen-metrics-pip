from rouge_score import rouge_scorer

from .codebleu.codebleu import codebleu as codebleu_impl
from .meteor.meteor import meteor as meteor_impl
from .ruby.similarity import ruby as ruby_impl
from .ruby.util import tokenize_tranx
from .sacrebleu_code.sacrebleu_methods.compat import sentence_bleu, sentence_chrf


def rougel(snippet: str, hypothesis: str) -> float:
    return rouge_scorer._score_lcs(tokenize_tranx(snippet), tokenize_tranx(hypothesis)).fmeasure


def codebleu(snippet: str, hypothesis: str, weights: tuple[float] = (0.1, 0.1, 0.4, 0.4)) -> float:
    return codebleu_impl(snippet, hypothesis, weights=weights)


def ruby(snippet: str, hypothesis: str) -> float:
    return ruby_impl(hypothesis, snippet)[0]


def meteor(snippet: str, hypothesis: str) -> float:
    return meteor_impl(tokenize_tranx(snippet), tokenize_tranx(hypothesis))


# chrF
def chrf(snippet: str, hypothesis: str) -> float:
    return sentence_chrf(hypothesis, [snippet]).score / 100


# BLEU
def bleu(snippet: str, hypothesis: str) -> float:
    return sentence_bleu(hypothesis, [snippet]).score / 100
