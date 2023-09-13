from rouge_score import rouge_scorer

from .codebleu.codebleu import codebleu as codebleu_impl
from .meteor.meteor import meteor as meteor_impl
from .ruby.similarity import ruby as ruby_impl
from .ruby.util import tokenize_tranx
from .sacrebleu_code.sacrebleu_methods.compat import sentence_bleu, sentence_chrf


def rougel(reference_code: str, generated_code: str) -> float:
    return rouge_scorer._score_lcs(tokenize_tranx(reference_code), tokenize_tranx(generated_code)).fmeasure


def codebleu(reference_code: str, generated_code: str, weights: tuple[float] = (0.1, 0.1, 0.4, 0.4)) -> float:
    return codebleu_impl(reference_code, generated_code, weights=weights)


def ruby(reference_code: str, generated_code: str) -> float:
    return ruby_impl(generated_code, reference_code)[0]


def meteor(reference_code: str, generated_code: str) -> float:
    return meteor_impl(tokenize_tranx(reference_code), tokenize_tranx(generated_code))


# chrF
def chrf(reference_code: str, generated_code: str) -> float:
    return sentence_chrf(generated_code, [reference_code]).score / 100


# BLEU
def bleu(reference_code: str, generated_code: str) -> float:
    return sentence_bleu(generated_code, [reference_code]).score / 100
