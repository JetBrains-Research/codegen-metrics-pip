from typing import Sequence

from .metrics.bleu import BLEU, BLEUScore
from .metrics.chrf import CHRF, CHRFScore


def sentence_bleu(
    hypothesis: str,
    references: Sequence[str],
    smooth_method: str = "exp",
    smooth_value: float = None,
    lowercase: bool = False,
    tokenize=BLEU.TOKENIZER_DEFAULT,
    use_effective_order: bool = True,
) -> BLEUScore:
    """
    Computes BLEU for a single sentence against a single (or multiple) reference(s).

    Disclaimer: Computing BLEU at the sentence level is not its intended use as
    BLEU is a corpus-level metric.

    :param hypothesis: A single hypothesis string.
    :param references: A sequence of reference strings.
    :param smooth_method: The smoothing method to use ('floor', 'add-k', 'exp' or 'none')
    :param smooth_value: The smoothing value for `floor` and `add-k` methods. `None` falls back to default value.
    :param lowercase: Lowercase the data
    :param tokenize: The tokenizer to use
    :param use_effective_order: Don't take into account n-gram orders without any match.
    :return: Returns a `BLEUScore` object.
    """
    metric = BLEU(
        lowercase=lowercase,
        tokenize=tokenize,
        force=False,
        smooth_method=smooth_method,
        smooth_value=smooth_value,
        effective_order=use_effective_order,
    )

    return metric.sentence_score(hypothesis, references)


def sentence_chrf(
    hypothesis: str,
    references: Sequence[str],
    char_order: int = CHRF.CHAR_ORDER,
    word_order: int = CHRF.WORD_ORDER,
    beta: int = CHRF.BETA,
    remove_whitespace: bool = True,
    eps_smoothing: bool = False,
) -> CHRFScore:
    """
    Computes chrF for a single sentence against a single (or multiple) reference(s).
    If `word_order` equals to 2, the metric is referred to as chrF++.

    :param hypothesis: A single hypothesis string.
    :param references: A sequence of reference strings.
    :param char_order: Character n-gram order.
    :param word_order: Word n-gram order. If equals to 2, the metric is referred to as chrF++.
    :param beta: Determine the importance of recall w.r.t precision.
    :param eps_smoothing: If `True`, applies epsilon smoothing similar
    to reference chrF++.py, NLTK and Moses implementations. Otherwise,
    it takes into account effective match order similar to sacreBLEU < 2.0.0.
    :param remove_whitespace: If `True`, removes whitespaces prior to character n-gram extraction.
    :return: A `CHRFScore` object.
    """
    metric = CHRF(
        char_order=char_order,
        word_order=word_order,
        beta=beta,
        whitespace=not remove_whitespace,
        eps_smoothing=eps_smoothing,
    )
    return metric.sentence_score(hypothesis, references)
