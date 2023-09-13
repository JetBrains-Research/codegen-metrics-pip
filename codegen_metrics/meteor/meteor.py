import nltk
from nltk.translate.meteor_score import single_meteor_score as meteor

try:
    nltk.data.find("corpora/wordnet.zip")
except LookupError:
    nltk.download("wordnet")
