import re
import string
import logging
from typing import Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)

_nltk_initialized = False
_stop_words = None
_lemmatizer = None


def ensure_nltk_resources() -> None:
    """Ensure required NLTK datasets (stopwords, wordnet) are available."""
    global _nltk_initialized, _stop_words, _lemmatizer
    if _nltk_initialized:
        return

    required = ["stopwords", "wordnet"]
    for corpus in required:
        try:
            nltk.data.find(f"corpora/{corpus}")
        except LookupError:
            try:
                logger.info("Downloading missing NLTK corpus '%s'...", corpus)
                nltk.download(corpus, quiet=True)
            except Exception as e:
                logger.warning("Could not download NLTK corpus '%s': %s", corpus, e)

    try:
        _stop_words = set(stopwords.words("english"))
    except Exception:
        _stop_words = set()

    _lemmatizer = WordNetLemmatizer()
    _nltk_initialized = True


def normalize_text(text: Optional[str]) -> str:
    """
    Standard text normalization pipeline for inference:
    1. Lowercase
    2. URL removal
    3. Digits removal
    4. Punctuation removal
    5. Stopwords removal
    6. Lemmatization
    """
    if not text or not isinstance(text, str):
        return ""

    ensure_nltk_resources()

    text = text.lower()
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove digits
    text = "".join([c for c in text if not c.isdigit()])
    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = text.replace("؛", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    words = [w for w in text.split() if w not in (_stop_words or set())]

    # Lemmatize
    lemmatizer = _lemmatizer or WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)
