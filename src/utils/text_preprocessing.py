import re
import string
from typing import Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.logger import get_logger

logger = get_logger(__name__)

_nltk_initialized = False
_stop_words = None
_lemmatizer = None


def ensure_nltk_resources() -> None:
    """Ensure required NLTK datasets are downloaded and available."""
    global _nltk_initialized, _stop_words, _lemmatizer
    if _nltk_initialized:
        return

    required_corpora = ["stopwords", "wordnet"]
    for corpus in required_corpora:
        try:
            nltk.data.find(f"corpora/{corpus}")
        except LookupError:
            try:
                logger.info("Downloading NLTK resource '%s'...", corpus)
                nltk.download(corpus, quiet=True)
            except Exception as e:
                logger.warning("Could not automatically download NLTK corpus '%s': %s", corpus, e)

    try:
        _stop_words = set(stopwords.words("english"))
    except Exception:
        _stop_words = set()

    _lemmatizer = WordNetLemmatizer()
    _nltk_initialized = True


def get_stop_words() -> set:
    global _stop_words
    if _stop_words is None:
        ensure_nltk_resources()
    return _stop_words or set()


def get_lemmatizer() -> WordNetLemmatizer:
    global _lemmatizer
    if _lemmatizer is None:
        ensure_nltk_resources()
    return _lemmatizer or WordNetLemmatizer()


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.sub("", text)


def remove_numbers(text: str) -> str:
    """Remove numerical digits from text."""
    return "".join([char for char in text if not char.isdigit()])


def remove_punctuations(text: str) -> str:
    """Remove punctuation and normalize whitespace."""
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = text.replace("؛", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str) -> str:
    """Filter out English stopwords."""
    stop_words = get_stop_words()
    return " ".join([word for word in text.split() if word not in stop_words])


def lemmatize_text(text: str) -> str:
    """Lemmatize individual words using WordNet."""
    lemmatizer = get_lemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


def normalize_text(text: Optional[str]) -> str:
    """
    Full text normalization pipeline:
    1. Case normalization (lowercase)
    2. URL removal
    3. Digits removal
    4. Punctuation removal
    5. Stopwords removal
    6. Lemmatization
    """
    if not text or not isinstance(text, str):
        return ""

    text = text.lower()
    text = remove_urls(text)
    text = remove_numbers(text)
    text = remove_punctuations(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text.strip()
