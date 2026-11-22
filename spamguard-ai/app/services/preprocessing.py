"""Consistent text normalisation shared by training and inference.

Design goals
------------
* Remove noise (HTML, punctuation, whitespace) without deleting signals that
  matter for fraud detection.
* Keep the exact same transformation in the scikit-learn pipeline used at
  training time, so inference can never drift from training.
* Stay conservative: optional stopword removal is available but disabled for
  the trained models, because short words such as "now", "not", "you" and
  "free" carry meaning in spam messages.

Normalisation steps (applied in order):
1. Decode HTML entities and strip HTML tags.
2. Unicode NFKC normalisation and lower-casing.
3. Replace URLs with a ``weburl`` token plus cleaned hostname tokens.
4. Replace e-mail addresses with an ``emailaddr`` token.
5. Convert currency symbols to a ``currency`` token (the amount is retained).
6. Collapse runs of three or more repeated characters (e.g. "freeeee").
7. Append signal tokens for heavy exclamation use / abnormal upper-case use.
8. Remove all remaining punctuation and collapse whitespace.
9. Optionally remove English stopwords using NLTK tokenisation.
"""
from __future__ import annotations

import html
import re
import unicodedata
from urllib.parse import urlsplit

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

PREPROCESSING_VERSION = "1.0.0"

URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
HTML_TAG_RE = re.compile(r"<[^>]{1,500}>")
REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")
WHITESPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")

CURRENCY_SYMBOLS = ("₹", "$", "£", "€", "¥", "₦", "₽")
COMMON_TLDS = {
    "com", "net", "org", "in", "co", "uk", "us", "io", "info", "biz", "edu",
    "gov", "me", "ly", "ru", "cn", "de", "fr", "au", "ca", "xyz", "top",
}

# Fallback stopword list (adapted from the NLTK English stopword list). Using
# a fixed fallback keeps behaviour reproducible when NLTK corpora are absent.
_FALLBACK_STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are aren't as at be
    because been before being below between both but by can't cannot could
    couldn't did didn't do does doesn't doing don't down during each few for
    from further had hadn't has hasn't have haven't having he he'd he'll he's
    her here here's hers herself him himself his how how's i i'd i'll i'm i've
    if in into is isn't it it's its itself let's me more most mustn't my myself
    no nor not of off on once only or other ought our ours ourselves out over
    own same shan't she she'd she'll she's should shouldn't so some such than
    that that's the their theirs them themselves then there there's these they
    they'd they'll they're they've this those through to too under until up
    very was wasn't we we'd we'll we're we've were weren't what what's when
    when's where where's which while who who's whom why why's with won't would
    wouldn't you you'd you'll you're you've your yours yourself yourselves
    """.split()
)

_NLTK_STOPWORDS: frozenset[str] | None = None


def _get_stopwords() -> frozenset[str]:
    """Use NLTK's corpus when installed, otherwise deterministic fallback."""
    global _NLTK_STOPWORDS
    if _NLTK_STOPWORDS is None:
        try:  # pragma: no cover - depends on local NLTK data availability
            from nltk.corpus import stopwords

            _NLTK_STOPWORDS = frozenset(stopwords.words("english"))
        except Exception:
            _NLTK_STOPWORDS = _FALLBACK_STOPWORDS
    return _NLTK_STOPWORDS


def _replace_url(match: re.Match) -> str:
    """Return a placeholder plus useful hostname tokens.

    The URL is not followed or fetched. Hostname parts are included as plain
    tokens so a model can learn domain-level patterns without keeping the
    full, potentially unique URL.
    """
    raw = match.group(0).rstrip(".,;:!?)\"'")
    candidate = raw if re.match(r"^https?://", raw, re.I) else f"http://{raw}"
    try:
        hostname = (urlsplit(candidate).hostname or "").lower()
    except ValueError:
        hostname = ""
    parts = [
        part
        for part in re.split(r"[.\-]", hostname)
        if part and part not in {"www", "http", "https"} and part not in COMMON_TLDS
    ]
    token = "weburl"
    if parts:
        token += " " + " ".join(parts[:4])
    return f" {token} "


def normalize_text(text: str, remove_stopwords: bool = False) -> str:
    """Normalise a message for TF-IDF feature extraction.

    Parameters
    ----------
    text:
        Raw message. Non-string inputs are treated as empty.
    remove_stopwords:
        If True, filters English stopwords after NLTK tokenisation. Disabled
        by default for the trained models (see module docstring).
    """
    if not isinstance(text, str):
        return ""

    original = text
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = unicodedata.normalize("NFKC", text).lower()

    # Signal extraction happens before punctuation removal.
    exclamations = text.count("!")
    letters = [ch for ch in original if ch.isalpha()]
    uppercase_ratio = (
        sum(1 for ch in letters if ch.isupper()) / len(letters) if letters else 0.0
    )

    text = URL_RE.sub(_replace_url, text)
    text = EMAIL_RE.sub(" emailaddr ", text)

    currency_found = any(symbol in text for symbol in CURRENCY_SYMBOLS)
    for symbol in CURRENCY_SYMBOLS:
        text = text.replace(symbol, " currency ")
    # Common textual currency abbreviations are preserved as a signal too.
    text = re.sub(r"\b(?:inr|usd|gbp|eur)\b", " currency ", text)

    text = REPEATED_CHAR_RE.sub(r"\1\1", text)
    text = NON_ALNUM_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()

    signal_tokens: list[str] = []
    if exclamations >= 3:
        signal_tokens.append("manyexclamations")
    if len(letters) >= 12 and uppercase_ratio > 0.55:
        signal_tokens.append("allcaps")
    if currency_found and "currency" not in text:
        signal_tokens.append("currency")

    if signal_tokens:
        text = f"{text} {' '.join(signal_tokens)}".strip()

    if remove_stopwords and text:
        tokens = tokenize(text)
        stopwords_set = _get_stopwords()
        text = " ".join(token for token in tokens if token not in stopwords_set)

    return WHITESPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Tokenise with NLTK's rule-based TweetTokenizer (no corpus download)."""
    try:
        from nltk.tokenize import TweetTokenizer

        return TweetTokenizer(
            preserve_case=False,
            reduce_len=False,
            strip_handles=False,
        ).tokenize(text)
    except Exception:  # pragma: no cover - NLTK is a declared dependency
        return re.findall(r"[a-z0-9]+", text.lower())


def message_stats(text: str) -> dict:
    """Lightweight descriptive statistics used by the API/UI."""
    tokens = tokenize(text) if text else []
    return {
        "characters": len(text),
        "tokens": len(tokens),
        "unique_tokens": len(set(tokens)),
        "urls": len(URL_RE.findall(text or "")),
        "uppercase_ratio": round(
            (
                sum(1 for ch in text if ch.isupper())
                / sum(1 for ch in text if ch.isalpha())
            )
            if any(ch.isalpha() for ch in text)
            else 0.0,
            3,
        ),
    }


class TextNormalizer(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer wrapping :func:`normalize_text`.

    Stored inside the joblib pipeline so the exact same normalisation is
    applied during training and inference.
    """

    def __init__(self, remove_stopwords: bool = False):
        self.remove_stopwords = remove_stopwords

    def fit(self, X, y=None):  # noqa: N803 - sklearn naming convention
        return self

    def transform(self, X):  # noqa: N803 - sklearn naming convention
        return np.asarray(
            [
                normalize_text(item, remove_stopwords=self.remove_stopwords)
                for item in X
            ],
            dtype=object,
        )

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)
