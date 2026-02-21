"""
preprocessing.py — Módulo compartilhado de pré-processamento
Liga Acadêmica de IA (LIGIA/UFPE) 2026 — Detecção de Desinformação Digital

Importado por: notebook_01_preprocessing.ipynb, notebook_03_inference.ipynb

Usage:
    from src.preprocessing import (
        clean_text, add_style_features, lemmatize_series,
        STYLE_COLS, SENSATIONAL_TERMS, REUTERS_SUBJECTS
    )
"""
import re
import time

import numpy as np
import pandas as pd

# ── Constantes ──────────────────────────────────────────────────────────────
REUTERS_SUBJECTS = frozenset({"politicsNews", "worldnews"})

SENSATIONAL_TERMS = (
    "shocking", "unbelievable", "amazing", "incredible",
    "must see", "breaking", "exclusive", "urgent",
)

STYLE_COLS = [
    "caps_ratio", "exclamation_count", "title_len",
    "text_len", "word_count", "avg_word_len",
    "sentence_count", "avg_sentence_len",
    "question_count", "quote_count", "ellipsis_count",
    "all_caps_words", "title_caps_ratio",
    "unique_word_ratio", "sensational_count",
]

# ── Padrões de remoção de data leakage ───────────────────────────────────────
_RE_LOCATION_AGENCY   = re.compile(
    r"^[A-Z][A-Z\s/,\.]+\s*\([A-Za-z\s]+\)\s*[-\u2013\u2014]?\s*", re.MULTILINE
)
_RE_AGENCY_TAG        = re.compile(r"\(\s*(?:Reuters|AP|AFP)\s*\)", re.IGNORECASE)
_RE_FAKE_SOURCES      = re.compile(
    r"\b(?:21st\s*Century\s*Wire|YourNewsWire|Infowars|Breitbart|RT\.com|NaturalNews|BeforeItsNews)\b",
    re.IGNORECASE,
)
_RE_URL               = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_MENTION           = re.compile(r"@\w+")
_RE_LEAKAGE_ARTIFACTS = re.compile(
    r"\b(video|image|featured|getty(\s+images)?|bit\.ly|pic\.twitter\.com|reuters)\b",
    re.IGNORECASE | re.MULTILINE,
)
_SENSATIONAL_RE       = re.compile(
    r"\b(" + "|".join(SENSATIONAL_TERMS) + r")\b", re.IGNORECASE
)

_CLEAN_PATTERNS = (
    _RE_LOCATION_AGENCY, _RE_AGENCY_TAG, _RE_LEAKAGE_ARTIFACTS,
    _RE_FAKE_SOURCES, _RE_URL, _RE_MENTION,
)


# ── Funções públicas ─────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove artefatos de data leakage do texto bruto."""
    for pattern in _CLEAN_PATTERNS:
        text = pattern.sub("", str(text))
    return re.sub(r"\s{2,}", " ", text).strip()


def _caps_ratio(text: str) -> float:
    alpha = [c for c in str(text) if c.isalpha()]
    return sum(c.isupper() for c in alpha) / len(alpha) if alpha else 0.0


def _avg_word_len(text: str) -> float:
    words = str(text).split()
    return float(np.mean([len(w) for w in words])) if words else 0.0


def _unique_word_ratio(text: str) -> float:
    words = str(text).split()
    return len(set(words)) / len(words) if words else 0.0


def _quote_count(text: str) -> int:
    return sum(1 for c in str(text) if c in ('"', "'", "\u2018", "\u2019", "\u201c", "\u201d"))


def add_style_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula as 15 features estilísticas sobre o texto BRUTO (antes da limpeza).
    Preserva o estilo original do autor — não deve ser chamado após clean_text().
    """
    df = df.copy()
    df["caps_ratio"]        = df["text"].apply(_caps_ratio)
    df["exclamation_count"] = df["text"].str.count("!")
    df["title_len"]         = df["title"].str.len()
    df["text_len"]          = df["text"].str.len()
    df["word_count"]        = df["text"].str.split().str.len()
    df["avg_word_len"]      = df["text"].apply(_avg_word_len)
    df["sentence_count"]    = df["text"].str.count(r"[.!?]+")
    df["avg_sentence_len"]  = df["word_count"] / df["sentence_count"].replace(0, 1)
    df["question_count"]    = df["text"].str.count(r"\?")
    df["quote_count"]       = df["text"].apply(_quote_count)
    df["ellipsis_count"]    = df["text"].str.count(r"\.{2,}")
    df["all_caps_words"]    = df["text"].str.findall(r"\b[A-Z]{2,}\b").str.len()
    df["title_caps_ratio"]  = df["title"].apply(_caps_ratio)
    df["unique_word_ratio"] = df["text"].apply(_unique_word_ratio)
    df["sensational_count"] = df["text"].apply(lambda x: len(_SENSATIONAL_RE.findall(str(x))))
    return df


def lemmatize_series(texts: pd.Series, split: str = "") -> pd.Series:
    """
    Lematiza uma Series de textos com NLTK WordNetLemmatizer + POS tagging.
    Remove stopwords e tokens com menos de 2 caracteres.

    Parâmetros
    ----------
    texts : pd.Series de strings já limpas (após clean_text)
    split : nome do split para logging ("TRAIN" / "TEST" / "")
    """
    import nltk
    from nltk import pos_tag
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    for resource in ["punkt_tab", "wordnet", "stopwords", "averaged_perceptron_tagger_eng"]:
        nltk.download(resource, quiet=True)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    pos_map    = {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV}

    def _wn_pos(tag: str) -> str:
        return pos_map.get(tag[0], wordnet.NOUN)

    t0     = time.time()
    result = []
    for text in texts:
        tokens = word_tokenize(str(text).lower())
        lemmas = [
            lemmatizer.lemmatize(w, _wn_pos(t))
            for w, t in pos_tag(tokens)
            if w.isalpha() and len(w) > 1 and w not in stop_words
        ]
        result.append(" ".join(lemmas))
    elapsed = time.time() - t0
    label = f"[{split}] " if split else ""
    print(f"  {label}Lematizacao: {len(texts):,} textos em {elapsed:.1f}s")
    return pd.Series(result, index=texts.index)
