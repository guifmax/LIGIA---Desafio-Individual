"""
constants.py — Constantes globais do projeto
Liga Acadêmica de IA (LIGIA/UFPE) 2026 — Detecção de Desinformação Digital

Importado por todos os notebooks e por src/preprocessing.py
"""

# Features de estilo calculadas antes da limpeza textual
STYLE_COLS = [
    "caps_ratio", "exclamation_count", "title_len",
    "text_len", "word_count", "avg_word_len",
    "sentence_count", "avg_sentence_len",
    "question_count", "quote_count", "ellipsis_count",
    "all_caps_words", "title_caps_ratio",
    "unique_word_ratio", "sensational_count",
]

# Sujeitos do dataset associados a fontes Reuters/AP (classe Real)
REUTERS_SUBJECTS = frozenset({"politicsNews", "worldnews"})

# Termos sensacionalistas detectados na EDA
SENSATIONAL_TERMS = (
    "shocking", "unbelievable", "amazing", "incredible",
    "must see", "breaking", "exclusive", "urgent",
)

# Hiperparametros do modelo final (justificados no notebook_02)
MODEL_PARAMS = {
    "C": 1.5,
    "loss": "squared_hinge",
    "dual": False,
    "max_iter": 3000,
    "class_weight": "balanced",
    "random_state": 42,
}

CALIB_PARAMS = {
    "method": "sigmoid",
    "cv": 3,
}

# TF-IDF
TFIDF_PARAMS = {
    "stop_words": "english",
    "max_features": 12_000,
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
    "ngram_range": (1, 3),   # unigramas, bigramas e trigramas
}

# Validacao cruzada
CV_SPLITS   = 5
CV_SEED     = 42
HOLDOUT_PCT = 0.15
