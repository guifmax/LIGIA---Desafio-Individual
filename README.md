# 🔍 Desafio Técnico Ligia 2026 — Classificação de Desinformação Digital (NLP)

Este repositório contém a solução completa para o desafio de **detecção de Fake News** da Liga Acadêmica de IA (LIGIA/UFPE) 2026, utilizando técnicas de NLP, Engenharia de Features Estilísticas e SVM Linear calibrado.

## 📊 Resultados

| Métrica | Valor (CV 5-Fold) |
|---|---|
| **F1-Score** | **0.99807** |
| Accuracy | 0.99904 |
| Precision | 0.99854 |
| Recall | 0.99762 |
| ROC AUC | 0.99998 |

- **Robustez:** F1 estável em 5 seeds distintas (amplitude = 0.00053)
- **Threshold otimizado:** 0.335 (tuning sem data leakage)

## 📂 Estrutura do Projeto

```
LIGIA_FINAL/
├── inputs/                      # Datasets originais
│   ├── train.csv                # Dados de treino (22.844 artigos)
│   └── test.csv                 # Dados de teste (5.712 artigos)
├── notebooks/                   # Pipeline sequencial
│   ├── notebook_00_EDA.ipynb            # 1. Análise Exploratória
│   ├── notebook_01_preprocessing.ipynb  # 2. Pré-processamento e Feature Engineering
│   ├── notebook_02_modeling.ipynb       # 3. Modelagem, CV, SHAP e Submissão
│   └── notebook_03_inference.ipynb      # 4. Inferência e Análise de Erros
├── outputs/
│   ├── artifacts/               # Modelos e artefatos salvos
│   │   ├── best_model.pkl       # LinearSVC + CalibratedClassifierCV (modelo final)
│   │   ├── best_model_v2.pkl    # Versão alternativa do modelo
│   │   ├── best_threshold.pkl   # Threshold otimizado (0.335)
│   │   ├── tfidf_vectorizer.pkl # Vetorizador TF-IDF fitado no treino
│   │   ├── style_scaler.pkl     # MaxAbsScaler para features de estilo
│   │   ├── subject_encoder.pkl  # LabelEncoder para a coluna 'subject'
│   │   ├── X_train.npz / X_test.npz  # Matrizes sparse (TF-IDF + estilo)
│   │   └── *.csv               # Datasets intermediários
│   └── figures/                 # Gráficos gerados
│       ├── shap_bar.png         # SHAP feature importance
│       ├── shap_summary.png     # SHAP summary plot
│       ├── confusion_matrix.png # Matriz de confusão
│       └── ...                  # Learning curve, threshold, CV results
├── report/                      # Relatório técnico-científico (IEEE)
├── submission.csv               # Arquivo de submissão Kaggle
├── requirements.txt             # Dependências com versões fixas
└── README.md                    # Este arquivo
```

## 🧠 Metodologia

### Pipeline
1. **Remoção de Data Leakage:** Tags de agência (Reuters, AP, AFP), URLs, bylines
2. **Feature Engineering Estilístico (15 features):** `caps_ratio`, `exclamation_count`, `word_count`, `avg_word_len`, `sentence_count`, `avg_sentence_len`, `question_count`, `quote_count`, `ellipsis_count`, `all_caps_words`, `title_caps_ratio`, `unique_word_ratio`, `sensational_count`, `title_len`, `text_len`
3. **Pré-processamento de Texto:** Lematização (NLTK WordNet + POS tagging), remoção de stopwords
4. **Vetorização:** TF-IDF (unigrams + bigrams, max 12.000 features)
5. **Modelo:** `LinearSVC(C=1.0, class_weight='balanced')` + `CalibratedClassifierCV(method='sigmoid', cv=3)`
6. **Threshold Tuning:** Otimização do limiar de decisão em holdout separado

### Interpretabilidade
- **Coeficientes SVC:** Ranking direto dos termos mais discriminativos (Fake vs Real)
- **SHAP LinearExplainer:** Explicação global e local das decisões do modelo

## 🚀 Como Executar (Reprodutibilidade)

### 1. Clonar o Repositório
```bash
git clone <URL_DO_REPOSITÓRIO>
cd LIGIA_FINAL
```

### 2. Configuração do Ambiente
```bash
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente (Windows)
venv\Scripts\activate

# Ativar ambiente (Linux/Mac)
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### 3. Dados
Coloque os arquivos `train.csv` e `test.csv` da competição Kaggle na pasta `inputs/`.

### 4. Execução dos Notebooks (em ordem)

```bash
jupyter notebook
```

Execute os notebooks **sequencialmente**:

| Ordem | Notebook | Descrição | Saídas |
|---|---|---|---|
| 1 | `notebook_00_EDA.ipynb` | Análise Exploratória: distribuição de classes, features estilísticas, correlações | Gráficos em `outputs/figures/` |
| 2 | `notebook_01_preprocessing.ipynb` | Limpeza, feature engineering, TF-IDF, salvamento de artefatos | `X_train.npz`, `X_test.npz`, `tfidf_vectorizer.pkl`, `style_scaler.pkl` |
| 3 | `notebook_02_modeling.ipynb` | Treinamento, CV, threshold tuning, SHAP, geração da submissão | `best_model.pkl`, `best_threshold.pkl`, `submission.csv` |
| 4 | `notebook_03_inference.ipynb` | Inferência em novos artigos, análise de erros, validação de coerência | Análises de confiança e zona de incerteza |

### 5. Submissão
Após executar o notebook 02 ou 03, o arquivo `submission.csv` será gerado na raiz do projeto, pronto para upload no Kaggle.

## 📝 Relatório
O relatório técnico-científico no formato IEEE encontra-se na pasta `report/`.

## 📦 Tecnologias
- Python 3.x
- scikit-learn (LinearSVC, CalibratedClassifierCV, TF-IDF)
- NLTK (lematização, stopwords, POS tagging)
- SHAP (interpretabilidade)
- pandas, numpy, matplotlib, seaborn