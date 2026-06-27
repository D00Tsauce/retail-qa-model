# Financial Question Classification Pipeline

An optimized, lightweight Machine Learning text classifier built with Python and `scikit-learn`. This pipeline processes financial text queries, extracts high-dimensional linguistic features, and categorizes them into distinct problem-solving types: **Conceptual**, **Basic**, and **Assumption**.

## 🚀 Features & Upgrades
* **Advanced Text Vectorization:** Utilizes `TfidfVectorizer` capturing both unigrams and bi-grams (`ngram_range=(1, 2)`) to preserve multi-word contextual structures (e.g., "assuming that", "calculate cogs").
* **Sublinear Scaling:** Smooths out term frequency spikes to prevent repetitive jargon from dominant text entries from overwhelming the feature weights.
* **Support Vector Machine Baseline:** Implements a `LinearSVC` with balanced class weights to maximize geometric separation boundaries on highly specialized, lower-frequency text data.

---

## 📊 Model Performance

The pipeline successfully cleared the initial training bottleneck to reach a high-accuracy classification threshold:

* **Overall Accuracy:** `87.00%`
* **Macro Average F1-Score:** `0.85`

### Classification Metrics Breakdown

| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Conceptual** | 0.93 | 1.00 | 0.96 | 13 |
| **Basic** | 0.78 | 0.88 | 0.82 | 8 |
| **Assumption** | 0.86 | 0.67 | 0.75 | 9 |

---

## 📂 Project Directory Structure

```text
ml-classifier-project/
│
├── data/
│   └── test.csv               # Raw financial text dataset
│
├── .gitignore                 # Excludes local run caches and environments
├── README.md                  # Project documentation
├── requirements.txt           # Active library dependency list
└── train.py                   # Main data-loading, extraction, and training script