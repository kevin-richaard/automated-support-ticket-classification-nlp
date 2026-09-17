# Automated Support-Ticket Classification and Priority Prediction Using NLP

NLP project for automated classification of customer support tickets into an appropriate **queue** and **priority level** using classical text-classification methods.

## Project Overview

The system takes a support ticket's subject and body as text input, performs multilingual preprocessing, converts the text into TF-IDF features, and predicts:

- **Queue:** 10 support queues
- **Priority:** Low, Medium, High

The implementation compares **Linear SVM** and **Complement Naive Bayes**. Linear SVM was used for the demonstration because it achieved higher performance in the executed experiment.

## Dataset

Dataset file used in the experiment:

`aa_dataset-tickets-multi-lang-5-2-50-version.csv`

The experiment used **28,587 records** with **16 columns**. The dataset contains English and German support tickets and includes ticket subject, body, answer, type, queue, priority, language, version, and tag fields.

Dataset source: Tobi Bueck, Customer Support Tickets — Hugging Face.

DOI: `10.57967/hf/6184`

The raw CSV is intentionally **not committed to this repository**. See [`data/README.md`](data/README.md) for dataset information and retrieval details.

### Dataset audit

- Records: 28,587
- Columns: 16
- Languages: English and German
- Ticket types: Incident, Request, Problem, Change
- Queue classes: 10
- Priority classes used: 3
- Exact duplicate rows: 0
- Missing subject values: 3,838 (13.43%)
- Missing body values: 0

## NLP Pipeline

```text
Support Ticket
      |
      v
Subject + Body
      |
      v
Text Cleaning & Normalization
      |
      v
Tokenization + Language-specific Stop-word Removal
      |
      v
TF-IDF (unigrams + bigrams)
      |
      v
Linear SVM / Complement Naive Bayes
      |
      +------------------+
      |                  |
      v                  v
Queue Prediction    Priority Prediction
```

### Preprocessing

- Lowercasing
- URL/noise removal
- Whitespace and punctuation normalization
- Tokenization
- English/German stop-word removal
- Subject and body combined as the model input
- Body remains available when the subject is missing

Stemming was not forced into the final pipeline because the tested configuration did not improve the experimental result.

### Input and leakage control

The primary text input is `subject + body`.

The following fields were excluded from the primary text experiment to avoid target leakage or use of metadata that would make the classification task less representative of raw ticket text:

- `answer`
- `tag_1` through `tag_8`
- Target labels (`queue`, `priority`)
- `version` metadata

## Feature Extraction

TF-IDF was configured as follows:

- `ngram_range=(1, 2)`
- `min_df=2`
- `max_df=0.98`
- `sublinear_tf=True`
- `max_features=150000`

The executed representation contained **88,870 features**.

## Models

### Linear SVM

`LinearSVC(C=1.5, class_weight="balanced")`

### Complement Naive Bayes

`ComplementNB(alpha=0.1)`

## Experimental Setup

- Train/test split: 80/20
- Training records: 22,869
- Testing records: 5,718
- Random state: 42
- Stratification: queue + priority combination
- Evaluation metrics: accuracy, precision, recall, weighted F1, macro F1

## Verified Experimental Results

| Task | Model | Accuracy | Weighted F1 | Macro F1 |
|---|---|---:|---:|---:|
| Queue | Linear SVM | 64.76% | 64.66% | 64.78% |
| Queue | Complement Naive Bayes | 58.27% | 57.56% | 55.28% |
| Priority | Linear SVM | 66.68% | 66.60% | 65.80% |
| Priority | Complement Naive Bayes | 63.40% | 63.34% | 62.21% |

## Repository Structure

```text
automated-support-ticket-classification-nlp/
├── README.md
├── requirements.txt
├── code/
│   ├── support_ticket_nlp.py
│   └── support_ticket_demo.py
├── data/
│   └── README.md
└── results/
    ├── final_experimental_summary.csv
    ├── queue_classification_report.csv
    ├── priority_classification_report.csv
    ├── queue_confusion_matrix.csv
    └── priority_confusion_matrix.csv
```

## How to Run

1. Obtain the dataset described in `data/README.md`.
2. Place `aa_dataset-tickets-multi-lang-5-2-50-version.csv` in the project root.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the full experiment:

```bash
python code/support_ticket_nlp.py
```

5. Run the interactive demonstration:

```bash
python code/support_ticket_demo.py
```

The experiment script writes generated comparison, classification-report, confusion-matrix, and sample-prediction CSV files into `results/`.

## Limitations

- The dataset is synthetic rather than production support data.
- The experiment covers English and German tickets.
- Queue classes are imbalanced.
- The implementation uses classical TF-IDF features rather than contextual transformer embeddings.
- Evaluation uses a single stratified train/test split.
- The demonstration is a prototype and is not connected to a live support-ticketing platform.

## Future Scope

Potential extensions include transformer-based text representations, larger multilingual datasets, confidence-aware routing, continuous model evaluation, and integration with a real support-ticket platform.

## Team Members

- 2463034 — Kevin Richaard G
- 2463071 — Vikash S
- 2463013 — Bincy Bijoy
- 2463024 — Heba Benny
- 2463012 — Bettina Soni

## Academic Context

This repository supports **CIA3 Component 2 — Project Implementation and Demonstration** for **AI532P – Introduction to NLP**, B.Tech AIML, CHRIST (Deemed to be University).
