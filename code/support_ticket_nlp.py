from pathlib import Path
import re
import numpy as np
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS as EN_STOP
from spacy.lang.de.stop_words import STOP_WORDS as DE_STOP
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_FILE = str(BASE_DIR / "aa_dataset-tickets-multi-lang-5-2-50-version.csv")
RANDOM_STATE = 42

df = pd.read_csv(CSV_FILE)
df["text"] = (df["subject"].fillna("") + " " + df["body"].fillna("")).str.strip()

TOKEN_RE = re.compile(r"(?u)\b[^\W\d_][^\W_]*\b")

def preprocess(text, language):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"_", " ", text)
    stop_words = DE_STOP if str(language).lower() == "de" else EN_STOP
    tokens = TOKEN_RE.findall(text)
    return " ".join(t for t in tokens if len(t) > 1 and t not in stop_words)

df["clean_text"] = [
    preprocess(text, language)
    for text, language in zip(df["text"], df["language"])
]

queue = df["queue"].astype(str)
priority = df["priority"].astype(str).str.lower()
joint = queue + "||" + priority
indices = np.arange(len(df))
train_idx, test_idx = train_test_split(
    indices, test_size=0.20, random_state=RANDOM_STATE, stratify=joint
)

X_train_text = df.loc[train_idx, "clean_text"]
X_test_text = df.loc[test_idx, "clean_text"]

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), min_df=2, max_df=0.98,
    sublinear_tf=True, max_features=150000
)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

def evaluate(model, y):
    model.fit(X_train, y.iloc[train_idx])
    pred = model.predict(X_test)
    wp, wr, wf, _ = precision_recall_fscore_support(
        y.iloc[test_idx], pred, average="weighted", zero_division=0
    )
    mp, mr, mf, _ = precision_recall_fscore_support(
        y.iloc[test_idx], pred, average="macro", zero_division=0
    )
    return model, pred, {
        "accuracy": accuracy_score(y.iloc[test_idx], pred),
        "weighted_precision": wp, "weighted_recall": wr,
        "weighted_f1": wf, "macro_precision": mp,
        "macro_recall": mr, "macro_f1": mf,
    }

models = {
    "Linear SVM": LinearSVC(C=1.5, class_weight="balanced"),
    "Complement Naive Bayes": ComplementNB(alpha=0.1),
}

all_results = []
trained = {}
for task_name, target in [("Queue", queue), ("Priority", priority)]:
    for model_name, base_model in models.items():
        model, pred, metrics = evaluate(clone(base_model), target)
        trained[(task_name, model_name)] = (model, pred)
        row = {"task": task_name, "model": model_name}
        row.update(metrics)
        all_results.append(row)

results = pd.DataFrame(all_results)
results.to_csv(BASE_DIR / "model_comparison_results.csv", index=False)

for task_name, target in [("Queue", queue), ("Priority", priority)]:
    model, pred = trained[(task_name, "Linear SVM")]
    labels = list(model.classes_)
    report = pd.DataFrame(
        classification_report(
            target.iloc[test_idx], pred, labels=labels,
            output_dict=True, zero_division=0
        )
    ).transpose()
    report.to_csv(BASE_DIR / f"{task_name.lower()}_classification_report.csv")

    y_true = target.iloc[test_idx].astype(str).to_numpy()
    pred = np.asarray(pred).astype(str)
    labels = list(map(str, model.classes_))
    cm = confusion_matrix(y_true, pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        BASE_DIR / f"{task_name.lower()}_confusion_matrix.csv"
    )

sample = df.loc[test_idx[:10], ["subject", "body", "language"]].copy()
sample_X = vectorizer.transform(df.loc[test_idx[:10], "clean_text"])
q_model = trained[("Queue", "Linear SVM")][0]
p_model = trained[("Priority", "Linear SVM")][0]
sample["predicted_queue"] = q_model.predict(sample_X)
sample["predicted_priority"] = p_model.predict(sample_X)
sample.to_csv(BASE_DIR / "sample_predictions.csv", index=False)

print("\nDataset shape:", df.shape)
print("Training samples:", len(train_idx))
print("Testing samples:", len(test_idx))
print("TF-IDF features:", X_train.shape[1])
print("\nModel comparison:")
print(results.round(4).to_string(index=False))
print("\nQueue classification report:")
print(classification_report(
    queue.iloc[test_idx], q_model.predict(X_test),
    digits=4, zero_division=0
))
print("\nPriority classification report:")
print(classification_report(
    priority.iloc[test_idx], p_model.predict(X_test),
    digits=4, zero_division=0
))
