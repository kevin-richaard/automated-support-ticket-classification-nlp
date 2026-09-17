from pathlib import Path
import re
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS as EN_STOP
from spacy.lang.de.stop_words import STOP_WORDS as DE_STOP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_FILE = BASE_DIR / "aa_dataset-tickets-multi-lang-5-2-50-version.csv"

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

df["clean_text"] = [preprocess(t, l) for t, l in zip(df["text"], df["language"])]

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), min_df=2, max_df=0.98,
    sublinear_tf=True, max_features=150000
)
X = vectorizer.fit_transform(df["clean_text"])

queue_model = LinearSVC(C=1.5, class_weight="balanced").fit(X, df["queue"])
priority_model = LinearSVC(C=1.5, class_weight="balanced").fit(
    X, df["priority"].astype(str).str.lower()
)

print("Support Ticket NLP Demo")
print("-" * 40)
subject = input("Enter ticket subject: ")
body = input("Enter ticket body: ")
language = input("Language (en/de): ").strip().lower()

text = preprocess(subject + " " + body, language)
x = vectorizer.transform([text])

print("\nPredicted Queue   :", queue_model.predict(x)[0])
print("Predicted Priority:", priority_model.predict(x)[0])
