"""Area B - classical baseline M1: tf-idf (word 1-2 grams) + LogisticRegression."""
import argparse
import random

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from config import MODELS_DIR, SEED
from src.data import label_names, load_split
from src.evaluate import evaluate
from src.logutil import log
from src.preprocess import preprocess



def paths(smoke=False):
    """Model files. Smoke runs save under models/smoke/ so they never overwrite the real model."""
    folder = MODELS_DIR / "smoke" if smoke else MODELS_DIR
    return folder / "m1_vectorizer.joblib", folder / "m1_classifier.joblib"


def clean(texts):
    return [preprocess(t)[0] for t in texts]


def train(smoke=False, rows=None):
    """Fit tf-idf + LogReg on the train split (smoke: 300 rows) and save both files."""
    random.seed(SEED)
    np.random.seed(SEED)
    df = load_split("train")
    if smoke:
        df = df.sample(rows or 300, random_state=SEED)
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    x = vec.fit_transform(clean(df["text"]))
    clf = LogisticRegression(C=10, max_iter=300).fit(x, df["label"])
    vec_path, clf_path = paths(smoke)
    vec_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, vec_path)
    joblib.dump(clf, clf_path)


def make_predict_proba(smoke=False):
    """Load saved files; return predict_proba(raw_texts) -> (n, n_classes) probabilities."""
    vec_path, clf_path = paths(smoke)
    vec, clf = joblib.load(vec_path), joblib.load(clf_path)
    n = len(label_names())

    def predict_proba(texts):
        p = clf.predict_proba(vec.transform(clean(texts)))
        full = np.zeros((len(texts), n))  # a smoke run may not have seen every class
        full[:, clf.classes_] = p
        return full
    return predict_proba


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    smoke = ap.parse_args().smoke
    train(smoke)
    row, _ = evaluate("M1", "tf-idf + LogReg", make_predict_proba(smoke), paths(smoke), smoke)
    log("m1", f"M1 smoke={smoke} in_scope={row['in_scope_acc']:.4f} oos_recall={row['oos_recall']:.4f}")
