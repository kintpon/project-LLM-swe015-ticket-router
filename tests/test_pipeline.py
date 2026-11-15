"""Pipeline tests: train M1 on a 300-row sample, evaluate, and check the results file."""
import pandas as pd

import src.evaluate as ev
from src import m1_tfidf


def test_m1_sample_run_and_no_duplicate_rows(tmp_path, monkeypatch):
    # write results into a temp folder so the real results/ stays untouched
    monkeypatch.setattr(ev, "RESULTS_DIR", tmp_path)
    m1_tfidf.train(smoke=True, rows=300)
    predict = m1_tfidf.make_predict_proba(smoke=True)
    files = m1_tfidf.paths(smoke=True)
    for _ in range(2):  # evaluating twice must leave one row for M1
        ev.evaluate("M1", "tf-idf + LogReg", predict, files, smoke=False)
    df = pd.read_csv(tmp_path / "results.csv")
    assert list(df["model"]) == ["M1"]
    assert (tmp_path / "thresholds.json").exists() and (tmp_path / "results.md").exists()
