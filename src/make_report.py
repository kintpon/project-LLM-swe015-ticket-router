"""Writes results/failures.csv (10 wrong M4 predictions) and results/accuracy_vs_latency.png from saved outputs."""
import json

import matplotlib
import pandas as pd

from config import HUMAN_TEAM, MODELS_DIR, OOS_INTENT, RESULTS_DIR, SEED
from src.data import label_names, load_split
from src.evaluate import route
from src.m4_quantize import make_predict_proba

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after choosing the backend)


def write_failures():
    """10 random wrong M4 test predictions (seed 42). Wrong = not the true intent, or sent to a human agent."""
    names, test = label_names(), load_split("test")
    thr = json.loads((RESULTS_DIR / "thresholds.json").read_text())["M4"]
    predict = make_predict_proba(MODELS_DIR / "m4_int8.onnx", MODELS_DIR / "m3s_distilbert")
    routed = route(predict(test["text"].tolist()), names, thr)
    rows = []
    for text, true, (team, pred, conf) in zip(test["text"], test["intent"], routed):
        # oos messages sent to the human agent are correct; every other message needs its own intent and a real team
        wrong = team != HUMAN_TEAM if true == OOS_INTENT else (pred != true or team == HUMAN_TEAM)
        if wrong:
            rows.append(dict(text=text, true=true, predicted=pred, confidence=round(conf, 4), routed_to=team))
    pd.DataFrame(rows).sample(10, random_state=SEED).to_csv(RESULTS_DIR / "failures.csv", index=False)
    return len(rows)


def write_plot():
    """One scatter: latency (log scale) against in-scope accuracy, one point per model."""
    df = pd.read_csv(RESULTS_DIR / "results.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["latency_mean_ms"], df["in_scope_acc"])
    for _, r in df.iterrows():
        ax.annotate(r["model"], (r["latency_mean_ms"], r["in_scope_acc"]), textcoords="offset points", xytext=(5, 5))
    ax.set_xscale("log")
    ax.set_xlabel("latency per message, mean (ms, CPU, log scale)")
    ax.set_ylabel("in-scope accuracy (test)")
    ax.set_title("Accuracy vs latency")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "accuracy_vs_latency.png", dpi=150)


if __name__ == "__main__":
    n_wrong = write_failures()
    write_plot()
    print(f"failures.csv written (10 of {n_wrong} wrong M4 predictions), accuracy_vs_latency.png written")
