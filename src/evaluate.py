"""ONE shared evaluation for all models: threshold tuning on val, metrics on test, size, latency."""
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import HUMAN_TEAM, OOS_INTENT, RESULTS_DIR, SEED
from src.data import label_names, load_split
from src.teams import TEAM_OF

COLUMNS = ["model", "name", "in_scope_acc", "oos_recall", "appliance_acc", "threshold",
           "size_mb", "latency_mean_ms", "latency_p95_ms"]
APPLIANCE = ["not_draining", "not_heating", "error_code", "spare_part_order", "warranty_claim"]


def route(probs, names, threshold):
    """probs (n, classes) -> list of (team, intent, confidence). Low confidence or oos -> Human agent."""
    out = []
    for p in probs:
        k = int(np.argmax(p))
        conf = float(p[k])
        human = names[k] == OOS_INTENT or conf < threshold
        out.append((HUMAN_TEAM if human else TEAM_OF[names[k]], names[k], conf))
    return out


def scores(df, probs, names, threshold):
    """(in-scope accuracy, oos recall, appliance accuracy) after routing."""
    routed = route(probs, names, threshold)
    true = df["intent"].tolist()
    correct = [r[1] == t and r[0] != HUMAN_TEAM for r, t in zip(routed, true)]
    in_scope = [i for i, t in enumerate(true) if t != OOS_INTENT]
    oos = [i for i, t in enumerate(true) if t == OOS_INTENT]
    app = [i for i, t in enumerate(true) if t in APPLIANCE]

    def acc(idx):
        return float(np.mean([correct[i] for i in idx])) if idx else float("nan")

    oos_recall = float(np.mean([routed[i][0] == HUMAN_TEAM for i in oos])) if oos else float("nan")
    return acc(in_scope), oos_recall, acc(app)


def tune_threshold(val_df, val_probs, names):
    """Grid 0.00-0.95 step 0.05 on VAL only; maximise (in-scope acc + oos recall) / 2."""
    best, best_score = 0.0, -1.0
    for thr in np.round(np.arange(0.0, 0.96, 0.05), 2):
        a, r, _ = scores(val_df, val_probs, names, thr)
        if (a + r) / 2 > best_score:
            best, best_score = float(thr), (a + r) / 2
    return best


def size_mb(files):
    """Total size of the files a model needs at inference (tokenizer files excluded)."""
    return sum(Path(f).stat().st_size for f in files) / 1e6


def latency(predict_proba, names, threshold, texts):
    """One message at a time, end to end (preprocess inside predict_proba, then route). Returns (mean, p95) ms."""
    for t in texts[:10]:  # warm-up
        route(predict_proba([t]), names, threshold)
    times = []
    for t in texts[10:210]:  # 200 messages, the same for every model
        start = time.perf_counter()
        route(predict_proba([t]), names, threshold)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.mean(times)), float(np.percentile(times, 95))


def hardware_line():
    """CPU model, cores, RAM, OS, Python and library versions."""
    import onnxruntime
    import torch
    try:
        cpu = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True).stdout.strip() or platform.processor()
    except OSError:
        cpu = platform.processor()
    return (f"{cpu} ({os.cpu_count()} logical cores), {platform.platform()}, Python {platform.python_version()}, "
            f"torch {torch.__version__}, onnxruntime {onnxruntime.__version__}; latency on CPU, 4 threads")


def evaluate(key, name, predict_proba, files, smoke=False):
    """Evaluate one model: threshold on val, metrics + latency on test; update the results files."""
    import torch
    torch.set_num_threads(4)
    names = label_names()
    val, test = load_split("val"), load_split("test")
    thr = tune_threshold(val, predict_proba(val["text"].tolist()), names)
    test_probs = predict_proba(test["text"].tolist())
    a, r, app = scores(test, test_probs, names, thr)
    sample = test["text"].sample(210, random_state=SEED).tolist()
    mean_ms, p95_ms = latency(predict_proba, names, thr, sample)
    row = dict(model=key, name=name, in_scope_acc=a, oos_recall=r, appliance_acc=app, threshold=thr,
               size_mb=size_mb(files), latency_mean_ms=mean_ms, latency_p95_ms=p95_ms)
    if not smoke:
        write_results(row, thr)
    print(f"{key}: in-scope {a:.4f}, oos recall {r:.4f}, thr {thr}, {row['size_mb']:.1f} MB, {mean_ms:.2f} ms")
    return row, test_probs


def write_results(row, thr):
    """Replace this model's row in results.csv, update thresholds.json, rebuild results.md."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "results.csv"
    df = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=COLUMNS)
    df = df[df["model"] != row["model"]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    order = {"M1": 0, "M2": 1, "M3": 2, "M3s": 3, "M4": 4}
    df = df.sort_values("model", key=lambda s: s.map(order)).reset_index(drop=True)
    df.to_csv(path, index=False)
    tpath = RESULTS_DIR / "thresholds.json"
    th = json.loads(tpath.read_text()) if tpath.exists() else {}
    th[row["model"]] = thr
    tpath.write_text(json.dumps(th, indent=1))
    lines = ["| Model | In-scope acc | OOS recall | Appliance acc (30 rows) | Threshold | Size (MB) | "
             "Latency mean (ms) | Latency p95 (ms) |", "|---|---|---|---|---|---|---|---|"]
    for _, x in df.iterrows():
        lines.append(f"| {x['model']} {x['name']} | {x.in_scope_acc:.4f} | {x.oos_recall:.4f} | {x.appliance_acc:.4f} | "
                     f"{x.threshold:.2f} | {x.size_mb:.1f} | {x.latency_mean_ms:.2f} | {x.latency_p95_ms:.2f} |")
    notes = ("\n\nHardware: " + hardware_line() + "\n\nNotes: size = all files needed at inference, tokenizer files "
             "excluded. Latency = one message at a time, preprocess -> route, 10 warm-up calls then 200 test messages "
             "(same for all models). Metrics are measured on the models alone (small-talk rules not applied). "
             "Appliance accuracy covers only 30 test rows, so treat it with caution.\n")
    (RESULTS_DIR / "results.md").write_text("\n".join(lines) + notes, encoding="utf-8")
