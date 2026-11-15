"""Data checks: split overlaps (raw and after preprocessing) and spelling-correction side effects.
Uses train and val only for the correction check (test is never looked at). Writes
results/data_checks.md and puts the same text into README.md between the DATA-CHECKS markers."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collections import Counter

from config import RESULTS_DIR
from src.data import load_split
from src.preprocess import preprocess

PAIRS = [("train", "test"), ("train", "val"), ("val", "test")]


def overlap(sets, fn_name):
    return {f"{a} & {b}": len(sets[a] & sets[b]) for a, b in PAIRS}, fn_name


def main():
    from datasets import load_dataset
    ds = load_dataset("clinc/clinc_oos", "plus")
    official = {"train": ds["train"]["text"], "val": ds["validation"]["text"], "test": ds["test"]["text"]}
    ours = {p: load_split(p)["text"].tolist() for p in ["train", "val", "test"]}

    lines = ["### Data checks and limitations (written by scripts/data_checks.py)", "",
             "Number of texts that appear in two splits (compared case-insensitively):", "",
             "| Splits | Official CLINC, raw | Our splits, raw | Our splits, after preprocessing |", "|---|---|---|---|"]
    low = lambda texts: {t.lower() for t in texts}
    clean = lambda texts: {preprocess(t)[0] for t in texts}
    raw_off = {p: low(t) for p, t in official.items()}
    raw_our = {p: low(t) for p, t in ours.items()}
    pre_our = {p: clean(t) for p, t in ours.items()}
    for a, b in PAIRS:
        lines.append(f"| {a} & {b} | {len(raw_off[a] & raw_off[b])} | {len(raw_our[a] & raw_our[b])} | "
                     f"{len(pre_our[a] & pre_our[b])} |")
    lines += ["", "Our splits = official CLINC splits with the train rows that also appear in val/test removed, "
              "plus the 100 appliance rows.", ""]

    # spelling correction side effects, train and val only (CLINC rows, no deliberate typos)
    changed = Counter()
    n_rows = 0
    for part in ["train", "val"]:
        df = load_split(part)
        for text in df[df["source"] == "clinc"]["text"]:
            n_rows += 1
            for pair in preprocess(text)[2]:
                changed[pair] += 1
    lines += [f"Spelling correction on {n_rows} CLINC rows of train+val: {sum(changed.values())} tokens changed "
              f"({len(changed)} distinct). Words in the CLINC training vocabulary are never changed, so all of them "
              "come from validation rows. Each one is a wrong correction (CLINC has no deliberate typos):", ""]
    lines += [f"- `{a}` -> `{b}` ({n}x)" for (a, b), n in changed.most_common()] or ["- none"]
    text = "\n".join(lines) + "\n"
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "data_checks.md").write_text(text, encoding="utf-8")

    readme = ROOT / "README.md"
    body = readme.read_text(encoding="utf-8") if readme.exists() else "# SWE015 Customer Support Ticket Router\n"
    block = f"<!-- DATA-CHECKS-START -->\n{text}<!-- DATA-CHECKS-END -->"
    if "<!-- DATA-CHECKS-START -->" in body:
        body = re.sub(r"<!-- DATA-CHECKS-START -->.*<!-- DATA-CHECKS-END -->", lambda m: block, body, flags=re.S)
    else:
        body = body.rstrip() + "\n\n" + block + "\n"
    readme.write_text(body, encoding="utf-8")
    print("wrote results/data_checks.md and README.md data-checks block")


if __name__ == "__main__":
    main()
