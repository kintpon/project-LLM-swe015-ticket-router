"""Copy numbers into the docs so no number is typed by hand.
Fills the blocks between <!-- NAME-START --> / <!-- NAME-END --> markers:
  docs/slides_material.md: DATA (rows per split), RESULTS (results/results.md), FAILURES (results/failures.csv)
  README.md:               RESULTS (results/results.md)"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from config import RESULTS_DIR  # noqa: E402
from src.data import label_names, load_split  # noqa: E402
from src.teams import TEAM_OF  # noqa: E402


def data_block():
    lines = ["| Split | CLINC rows | Appliance rows | Total |", "|---|---|---|---|"]
    for name in ["train", "val", "test"]:
        n = load_split(name)["source"].value_counts()
        lines.append(f"| {name} | {n.get('clinc', 0)} | {n.get('appliance', 0)} | {n.sum()} |")
    names = label_names()
    lines += ["", f"Classes (intents incl. `oos`): {len(names)}. Teams: {len(set(TEAM_OF.values()))}."]
    return "\n".join(lines)


def failures_block():
    df = pd.read_csv(RESULTS_DIR / "failures.csv")
    lines = ["| Text | True | Predicted | Confidence | Routed to |", "|---|---|---|---|---|"]
    lines += [f"| {r.text} | {r.true} | {r.predicted} | {r.confidence} | {r.routed_to} |" for r in df.itertuples()]
    return "\n".join(lines)


def fill(path, blocks):
    body = path.read_text(encoding="utf-8")
    for name, text in blocks.items():
        start, end = f"<!-- {name}-START -->", f"<!-- {name}-END -->"
        if start not in body:
            raise SystemExit(f"{path.name}: marker {start} missing")
        body = re.sub(re.escape(start) + ".*?" + re.escape(end), lambda m: f"{start}\n{text}\n{end}", body, flags=re.S)
    path.write_text(body, encoding="utf-8")


if __name__ == "__main__":
    results = (RESULTS_DIR / "results.md").read_text(encoding="utf-8").strip()
    fill(ROOT / "docs" / "slides_material.md", dict(DATA=data_block(), RESULTS=results, FAILURES=failures_block()))
    fill(ROOT / "README.md", dict(RESULTS=results))
    print("filled docs/slides_material.md (DATA, RESULTS, FAILURES) and README.md (RESULTS)")
