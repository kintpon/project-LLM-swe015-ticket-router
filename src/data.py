"""Data: CLINC150 'plus' + 100 appliance rows -> cached parquet splits (seed 42)."""
import random

import pandas as pd

from config import APPLIANCE_CSV, SEED, SPLITS_DIR


def build_splits():
    """Build train/val/test once and cache to data/splits/*.parquet."""
    from datasets import load_dataset

    ds = load_dataset("clinc/clinc_oos", "plus")
    names = ds["train"].features["intent"].names  # 150 intents + 'oos'
    app = pd.read_csv(APPLIANCE_CSV)
    all_names = names + sorted(app["intent"].unique())  # label id = position in this list
    label_of = {n: i for i, n in enumerate(all_names)}

    splits = {}
    for part in ["train", "validation", "test"]:
        df = ds[part].to_pandas()[["text", "intent"]]
        df["intent"] = df["intent"].map(lambda i: names[i])
        df["source"] = "clinc"
        splits["val" if part == "validation" else part] = df

    # appliance rows: per intent 14 train / 3 val / 3 test (random with seed 42)
    rng = random.Random(SEED)
    parts = {"train": [], "val": [], "test": []}
    for intent, g in app.groupby("intent"):
        idx = list(g.index)
        rng.shuffle(idx)
        parts["train"] += idx[:14]
        parts["val"] += idx[14:17]
        parts["test"] += idx[17:20]
    for part, idx in parts.items():
        df = app.loc[idx, ["text", "intent"]].copy()
        df["source"] = "appliance"
        splits[part] = pd.concat([splits[part], df], ignore_index=True)

    # drop train rows whose text also appears in val/test (a few exact duplicates in CLINC)
    held_out = set(splits["val"]["text"].str.lower()) | set(splits["test"]["text"].str.lower())
    splits["train"] = splits["train"][~splits["train"]["text"].str.lower().isin(held_out)]

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for part, df in splits.items():
        df["label"] = df["intent"].map(label_of)
        df = df[["text", "label", "intent", "source"]]
        df.to_parquet(SPLITS_DIR / f"{part}.parquet", index=False)
    return splits


def load_split(name):
    """Read a cached split ('train', 'val' or 'test'); build them if missing."""
    path = SPLITS_DIR / f"{name}.parquet"
    if not path.exists():
        build_splits()
    return pd.read_parquet(path)


def label_names():
    """List of intent names, index = label id (read from the train split)."""
    df = load_split("train")
    pairs = df[["label", "intent"]].drop_duplicates().sort_values("label")
    return pairs["intent"].tolist()


if __name__ == "__main__":
    s = build_splits()
    print({k: len(v) for k, v in s.items()}, "classes:", len(label_names()))
