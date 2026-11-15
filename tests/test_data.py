"""Data integrity tests on the cached splits."""
import difflib

from src.data import label_names, load_split
from src.teams import TEAM_OF

APPLIANCE = {"not_draining", "not_heating", "error_code", "spare_part_order", "warranty_claim"}


def test_no_text_in_both_train_and_test():
    train = set(load_split("train")["text"].str.lower())
    test = set(load_split("test")["text"].str.lower())
    assert not train & test


def test_every_intent_has_a_team():
    assert all(name in TEAM_OF for name in label_names())


def test_label_ids_identical_across_splits():
    maps = []
    for part in ["train", "val", "test"]:
        df = load_split(part)
        maps.append(dict(zip(df["intent"], df["label"])))
    for m in maps[1:]:
        assert all(maps[0][k] == v for k, v in m.items())


def test_appliance_rows_split_14_3_3():
    for part, n in [("train", 14), ("val", 3), ("test", 3)]:
        df = load_split(part)
        counts = df[df["source"] == "appliance"]["intent"].value_counts()
        assert set(counts.index) == APPLIANCE and (counts == n).all()


def test_class_count_equals_distinct_intents():
    train = load_split("train")
    assert len(label_names()) == train["intent"].nunique() == train["label"].nunique()


def test_appliance_test_rows_not_near_copies_of_train():
    train = load_split("train")
    train = train[train["source"] == "appliance"]["text"].str.lower().tolist()
    test = load_split("test")
    for text in test[test["source"] == "appliance"]["text"].str.lower():
        assert max(difflib.SequenceMatcher(None, text, t).ratio() for t in train) < 0.9
