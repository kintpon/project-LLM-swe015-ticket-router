"""App handler tests: every input returns a route, nothing crashes (behaviour only, no accuracy numbers)."""
import json

import pytest

import app
from config import MODELS_DIR, RESULTS_DIR
from src.data import label_names

pytestmark = pytest.mark.skipif(not (MODELS_DIR / "m4_int8.onnx").exists(), reason="models/ missing")


@pytest.fixture(scope="module")
def ctx():
    thresholds = json.loads((RESULTS_DIR / "thresholds.json").read_text())
    return app.load_models(), thresholds, label_names()


@pytest.mark.parametrize("text", app.EXAMPLES + ["DW-6534", "Bugün hava çok güzel, kargom nerede?", "balance " * 300])
def test_handler_returns_route(ctx, text):
    out = app.handle(text, *ctx)
    assert out["team"] and out["routed_by"]


@pytest.mark.parametrize("text", ["", "   ", None])
def test_empty_input_no_model_call(ctx, text):
    out = app.handle(text, *ctx)
    assert out["table"] == [] and "type a message" in out["routed_by"]


def test_small_talk_uses_rules_and_model_examples_list_all_models(ctx):
    assert app.handle("hello there", *ctx)["routed_by"] == "rules"
    assert len(app.handle(app.EXAMPLES[0], *ctx)["table"]) == len(app.KEYS)


def test_non_english_goes_to_human_agent(ctx):
    assert app.handle("Bugün hava çok güzel, kargom nerede?", *ctx)["team"] == "Human agent"
