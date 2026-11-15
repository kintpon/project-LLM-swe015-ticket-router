"""Unit tests: Levenshtein, code extraction, spelling guard, small-talk rule, routing."""
import numpy as np

from src.evaluate import route
from src.preprocess import correct_token, levenshtein, preprocess
from src.rules import match_rule

VOCAB = frozenset({"over", "even", "the", "is", "my", "hot"})


def test_levenshtein_known_pairs():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("oven", "oven") == 0
    assert levenshtein("dishwaser", "dishwasher") == 1


def test_model_code_extraction_case_and_dash():
    for text, code in [("my DW-6534 is broken", "DW-6534"), ("my dw6534 is broken", "dw6534"),
                       ("model WD-90412 leaks", "WD-90412")]:
        clean, codes, _ = preprocess(text, VOCAB)
        assert codes == [code] and code.lower() not in clean


def test_over_oven_guard():
    assert correct_token("over", VOCAB) == "over"    # too short, in vocab
    assert correct_token("even", VOCAB) == "even"
    assert correct_token("ovn", VOCAB) == "ovn"      # under 5 letters: never corrected
    assert correct_token("dishwaser", VOCAB) == "dishwasher"


def test_preprocess_corrections_and_edge_inputs():
    clean, _, corr = preprocess("My refridgerator is soooo broken", VOCAB)
    assert "refrigerator" in clean and ("refridgerator", "refrigerator") in corr and "sooo" not in clean
    assert preprocess("", VOCAB)[0] == ""
    assert isinstance(preprocess("a" * 2000, VOCAB)[0], str)  # long input does not crash


def test_small_talk_rule():
    assert match_rule("Hello!") is not None
    assert match_rule("thanks") is not None
    assert match_rule("my dishwasher is not draining") is None


def test_routing_with_threshold():
    names = ["oos", "error_code", "greeting"]
    probs = np.array([[0.1, 0.8, 0.1],    # confident error_code
                      [0.1, 0.5, 0.4],    # low confidence
                      [0.7, 0.2, 0.1]])   # predicted oos
    out = route(probs, names, 0.6)
    assert out[0][0] == "Technical Service"
    assert out[1][0] == "Human agent"
    assert out[2][0] == "Human agent"
    assert route(probs, names, 0.0)[1][0] == "Technical Service"
