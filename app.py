"""Gradio app: small-talk rules first, then M4 (INT8 student, ONNX Runtime) decides the route.
All five models run on every message so they can be compared side by side."""
import json
import time

from config import HF_MODEL_REPO, HUMAN_TEAM, MODELS_DIR, RESULTS_DIR
from src.data import label_names
from src.evaluate import route
from src.preprocess import preprocess
from src.rules import match_rule

MAX_CHARS = 500
KEYS = ["M1", "M2", "M3", "M3s", "M4"]
EXAMPLES = ["I want to check the balance of my savings account",
            "my dishwaher DW-6534 does not drain the water anymore",
            "who won the football match yesterday",
            "hello there"]


def ensure_models():
    """Download the weights from the public HF repo if models/ is missing (no token needed)."""
    if not (MODELS_DIR / "m4_int8.onnx").exists():
        from huggingface_hub import snapshot_download
        snapshot_download(HF_MODEL_REPO, local_dir=MODELS_DIR)


def load_models():
    """Load all five models once. Returns {key: predict_proba}."""
    ensure_models()
    from src import m1_tfidf, m2_ffn, m3_bert, m4_quantize
    m3s = MODELS_DIR / "m3s_distilbert"
    return {"M1": m1_tfidf.make_predict_proba(), "M2": m2_ffn.make_predict_proba(),
            "M3": m3_bert.make_predict_proba(MODELS_DIR / "m3_bert"), "M3s": m3_bert.make_predict_proba(m3s),
            "M4": m4_quantize.make_predict_proba(MODELS_DIR / "m4_int8.onnx", m3s)}


def handle(text, models, thresholds, names):
    """Route one message. Returns a dict with everything the UI shows (never raises on odd input)."""
    text = (text or "").strip()[:MAX_CHARS]  # long input is truncated before the models
    out = dict(cleaned="", corrections="", codes="", team="", intent="", confidence=None, routed_by="", table=[])
    if not text:
        out["routed_by"] = "Please type a message first."
        return out
    clean, codes, corrections = preprocess(text)
    out.update(cleaned=clean, codes=", ".join(codes),
               corrections=", ".join(f"{a} -> {b}" for a, b in corrections))
    reply = match_rule(text)
    if reply:  # small talk: answered by rules, no model decides
        out.update(team="Auto reply", intent="small talk", routed_by="rules", reply=reply)
        return out
    for key in KEYS:
        start = time.perf_counter()
        team, intent, conf = route(models[key]([text]), names, thresholds[key])[0]
        ms = (time.perf_counter() - start) * 1000
        out["table"].append([key, intent, team, round(conf, 3), round(ms, 1)])
        if key == "M4":  # M4 decides the route
            out.update(team=team, intent=intent, confidence=round(conf, 3),
                       routed_by="human-agent fallback" if team == HUMAN_TEAM else "model")
    return out


def build_ui():
    import gradio as gr
    models, names = load_models(), label_names()
    thresholds = json.loads((RESULTS_DIR / "thresholds.json").read_text())

    def run(text):
        r = handle(text, models, thresholds, names)
        summary = r["routed_by"] if not r["team"] else (
            f"**Team:** {r['team']}  \n**Intent:** {r['intent']}  \n"
            f"**Confidence:** {r['confidence'] if r['confidence'] is not None else '-'}  \n**Routed by:** {r['routed_by']}"
            + (f"  \n**Reply:** {r['reply']}" if r.get("reply") else ""))
        return summary, r["cleaned"], r["corrections"], r["codes"], r["table"]

    with gr.Blocks(title="Ticket Router") as demo:
        gr.Markdown("# Customer support ticket router\nType a message; the quantized model (M4) picks the team.")
        box = gr.Textbox(label="Message", lines=2)
        btn = gr.Button("Route")
        summary = gr.Markdown()
        with gr.Row():
            cleaned = gr.Textbox(label="Cleaned text")
            fixes = gr.Textbox(label="Spelling corrections")
            code = gr.Textbox(label="Model code")
        table = gr.Dataframe(headers=["Model", "Intent", "Team", "Confidence", "ms"], label="All models side by side")
        outputs = [summary, cleaned, fixes, code, table]
        gr.Examples(EXAMPLES, inputs=box, outputs=outputs, fn=run, run_on_click=True)
        btn.click(run, box, outputs)
        box.submit(run, box, outputs)
    return demo


if __name__ == "__main__":
    build_ui().launch()
