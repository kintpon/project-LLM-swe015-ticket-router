# Architecture

Everything runs on CPU at inference time (torch and ONNX Runtime set to 4 threads). Each box names the file that does the work.

## App: one message (`python app.py`)

```mermaid
flowchart TD
    U["User types a message<br/>Gradio Blocks UI · app.py build_ui()"] --> H["handle(): strip, cut to 500 chars<br/>empty → 'Please type a message first.', no model call<br/>app.py"]
    H --> P["preprocess(): model codes out, lowercase, URL/email tokens,<br/>squeeze repeats, Levenshtein spelling fix<br/>src/preprocess.py"]
    P --> R{"match_rule(): ELIZA-style regex,<br/>whole message must match<br/>src/rules.py"}
    R -- "small talk" --> AR["Canned reply<br/>team = Auto reply, routed by: rules"]
    R -- "no rule" --> MODELS
    subgraph MODELS["All five models get the raw text; each calls preprocess() itself"]
        M1["M1 tf-idf + LogReg<br/>src/m1_tfidf.py"]
        M2["M2 word2vec + MLP<br/>src/m2_ffn.py"]
        M3["M3 BERT-base fp32<br/>src/m3_bert.py"]
        M3s["M3s DistilBERT student fp32<br/>src/m3_bert.py make_predict_proba"]
        M4["M4 DistilBERT student INT8<br/>ONNX Runtime · src/m4_quantize.py"]
    end
    MODELS --> RT["route(): argmax intent, confidence = max prob<br/>oos or confidence < threshold → Human agent<br/>else intent → team (src/teams.py)<br/>src/evaluate.py · thresholds from results/thresholds.json"]
    RT --> D["M4's result decides: team, intent, confidence,<br/>routed by: model / human-agent fallback<br/>app.py"]
    RT --> T["Side-by-side table: intent, team, confidence, ms<br/>for M1, M2, M3, M3s, M4"]
    D --> UI["Gradio UI shows cleaned text, corrections, model code,<br/>route and table"]
    T --> UI
    AR --> UI
```

`preprocess()` is called in `handle()` only to show the cleaned text, corrections and model code in the UI; the rules then look at the original text. Model weights load once at startup (`load_models()` in app.py); if `models/m4_int8.onnx` is missing they are downloaded from the HF Hub repo in `config.py`.

## Training and evaluation (`python train_all.py`)

```mermaid
flowchart LR
    DS["CLINC150 'plus' (HF) +<br/>data/appliance.csv<br/>src/data.py → data/splits/*.parquet"] --> M1T["M1 train<br/>src/m1_tfidf.py"]
    DS --> M2T["M2 train: word2vec, then MLP with SGD<br/>src/m2_ffn.py"]
    DS --> M3T["M3 fine-tune BERT<br/>src/m3_bert.py"]
    M3T -- "teacher logits (cached)" --> M3sT["M3s distillation<br/>src/m4_distill.py"]
    M3sT --> Q["ONNX export + quantize_dynamic INT8<br/>src/m4_quantize.py"]
    M1T & M2T & M3T & M3sT & Q --> EV["evaluate(): threshold on val, metrics on test,<br/>size, latency<br/>src/evaluate.py"]
    EV --> RES["results/results.csv, results.md, thresholds.json"]
    RES --> REP["failures.csv + accuracy_vs_latency.png<br/>src/make_report.py"]
    RES --> DOC["numbers copied into docs<br/>scripts/fill_docs.py"]
```

The small-talk rules are not part of the evaluation: the models are measured alone so the comparison is fair.
