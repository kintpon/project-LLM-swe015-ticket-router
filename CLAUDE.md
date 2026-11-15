# CLAUDE.md – SWE015 Homework, Use Case 2: Customer Support Ticket Router

This file is the project specification and your standing instructions. It is loaded automatically in every session, so never paste it back to me and never edit it (ask me if you think it is wrong). Progress and decisions go to `PROGRESS.md`, not here.

**To start or resume:** read `PROGRESS.md` if it exists, then continue with the next unfinished phase in Section 6.

## Context
This is my individual university homework (15% of the grade, graded only 0 / 50 / 100). You build it; I present it,
record a 2-minute demo and may be asked to explain any component. So the code must be small, plain and readable:
functions instead of classes where possible, no abstractions "for later", no features this file doesn't ask for.
A modest accuracy is fine. A wrong, invented or unexplained number is not: fabricated results mean grade 0.

## Goals, non-goals, priorities
**Goal:** pass every grading gate in Section 0 by Saturday 21 Nov 2026 (deadline Sunday 22 Nov), with code I can explain.

**Non-goals** (don't build these, even if they look useful):
- Turkish-language support, extra datasets, or more models than the five in Section 4
- Hyperparameter search or chasing accuracy; one sensible configuration per model is enough
- UI styling, Docker, CI pipelines, deployment to HF Spaces, an API server
- Refactoring working code for elegance

**Priorities:** P0 = everything in Section 0 (must ship). P1 = helpful but cuttable: the accuracy–latency plot, the p95 latency column, the appliance-subset metric, `docs/explain.md`, `docs/code_review.md`.
If time or usage limits get tight, tell me and propose cutting P1 items in that order. Never cut or weaken a P0 item without asking.

## 0. Grading gates (anything missing = 0 or 50 points)
- A working Gradio app that takes NEW typed input and runs the **quantized Transformer**.
- The model ladder evaluated on the **same held-out test set**: classical baseline, own neural net, Transformer, quantized Transformer.
- Results table: task metric + size (MB) + latency (ms, ≥100 runs, hardware stated) for every model.
- At least one technique from each area A/B/C/D, shown in a technique map.
- Architecture diagram that matches the code exactly.
- Repo installable from the README in a fresh environment; one run command; requirements.txt; no weight files in git.
- **Every number comes from a script's output file. Never type, round by hand, estimate or "expect" a metric.**
  If a run fails or a result looks bad, report it as it is.

## 1. Working rules
### Autonomy – work on your own, but ASK FIRST in these cases
Go through the phases (Section 6) automatically. After each phase: run its "done" check for real, commit locally
(`git commit`, never push), update PROGRESS.md, write ONE status line ("P2 done – M2 row added"), and continue.
If a done check fails and you cannot fix it, say so plainly; never mark a phase done with a failing check.

Stop and ask me BEFORE you do any of the following (see "How to ask"):
1. **Change how the system works**: a different model, technique, dataset, label set, routing logic, metric definition, or architecture than this file says, or replacing one approach with another (e.g. "word2vec isn't working, switch to TF-IDF features").
2. **Delete a part completely**: remove a file, a component, a model, a whole function or section, or drop a requirement. Rewriting more than about half of an existing file counts as deleting it. (Splitting a file because of the 500-line limit is NOT deletion – just do it.)
3. Start a run estimated to take **>60 min**, or do anything that needs my accounts/tokens (HF login, git push).

### How to ask (write questions and explanations in Turkish)
- Use the AskUserQuestion tool (multiple choice). One short question: what happened + what you want to change, in 1–2 sentences.
- Options: 2–3 concrete choices; your recommended one first, labelled "(Önerilen)". Each label one short line.
- ALWAYS add a final option: **"Detaylıca anlat"**.
- If I pick "Detaylıca anlat", reply in plain text and for EACH option explain:
  - **Ne yapar:** what exactly changes in the code/files and in the system's behavior
  - **Artıları** and **Eksileri:** accuracy, runtime, complexity, how easy it is for me to explain in the presentation
  - **Ödev etkisi:** does it touch a grading gate in Section 0 (0/50/100 risk)?
  - then one line with your recommendation and why.
  Then ask the same question again with the same options (minus "Detaylıca anlat").
- Do nothing related to the question until I answer; you may continue with independent work meanwhile.

Do NOT ask for: bug fixes, small edits, adding files this file asks for, hyperparameters within this file's limits, package
versions, variable names, and so on. Decide these yourself and note them in PROGRESS.md under "Decisions".

### File size limits
- **Code files: max 500 lines.** If a file would pass 500, move the remaining code into a new, logically named module (e.g. `m3_bert.py` → `m3_bert_train.py` + `m3_bert_infer.py`) and import it. Split at function boundaries. Check with `wc -l` after larger edits. Update the file map in PROGRESS.md and the architecture diagram if affected.
- **Log files: max 300 lines.** Applies to the work journal and to every script log in `logs/`. At 300 lines continue in the next numbered file (`_001` → `_002` …). Script logs go through one shared helper, `src/logutil.py` (a ~20-line line-counting file logger that rolls over), and log one line per epoch, not per step.

### Work journal (I will write my report from it)
- Keep `worklog/worklog_001.md`, `worklog_002.md`, … (committed to git). Write each entry **right after** the step it describes; don't batch entries at the end of a phase.
- One entry per meaningful step: creating or splitting a file, a design choice, a training run, an error and its fix, a question asked + my answer, a result. Skip trivial commands.
- Take the timestamp from the `date` command, never from memory.
- Entry format (≤8 lines):
  ```
  ## 2026-11-15 14:05 · P2 · Trained M2 (word2vec + MLP)
  - What: …
  - Why: … (the reason and the alternative considered – the most important line)
  - Files: src/m2_ffn.py (+120 lines)
  - Result: … (numbers copied from script output, with the file they came from)
  ```
- Append without reading the file (`cat >> worklog/worklog_00N.md <<'EOF'`). Check `wc -l` first; if the entry would pass 300 lines, start the next file and end the old one with `(continued in worklog_00N+1.md)`.

### Token efficiency
- Never print datasets, full logs or whole files. Scripts print only a 1–3 line final summary. Pipe commands through `| tail -n 15`.
- Don't re-read files you just wrote. Don't explore outside the repo. No unrelated refactors or reformatting.
- Every training script has `--smoke` (~300 samples, 1 epoch). Run it before the full run.
- Runs longer than ~2 min: start in the background with output to a log file, check with `tail`.
- Fix errors with minimal diffs. If the same error happens 3 times, stop and tell me in 3 lines.
- Keep `PROGRESS.md` ≤30 lines: current phase, current worklog file, environment, Decisions, file map – so work can resume after /compact or /clear.

### Environment
- In P0 detect OS, Python version, CPU/GPU, and write them to PROGRESS.md. All commands in the README must work on this OS.
- Use a `.venv`. Install the CPU-only torch wheel unless a GPU is found. Resolve a working package set first (gensim, numpy and scipy versions can conflict), then pin exact versions of the direct dependencies in requirements.txt.
- Fix seeds (python, numpy, torch = 42) in every script.

## 2. Data (`src/data.py`)
- HF dataset `clinc/clinc_oos`, config `plus`, official splits: train ≈15.3k, validation ≈3.1k, test ≈5.5k rows. Columns are `text` and `intent` (a ClassLabel with 150 intents + `oos`). License CC BY 3.0: cite the dataset and the CLINC150 paper in the README and slides material.
- Domain data the homework allows: `data/appliance.csv` with 5 new appliance-support intents (`not_draining`,
  `not_heating`, `error_code`, `spare_part_order`, `warranty_claim`), 20 short English messages each = 100 rows.
  Vary the wording, products and length; include model codes (e.g. `DW-6534`) and misspelled product names in some.
  Split per intent 14/3/3 into train/val/test. A test message must not be a near-copy of a train message.
  State in the README that these rows were written with AI assistance.
- Number of classes = whatever the merged label set gives (expected 156). Read it from the data; never hard-code it.
- Build the splits ONCE (seed 42) and cache them to `data/splits/{train,val,test}.parquet` with columns text, label, intent, source (clinc/appliance). Every model reads these files only.
- `src/teams.py`: dict intent → one of ~8 teams (by CLINC domain; appliance intents → "Technical Service" / "Spare Parts" / "Warranty"). `oos` → "Human agent". Small-talk intents (greeting, goodbye, thank_you, …) → "Auto reply".

## 3. Components – one file each, the technique area named in the file's top comment
**B – Classical NLP**: `src/preprocess.py`, `src/rules.py`, `src/m1_tfidf.py`
- Order inside `preprocess(text)`: (1) extract model codes case-insensitively (`[A-Za-z]{2,3}-?\d{3,5}`) and remove them from the text, (2) lowercase, (3) URLs/emails → placeholder token, (4) squeeze repeated chars (`soooo`→`soo`), (5) spelling correction, (6) whitespace. Returns the clean text + codes + list of corrections.
- Spelling correction: own ~15-line Levenshtein (dynamic programming, no library). Correct a token to a product-lexicon word (dishwasher, washing, machine, oven, fridge, refrigerator, dryer, freezer, microwave…) only if the token has ≥5 letters, is NOT in the training vocabulary, and the distance is ≤2. This guard keeps real words like "over" or "even" from turning into "oven".
- ELIZA-style regex rules for greetings/thanks/small talk → canned reply. The rules are used **only in the app**, before the models. The evaluation in Section 4 measures the models alone, so the comparison stays fair.
- The same `preprocess` is applied to every model's input, in training, evaluation and the app.
- **M1**: tf-idf (word 1–2 grams) + LogisticRegression.

**A – Neural network**: `src/m2_ffn.py`
- **M2**: gensim word2vec trained on the train texts (dim 100) → mean vector per message (zero vector if no known word) → PyTorch MLP (100 → 256 → n_classes, ReLU, dropout 0.2). Write the training loop by hand: `torch.optim.SGD(momentum=0.9)`, cross-entropy, `loss.backward()`, early stopping on val accuracy.

**C – Transformer**: `src/m3_bert.py`
- **M3 (teacher)**: fine-tune `google-bert/bert-base-uncased` (≈110M parameters) for classification. max_length 32, lr 3e-5, batch 32, 3 epochs, keep the best epoch by val accuracy.
- The script must also run unchanged on Google Colab (GPU); add a short README section (upload repo, run, download `models/`).
- No GPU: estimate the full time from the smoke run. If >60 min, ask me (CPU vs. Colab) per Section 1.

**D – Efficient model**: `src/m4_distill.py`, `src/m4_quantize.py` – the homework asks for "distillation into a smaller student, then quantization"
- **M3s (student, fp32)**: `distilbert/distilbert-base-uncased` (≈67M parameters) trained with knowledge distillation from M3: loss = 0.5·CE(labels) + 0.5·T²·KL(student/T ‖ teacher/T), T = 2, 5 epochs. Compute the teacher logits for the train set once and cache them.
- **M4 (student, INT8)**: export M3s to ONNX, then `onnxruntime.quantization.quantize_dynamic(..., weight_type=QInt8)`. Run it with ONNX Runtime on CPU.
  Why ONNX Runtime and not `torch.quantization.quantize_dynamic`: PyTorch has deprecated that API and plans to remove it, and ONNX Runtime is on the homework's own technique list. If the ONNX export fails with the default exporter, try the legacy exporter (`dynamo=False`) before asking me.
- Also save the un-quantized ONNX file of M3s; M4's INT8 file must load without any PyTorch model.

## 4. Routing + evaluation (`src/evaluate.py` – ONE shared function for all models)
- Each model exposes `predict_proba(list_of_texts) → array`. Confidence = max probability.
- Route = "Human agent" if predicted intent == `oos` OR confidence < threshold.
- Threshold: per model, tuned on **val only**, grid 0.00–0.95 step 0.05, maximizing (in-scope accuracy + oos recall) / 2. Save the thresholds to `results/thresholds.json`; the app reads them from there.
- Metrics on the test split, after routing:
  - **in-scope accuracy**: share of in-scope messages that get the correct intent (one sent to "Human agent" counts as wrong)
  - **oos recall**: share of `oos` messages routed to "Human agent"
  - accuracy on the appliance test rows (30 rows – report it with that caveat)
- **Size (MB)**: total bytes of every file the model needs at inference (M1: vectorizer + classifier; M2: word2vec + net; M3/M3s: weights; M4: the INT8 .onnx). Tokenizer files excluded, and say so under the table.
- **Latency (ms)**: one message at a time, end to end (preprocess → route), on CPU, 4 threads for torch and ONNX Runtime. 10 warm-up calls, then 200 test messages (the same 200 for every model); report mean and p95.
- Hardware line: CPU model, cores, RAM, OS, Python, torch and onnxruntime versions.
- Outputs, all written by code: `results/results.csv` (one row per model; a re-run replaces that model's row), `results/results.md` (table + hardware line + notes), `results/failures.csv` (10 wrong M4 predictions: text, true, predicted, confidence), `results/accuracy_vs_latency.png` (one simple scatter).
- Rows: M1 tf-idf+LR · M2 word2vec+MLP · M3 BERT (fp32) · M3s DistilBERT student (fp32) · M4 student INT8. The ladder for the slides is M1 → M2 → M3 → M4; M3s shows what distillation alone and quantization alone each cost.

## 4b. Tests (`tests/`, run with `pytest -q`; keep the whole suite under ~60 s without the Transformer models)
Three small levels; write each test file in the phase that creates the code it tests.
- **Unit** (`test_basic.py`, P1): Levenshtein on known pairs; model-code extraction (upper/lower case, with/without dash); the "over"/"oven" guard; one small-talk rule; routing with a threshold; `preprocess("")` and a 2,000-character input don't crash.
- **Data integrity** (`test_data.py`, P1): no text appears in both train and test; every intent has a team; label ids are identical across the three splits; the appliance rows are 70/15/15; class count equals the number of distinct intents.
- **Pipeline** (`test_pipeline.py`, P2 onward): on a 300-row sample, train M1 → evaluate → results row is written, and running evaluate twice leaves ONE row per model (no duplicates). From P4: the app handler returns a route for the 4 examples plus empty input, a message that is only a model code, and a Turkish sentence (expected: Human agent or a low-confidence fallback, never a crash).
Tests check behaviour, not accuracy numbers. A failing test blocks the phase's "done".

## 4c. Decision records (`docs/decisions.md`)
Short ADRs so I can defend the design. One entry (≤10 lines) per real decision: Context · Options considered · Decision · Consequence. Pre-fill these in the phase where each becomes real: why Use Case 2; why CLINC150 `plus`; word2vec mean-pooling for M2; BERT teacher → DistilBERT student; ONNX Runtime INT8 instead of PyTorch quantization; confidence-threshold routing tuned on validation. Every question you ask me (Section 1) becomes an entry with my answer as the decision. The worklog records what happened; this file records why the design is what it is.

## 5. App (`app.py`, Gradio Blocks)
- Text box + 4 clickable examples: an easy one, one with a typo + model code, one out-of-scope, one small talk.
- Flow: rules first (small talk → canned reply, "routed by: rules"); otherwise preprocess → **M4 decides the route**.
- Shows: cleaned text, corrections made, extracted model code, routed team + intent + confidence, "routed by: rules / model / human-agent fallback", and a table with all models side by side (intent, confidence, ms).
- Load everything once at startup. If `models/` is missing, download it from the HF Hub repo named in `config.py` (public repo, no token needed).
- Input edge cases: empty text → a friendly prompt, no model call; text longer than 500 characters is truncated before the models.
- Keep the UI plain; no styling work.

## 6. Phases (each ends with a "done" check that you actually run)
- **P0** `git init`, structure, `.venv`, requirements.txt, `.gitignore` (models/, data/splits/, logs/, .venv), `config.py` (paths, seed, HF model repo `kintpon/swe015-ticket-router`, GitHub remote `https://github.com/kintpon/project-LLM.git`), `src/logutil.py`, first worklog entry (setup + why Use Case 2), PROGRESS.md.
  Done: packages import in the venv.
- **P1** data.py, appliance.csv, teams.py, preprocess.py, rules.py, evaluate.py, M1, `tests/test_basic.py`, `tests/test_data.py` (Section 4b).
  Done: tests pass; M1 row in results.csv; every intent has a team.
- **P2** M2, `tests/test_pipeline.py`. Done: tests pass; M2 row in results.csv.
- **P3** M3, M3s, M4. Done: three new rows; the M4 file is smaller than M3s; M4 and M3s agree on ≥90% of test predictions (if not, something is broken – investigate before continuing).
- **P4** app.py, failures.csv, plot. Done: call the app's handler function directly on the 4 examples and get sensible output for each; the edge-case tests in Section 4b pass; the app starts.
- **P5** Docs and handoff:
  - `docs/architecture.md`: Mermaid diagram written from the real call path (input → rules → preprocess → M1, M2, M3 fp32, M3s fp32, M4 INT8/ONNX Runtime → threshold routing → Gradio UI; all on CPU), with the file name on each box
  - `docs/slides_material.md`: technique map (component | area A/B/C/D | technique | file), data summary (sources, rows per split, preprocessing), how each model was trained, the results table copied by script from results.md, 3 failure examples with a one-line reason each, where each model wins and loses, a draft AI-usage slide (I will edit it)
  - `docs/explain.md`: for each component, 3–5 lines in plain Turkish on what it does and why, plus the 2 questions an instructor is most likely to ask, with short answers
  - `scripts/upload_weights.py` for the HF Hub (I run `huggingface-cli login` myself)
  - README: setup, `python train_all.py` (all training + evaluation in order), `python app.py`, the Colab note, dataset citations (CLINC150 paper + HF link), AI-assistance note
  - Done: create a second fresh venv, follow the README literally, and `python app.py` starts.
- **P6a** Self-review, written to `docs/code_review.md` (one table: file · finding · severity · fixed?). Read each source file once with fresh eyes and check:
  - Secrets: no token, password or personal path in code, logs, worklog or git history (`git log -p | grep -i "hf_"` must be empty)
  - Leakage: the test split is never used for training, early stopping, threshold tuning or vocabulary building
  - Correctness: empty/odd inputs, label-id mapping identical in training, evaluation and the app, same preprocessing everywhere
  - Readability: names say what things are; every non-obvious line has a short comment; no dead code or unused files
  Fix critical findings with minimal diffs (the "ask first" rules still apply).
- **P6b** Release checklist (stop and ask before the push itself):
  - [ ] `pytest -q` passes
  - [ ] `git ls-files` shows no file larger than 5 MB and nothing from models/, logs/, data/splits/
  - [ ] A fresh clone in a new folder + the README steps → `python app.py` starts and downloads the weights from the HF repo
  - [ ] `results/results.md` is complete: metric, size, latency and hardware for every row
  - [ ] Create the git tag `submission` so there is a known-good version to return to
- **P6c** Final audit: a table "requirement in Section 0 → file/evidence → OK/missing"; check that no code file is >500 lines and no log/worklog file is >300; check that every number in `docs/` equals the one in `results/results.csv`. Fix only what is missing, then give me a ≤10-line summary in Turkish with what I still have to do myself.
