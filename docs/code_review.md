# Code review (P6a)

Each source file was read once with fresh eyes. Checks run on 2026-11-16:

- **Secrets:** `git log -p | grep -i "hf_<20+ chars>"` finds 0 matches; `git grep` finds no personal path in tracked files.
- **Leakage:** test split is never used for training, early stopping, threshold tuning or vocabulary building. Thresholds use val (`src/evaluate.py` `tune_threshold`), best epoch / early stopping use val (`src/m2_ffn.py`, `src/m3_bert.py`), tf-idf, word2vec and the spelling vocabulary are built from train only (`src/m1_tfidf.py`, `src/m2_ffn.py`, `src/preprocess.py` `get_vocab`).
- **Label mapping:** training, evaluation and the app all use `label_names()` from `src/data.py` (label id = position in one list); `tests/test_data.py` checks ids are identical across splits.
- **Same preprocessing:** every model's `predict_proba` calls `preprocess()`; the app calls the same functions.
- **Readability:** no TODO/FIXME or debug prints; no unused files found.

| File | Finding | Severity | Fixed? |
|---|---|---|---|
| scripts/upload_weights.py | `python scripts/upload_weights.py` (the README command) failed with "No module named 'config'" | High | Yes (df0e8a0) |
| src/evaluate.py | results.md said 30 appliance test rows; real count is 15 | Medium | Yes (P3 bug hunt) |
| src/m3_bert.py, src/m4_distill.py | classification head was created before the seed was set | Medium | Yes (P3 bug hunt) |
| results/results.md | M3/M3s/M4 rows come from the run before the seeding fix, so they are not exactly reproducible; no note said so | Medium | Yes, note added (23152c9) |
| src/m3_bert.py, src/m4_distill.py, docs | text said KL(student ‖ teacher); code computes KL(teacher ‖ student) (standard Hinton loss) | Medium | Yes, text corrected (23152c9) |
| src/m3_bert.py | cached teacher logits could stay from an older teacher after re-training M3 | Medium | Yes (P3 bug hunt) |
| src/m2_ffn.py | epoch log showed the last batch loss, not the epoch mean | Low | Yes (P3 bug hunt) |
| README (Colab) + src/evaluate.py | training M3 on Colab would measure its latency on Colab's CPU while the other rows come from the laptop, under one hardware line; on Linux the CPU name can be empty | Medium-low | No (current results were all measured locally; user decision) |
| src/preprocess.py | model-code regex from the spec also catches flight numbers ("flight dl123", 66 CLINC rows) | Low | No (spec behaviour; user decision) |
| src/preprocess.py | an email like `abc1234@x.com` is split by code extraction before the email rule | Low | No (spec order; user decision) |
| app.py | `ensure_models` only checks `m4_int8.onnx`; a partly missing `models/` is not re-downloaded | Low | No (user decision) |
| app.py | first start downloads the CLINC dataset to build `data/splits/` (needs internet) | Info | Already stated in README |
