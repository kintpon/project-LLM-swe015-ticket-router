## 2026-11-15 15:44 · P0 · Project setup
- What: git init, folders, .venv (uv, Python 3.12.13), torch 2.11.0+cu128 and other packages, requirements.txt pinned, config.py, src/logutil.py, .gitignore.
- Why: Use Case 2 (ticket router) chosen: clear intent labels (CLINC150) plus a small domain extension, fits the A/B/C/D ladder. Python 3.12 instead of the system 3.14 because torch/gensim/onnxruntime wheels are safer there. CUDA torch because an RTX 3060 was found.
- Files: config.py, src/logutil.py, .gitignore, requirements.txt
- Result: all packages import; logutil rollover checked (305 lines -> _001 has 300, _002 has 5).

## 2026-11-15 15:44 · P0 · Question: how to tick the plan PDF
- What: asked how to tick PDF boxes (draw on PDF vs separate checklist). User chose a separate checklist file.
- Why: avoids installing reportlab; PLAN_CHECKLIST.md is tracked in the project.
- Files: PLAN_CHECKLIST.md

## 2026-11-15 15:51 · P1 · Wrote data/appliance.csv
- What: 100 appliance-support messages, 5 intents x 20, varied wording/length, model codes and typos in some.
- Why: homework allows domain data; hand-written (AI-assisted) so wording differs between rows.
- Files: data/appliance.csv (+101 lines)

## 2026-11-15 15:52 · P1 · data.py + teams.py
- What: src/data.py builds splits (CLINC plus + appliance 14/3/3) to data/splits/*.parquet; src/teams.py maps every intent to a team.
- Why: found 2 train/test and 3 train/val exact duplicate texts in CLINC; removed them from train so the test is clean (alternative: keep and fail the no-overlap check).
- Files: src/data.py, src/teams.py
- Result: python -m src.data printed train/val/test sizes and 156 classes (see run output).

## 2026-11-15 15:52 · P1 · preprocess.py + rules.py
- What: preprocess (code extraction, lowercase, url/email tokens, repeat squeeze, own Levenshtein spelling fix, whitespace) and 5 ELIZA-style rules.
- Why: lexicon correction only for tokens >=5 letters, not in vocab, distance <=2 (keeps "over"/"even" safe); rules stay app-only so model comparison is fair.
- Files: src/preprocess.py, src/rules.py

## 2026-11-15 15:55 · P1 · evaluate.py + M1 trained and evaluated
- What: src/evaluate.py (route, threshold on val, metrics, size, latency, results.csv/md/thresholds.json), src/m1_tfidf.py (tf-idf 1-2gram + LogReg C=10), smoke then full run.
- Why: one shared evaluate() so all models are measured the same way; vocab for the spelling guard = train words seen >=4 times, so typos that occur 1-3 times in train stay correctable.
- Files: src/evaluate.py, src/m1_tfidf.py, src/preprocess.py (get_vocab)
- Result (results/results.csv): M1 in-scope 0.7973, oos recall 0.9110, appliance acc 0.0, threshold 0.45, 40.8 MB, latency mean 0.45 ms.

## 2026-11-15 15:55 · P1 · Finding: M1 appliance accuracy is 0.0
- What: inspected predictions: M1 often names the right appliance intent but with confidence 0.1-0.2 (only 14 train rows per appliance class), below threshold 0.45, so they are routed to Human agent.
- Why: not a bug; threshold was tuned on val where CLINC rows dominate. Reported as is, no tuning to hide it.
- Result: noted for the report; M2/M3 will show whether this changes.

## 2026-11-15 15:55 · P1 · Tests
- What: tests/test_basic.py (6 tests) and tests/test_data.py (6 tests).
- Result: pytest -q -> 12 passed; every intent has a team (156 classes, 10 teams).

## 2026-11-15 16:02 · P1 · Bug hunt (user asked) – 6 fixes
- What: (1) spelling fix corrupted real words (bridge->fridge, vaccine->machine; 53 wrong on CLINC val+test): lexicon back to the spec list, vocab = all CLINC train words -> 20 wrong. (2) repeat-squeeze changed numbers (1000->100): now letters only. (3) small-talk rules swallowed real requests ("thanks my oven is still broken"): rules must match the whole message. (4) --smoke overwrote the real M1 files: smoke now saves to models/smoke/. (5) hardware line had no RAM. (6) test hard-coded 156; logutil path depended on cwd.
- Why: (1) was caused by my own additions in P1 (extra lexicon words, count>=4 vocab), measured before and after; alternative d<=1 for short words changes the spec, so it is asked to the user.
- Files: src/preprocess.py, src/rules.py, src/evaluate.py, src/m1_tfidf.py, src/logutil.py, tests/ (+3 tests)
- Result: pytest 14 passed. M1 retrained (results/results.csv): in-scope 0.7991, oos recall 0.9080, thr 0.45, 40.9 MB, 1.01 ms. The earlier M1 row (0.7973) is replaced.

## 2026-11-15 16:05 · P1 · Question: spelling distance for short words (user chose <=1)
- What: asked whether to keep "distance <=2" from CLAUDE.md or use <=1 for lexicon words shorter than 8 letters. User chose <=1 for short words. One line changed in correct_token; M1 retrained.
- Why: 2 edits on a 4-5 letter word change half of it (lover->oven, drawer->dryer, flashing->washing); alternative was to keep <=2 and report 20 wrong corrections.
- Files: src/preprocess.py, tests/test_basic.py
- Result: wrong corrections on CLINC val+test 20 -> 4 (refrigerated, ridge, microwaves); appliance typos still fixed; pytest 14 passed. M1 (results/results.csv): in-scope 0.7980, oos recall 0.9070, thr 0.45, 40.9 MB, 0.43 ms.
