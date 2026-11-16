# PROGRESS
Phase: P6a done, P6b in progress | Worklog: worklog/worklog_001.md | Checklist: PLAN_CHECKLIST.md
Env: Windows 11, i7-12700H 14c/20t, 31.8 GB RAM, RTX 3060 Laptop (CUDA ok), Python 3.12.13 (.venv via uv), torch 2.11.0+cu128

## Decisions
- Python 3.12 venv; CUDA torch since GPU found; latency still measured on CPU, 4 threads
- Plan ticks in PLAN_CHECKLIST.md (user choice), PDF untouched; token file gitignored, never read
- Dropped 5 exact-duplicate texts from train; spelling vocab = CLINC train words; smoke runs save to models/smoke/
- Spelling: distance <=1 for lexicon words <8 letters, <=2 for longer (user choice, D7)
- User-approved deviations: D7 spelling <=1 short words, D8 5 dup rows out of train, D9 vocab = CLINC train only, D10 10 teams. Ask BEFORE any new deviation from CLAUDE.md
- Spelling rules judged on train+val only (never test). README gets data-checks block from scripts/data_checks.py
- M1 appliance acc is 0.0 (low confidence -> Human agent); reported as is

## File map
config.py | src/logutil.py (rolling log) | src/data.py (splits) | src/teams.py | src/preprocess.py | src/rules.py
src/evaluate.py (shared eval) | src/m1_tfidf.py | src/m2_ffn.py | tests/test_basic.py, test_data.py, test_pipeline.py | docs/decisions.md | data/appliance.csv | scripts/data_checks.py | src/m3_bert.py src/m4_distill.py src/m4_quantize.py | app.py src/make_report.py tests/test_app.py results/failures.csv results/accuracy_vs_latency.png | train_all.py scripts/fill_docs.py (numbers -> docs) scripts/upload_weights.py | docs/architecture.md slides_material.md explain.md
- P5 fresh-env check: clone + README steps (uv, CPU torch) + models copied in (HF repo not uploaded yet); HF download path untested until upload
