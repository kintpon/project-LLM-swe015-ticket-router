## 2026-11-15 15:44 · P0 · Project setup
- What: git init, folders, .venv (uv, Python 3.12.13), torch 2.11.0+cu128 and other packages, requirements.txt pinned, config.py, src/logutil.py, .gitignore.
- Why: Use Case 2 (ticket router) chosen: clear intent labels (CLINC150) plus a small domain extension, fits the A/B/C/D ladder. Python 3.12 instead of the system 3.14 because torch/gensim/onnxruntime wheels are safer there. CUDA torch because an RTX 3060 was found.
- Files: config.py, src/logutil.py, .gitignore, requirements.txt
- Result: all packages import; logutil rollover checked (305 lines -> _001 has 300, _002 has 5).

## 2026-11-15 15:44 · P0 · Question: how to tick the plan PDF
- What: asked how to tick PDF boxes (draw on PDF vs separate checklist). User chose a separate checklist file.
- Why: avoids installing reportlab; PLAN_CHECKLIST.md is tracked in the project.
- Files: PLAN_CHECKLIST.md
