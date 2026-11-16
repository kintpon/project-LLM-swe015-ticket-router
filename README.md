# SWE015 Customer Support Ticket Router

Routes a customer message to the right support team. Five models are compared on the same test set, from a classical baseline to a distilled, INT8-quantized Transformer; the Gradio app lets the quantized model (M4) decide the route for new typed messages.

- Architecture diagram: [docs/architecture.md](docs/architecture.md)
- Results: [results/results.md](results/results.md) · design decisions: [docs/decisions.md](docs/decisions.md)
- Slides material: [docs/slides_material.md](docs/slides_material.md)

## Setup (Windows, macOS or Linux)

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/) (`pip install uv`). From the repo folder:

```
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
```

Then activate the environment:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS / Linux: `source .venv/bin/activate`

All commands below are run inside the activated environment. Without uv: create a Python 3.12 venv with `python -m venv .venv`, activate it and run `pip install -r requirements.txt`.

## Run the app

```
python app.py
```

Open the address it prints (http://127.0.0.1:7860). On the first start the trained weights are downloaded from the public Hugging Face repo [`kintpon/swe015-ticket-router`](https://huggingface.co/kintpon/swe015-ticket-router) into `models/` (no account needed), and the CLINC150 dataset is downloaded once to build `data/splits/`.

## Train and evaluate everything

```
python train_all.py
```

Runs in order: build the splits (`src/data.py`), data checks, M1, M2, M3 (BERT), M3s (distillation), M4 (ONNX INT8), then writes `results/` and copies the numbers into the docs. M3 and M3s need a GPU to finish in reasonable time; on CPU use Colab (below). `python train_all.py --smoke` runs every model on 300 rows for 1 epoch as a quick check and does not write results.

Tests: `pytest -q` (the app tests are skipped when `models/` is missing).

## Training on Google Colab (GPU)

1. Runtime → Change runtime type → GPU.
2. In a cell:
   ```
   !git clone https://github.com/kintpon/project-LLM.git
   %cd project-LLM
   !pip install -r requirements.txt
   !python train_all.py
   ```
3. Download the weights: `!zip -r models.zip models -x "models/smoke/*"`, then download `models.zip` from the file panel and unzip it into the repo folder on your computer.

The scripts pick the GPU automatically when one is found; latency is always measured on CPU.

## Results

Written by the evaluation scripts and copied here by `scripts/fill_docs.py`:

<!-- RESULTS-START -->
| Model | In-scope acc | OOS recall | Appliance acc (15 rows) | Threshold | Size (MB) | Latency mean (ms) | Latency p95 (ms) |
|---|---|---|---|---|---|---|---|
| M1 tf-idf + LogReg | 0.7980 | 0.9070 | 0.0000 | 0.45 | 40.9 | 0.43 | 0.61 |
| M2 word2vec + MLP | 0.6726 | 0.8770 | 0.1333 | 0.90 | 2.6 | 0.14 | 0.36 |
| M3 BERT-base (fp32) | 0.9094 | 0.8720 | 0.0000 | 0.25 | 438.4 | 61.43 | 70.90 |
| M3s DistilBERT student (fp32) | 0.9079 | 0.9100 | 0.0000 | 0.65 | 268.3 | 31.76 | 34.29 |
| M4 DistilBERT student INT8 (ONNX Runtime) | 0.8930 | 0.9320 | 0.0000 | 0.65 | 67.5 | 12.01 | 13.05 |

Hardware: 12th Gen Intel(R) Core(TM) i7-12700H (20 logical cores, 31.8 GB RAM), Windows-11-10.0.26200-SP0, Python 3.12.13, torch 2.11.0+cu128, onnxruntime 1.32.0; latency on CPU, 4 threads

Notes: size = all files needed at inference, tokenizer files excluded. Latency = one message at a time, preprocess -> route, 10 warm-up calls then 200 test messages (same for all models). Metrics are measured on the models alone (small-talk rules not applied). Appliance accuracy covers only 15 test rows (3 per appliance intent), so treat it with caution. Reproducibility: the M3, M3s and M4 rows come from a training run made before a seeding fix (the new classification head was created before the seed was set; see worklog_001.md), and GPU training is not bit-exact, so re-running train_all.py gives different numbers for these rows.
<!-- RESULTS-END -->

## Uploading the weights (maintainer only)

```
huggingface-cli login
python scripts/upload_weights.py
```

Weights are never committed to git.

## Data and citations

- **CLINC150** (`plus` config), CC BY 3.0: Stefan Larson, Anish Mahendran, Joseph J. Peper, Christopher Clarke, Andrew Lee, Parker Hill, Jonathan K. Kummerfeld, Kevin Leach, Michael A. Laurenzano, Lingjia Tang, Jason Mars. *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction.* EMNLP-IJCNLP 2019. Dataset: https://huggingface.co/datasets/clinc/clinc_oos
- **Appliance rows** (`data/appliance.csv`, 100 messages, 5 intents): written for this project **with AI assistance**.
- Pretrained models: `google-bert/bert-base-uncased`, `distilbert/distilbert-base-uncased` from the Hugging Face Hub.

## AI assistance

The code, tests and documents were written with an AI coding assistant (Claude Code) following a written specification; design questions and their answers are recorded in `docs/decisions.md`. Every number in this repo is produced by a script.

<!-- DATA-CHECKS-START -->
### Data checks and limitations (written by scripts/data_checks.py)

Number of texts that appear in two splits (compared case-insensitively):

| Splits | Official CLINC, raw | Our splits, raw | Our splits, after preprocessing |
|---|---|---|---|
| train & test | 2 | 0 | 15 |
| train & val | 3 | 0 | 12 |
| val & test | 0 | 0 | 5 |

Our splits = official CLINC splits with the train rows that also appear in val/test removed, plus the 100 appliance rows.

Spelling correction on 18345 CLINC rows of train+val: 2 tokens changed (2 distinct). Words in the CLINC training vocabulary are never changed, so all of them come from validation rows. Each one is a wrong correction (CLINC has no deliberate typos):

- `refrigerated` -> `refrigerator` (1x)
- `ridge` -> `fridge` (1x)
<!-- DATA-CHECKS-END -->
