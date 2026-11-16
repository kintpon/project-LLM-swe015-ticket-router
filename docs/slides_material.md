# Slides material – Customer Support Ticket Router (SWE015, Use Case 2)

Blocks between `<!-- …-START -->` and `<!-- …-END -->` are written by `scripts/fill_docs.py` from the result files. Do not edit them by hand; re-run the script instead. Prose outside the blocks contains no metric values on purpose — read them from the tables.

## 1. Technique map

| Component | Area | Technique | File |
|---|---|---|---|
| Text cleaning | B – Classical NLP | regex: model-code extraction, URL/email tokens, repeated letters | src/preprocess.py |
| Spelling correction | B – Classical NLP | own Levenshtein distance (dynamic programming) towards a product lexicon | src/preprocess.py |
| Small-talk replies | B – Classical NLP | ELIZA-style regex rules, whole-message match (app only) | src/rules.py |
| M1 | B – Classical NLP | tf-idf (word 1–2 grams) + logistic regression | src/m1_tfidf.py |
| M2 | A – Neural network | word2vec (gensim, dim 100) mean vector → PyTorch MLP 100→256→classes, hand-written SGD loop, early stopping | src/m2_ffn.py |
| M3 | C – Transformer | fine-tuned bert-base-uncased (teacher) | src/m3_bert.py |
| M3s | D – Efficient model | knowledge distillation into DistilBERT (student) | src/m4_distill.py |
| M4 | D – Efficient model | ONNX export + ONNX Runtime dynamic INT8 quantization | src/m4_quantize.py |
| Routing | – | confidence threshold tuned on validation; oos / low confidence → Human agent | src/evaluate.py, src/teams.py |
| Demo | – | Gradio Blocks app, M4 decides the route | app.py |

## 2. Data

- **CLINC150 `plus`** (Larson et al., 2019, CC BY 3.0) from the HF Hub `clinc/clinc_oos`: 150 intents + `oos`, official train/validation/test splits.
- **Appliance support rows**: `data/appliance.csv`, 5 new intents (`not_draining`, `not_heating`, `error_code`, `spare_part_order`, `warranty_claim`), 20 short English messages each, written with AI assistance. Per intent 14 train / 3 val / 3 test.
- Clean-up: train rows whose text also appears in val or test are dropped; val and test stay official (decision D8).
- Preprocessing (same for every model, in training, evaluation and the app): extract model codes such as `DW-6534` → lowercase → URL/email placeholders → squeeze repeated letters → spelling fix towards product words (only unknown tokens with ≥5 letters) → whitespace.
- Intents map to teams by CLINC domain; appliance intents go to Technical Service / Spare Parts / Warranty, `oos` to Human agent, small talk to Auto reply.

<!-- DATA-START -->
| Split | CLINC rows | Appliance rows | Total |
|---|---|---|---|
| train | 15245 | 70 | 15315 |
| val | 3100 | 15 | 3115 |
| test | 5500 | 15 | 5515 |

Classes (intents incl. `oos`): 156. Teams: 10.
<!-- DATA-END -->

## 3. How each model was trained

All scripts use seed 42 and are run in order by `python train_all.py`. Thresholds are tuned on validation only.

- **M1** – tf-idf on word 1–2 grams (sublinear tf), logistic regression with C = 10. Trains in seconds on CPU.
- **M2** – word2vec trained on our train texts (dim 100, window 5, 30 epochs). A message is the mean of its word vectors (zero vector if no word is known). MLP 100 → 256 → classes with ReLU and dropout 0.2, plain SGD with momentum 0.9 (lr 0.1, batch 64), loss written out with `loss.backward()`; early stopping after 15 epochs without a better validation accuracy.
- **M3** – `google-bert/bert-base-uncased` fine-tuned for classification: max length 32, lr 3e-5, batch 32, 3 epochs, AdamW with linear warm-up; best epoch kept by validation accuracy. Trained on GPU.
- **M3s** – `distilbert/distilbert-base-uncased` trained from M3: loss = 0.5·CE(labels) + 0.5·T²·KL(teacher/T ‖ student/T), T = 2, 5 epochs, lr 5e-5. Teacher logits for the train set are computed once and cached.
- **M4** – M3s exported to ONNX (legacy exporter), then `quantize_dynamic(..., weight_type=QInt8)`. Runs with ONNX Runtime only, no PyTorch model needed.

## 4. Results (test split, after routing)

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

Plot: `results/accuracy_vs_latency.png`.

## 5. Failure examples (M4)

10 random wrong M4 test predictions (seed 42), from `results/failures.csv`:

<!-- FAILURES-START -->
| Text | True | Predicted | Confidence | Routed to |
|---|---|---|---|---|
| where can i buy a replacement filter for my dishwasher | spare_part_order | oos | 0.2268 | Human agent |
| give me an eta on when i will arrive at my destination | distance | flight_status | 0.3403 | Human agent |
| please note vacuuming on my to do list | todo_list_update | todo_list_update | 0.488 | Human agent |
| am i close to arriving at the post office | distance | directions | 0.4565 | Human agent |
| can i reserve tables at the new york restaurant | accept_reservations | restaurant_reservation | 0.4892 | Human agent |
| my favorite, beef tacos at the buffet | restaurant_reviews | meal_suggestion | 0.2166 | Human agent |
| what do you think about me calling you cora | change_ai_name | change_ai_name | 0.5607 | Human agent |
| how many points do i have to cash | redeem_rewards | redeem_rewards | 0.4675 | Human agent |
| when will i get to orlando | distance | distance | 0.5519 | Human agent |
| is it possible i have a meeting with kim today | meeting_schedule | meeting_schedule | 0.4918 | Human agent |
<!-- FAILURES-END -->

Three to show, with a one-line reason each:

1. *"where can i buy a replacement filter for my dishwasher"* (true `spare_part_order`, predicted `oos`) – the appliance intents have only 14 training rows each, so the student has seen few ways to ask for a part.
2. *"give me an eta on when i will arrive at my destination"* (true `distance`, predicted `flight_status`) – "eta" and "arrive" are typical flight words; the two intents overlap in meaning.
3. *"when will i get to orlando"* (true `distance`, predicted `distance`) – the intent is right but the confidence is below M4's threshold, so it goes to a human agent and counts as wrong. Half of the 10 sampled failures are of this kind: a safe fallback, not a wrong team.

## 6. Where each model wins and loses (read the numbers from the table above)

- **M1 tf-idf + LogReg** – strong baseline for its cost and needs no GPU. Loses on paraphrases with no shared words, and its large vocabulary of bigrams makes the file bigger than M2's.
- **M2 word2vec + MLP** – the smallest and fastest model. Mean-pooling throws away word order and word2vec is trained on only our train texts, so its accuracy is the lowest; its threshold had to be high to keep oos recall.
- **M3 BERT** – the best in-scope accuracy, but the largest and slowest model on CPU.
- **M3s DistilBERT student** – almost the same accuracy as the teacher at about half the latency and a smaller file: distillation alone costs little.
- **M4 INT8** – the size and latency winner among the Transformers, with a small accuracy drop against M3s: quantization alone costs a little accuracy. It is the model the app uses.
- **Appliance rows** – only 15 test rows (3 per intent). Most models give these rows a confidence below their threshold and send them to the Human agent, so appliance accuracy is low for every model; this is reported as it is.

## 7. AI usage (draft – edit before presenting)

- I used Claude Code (Anthropic) as a coding assistant, guided by a written specification (`CLAUDE.md`) that I wrote and that fixed the models, data, metrics and phases in advance.
- The assistant wrote most of the code, tests and documents; I reviewed each phase, answered every design question it asked (recorded in `docs/decisions.md`), and ran the demo myself.
- The 100 appliance messages in `data/appliance.csv` were written with AI assistance.
- Every number in the slides is produced by a script and copied by `scripts/fill_docs.py`; none was typed by hand.
- What I checked myself: …
