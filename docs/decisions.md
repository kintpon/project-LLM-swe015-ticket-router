# Decision records

## D1 · Why Use Case 2 (Customer Support Ticket Router)
Context: homework offers several use cases. Options: other use cases. Decision: ticket router. Consequence: labelled intent data (CLINC150) exists and the routing result is easy to demo.

## D2 · How to tick the plan checklist
Context: the plan PDF has checkboxes. Options: draw ticks on the PDF (needs reportlab) or keep a separate checklist file. Decision (user): separate file, PLAN_CHECKLIST.md. Consequence: no extra package; the PDF stays untouched.

## D3 · Python 3.12 venv with CUDA torch
Context: system Python is 3.14 and the machine has an RTX 3060. Options: 3.14 as is, 3.12 via uv. Decision: 3.12 venv, torch cu128. Consequence: BERT can likely be trained locally; latency is still measured on CPU.

## D4 · Why CLINC150 `plus`
Context: need many intents plus an out-of-scope class. Options: CLINC150 plus, small custom set. Decision: CLINC150 plus (150 intents + oos, official splits) plus 100 appliance rows. Consequence: 156 classes; oos gives the "Human agent" fallback a real test.

## D5 · Confidence-threshold routing tuned on validation
Context: messages the model is unsure about should go to a person. Options: always take the top class; threshold. Decision: route to Human agent if predicted oos or confidence < threshold; threshold per model tuned on val only. Consequence: honest oos recall; but low-confidence classes (appliance, 14 rows each) can be sent to a human (seen in M1).

## D6 · Small data clean-up and spelling vocabulary
Context: CLINC has 2 train/test and 3 train/val exact duplicates; the spelling guard needs a training vocabulary. Decision: drop the duplicates from train; vocabulary = all words of the CLINC train rows (appliance rows excluded because they contain deliberate typos). Consequence: test is clean; real English words are protected and appliance typos stay correctable. (First version used 'seen >=4 times' plus extra lexicon words; a bug hunt showed it corrupted 53 real words on val+test, so it was reverted.)

## D7 · Spelling distance: <=1 for short lexicon words (deviation from the spec's "<=2")
Context: with distance <=2 everywhere, 20 real words on CLINC val+test were "corrected" (lover->oven, warning->washing) and two in the appliance data (flashing->washing, drawer->dryer). Options: keep <=2 as in CLAUDE.md; or <=1 when the lexicon word has fewer than 8 letters, <=2 for long ones. Decision (user): <=1 for short words. Consequence: wrong corrections 20 -> 4, real typos (dishwaser, washin, machin, refridgerator) still fixed; swapped letters (frigde) are no longer corrected.

## D8 · The 5 duplicate CLINC rows (user decision: remove from train, keep val/test official)
Context: 2 texts appear in train and test, 3 in train and val (exact match). CLAUDE.md says "official splits" and also forbids train/test overlap. Options: remove them from train; keep official splits and narrow the overlap test. Decision (user): remove from train. Also report overlaps after preprocessing and for val/test. Consequence: train 15,315 rows; scripts/data_checks.py shows 0 raw overlaps but 15 / 12 / 5 (train-test / train-val / val-test) after preprocessing, which are near-duplicates that differ only in case, punctuation or model codes; they are reported in the README, not removed.

## D9 · Spelling vocabulary from CLINC train rows only (user decision)
Context: CLAUDE.md says "not in the training vocabulary"; the appliance training rows contain deliberate typos. Decision (user): vocabulary = CLINC train rows only; spelling rules are judged on train and val only, never on test. Consequence: appliance typos are still corrected; on train+val CLINC rows 2 tokens are wrongly corrected (refrigerated -> refrigerator, ridge -> fridge), listed as a limitation in the README. Note: the earlier counts used to pick the distance rule (D7) included test rows; they were recomputed without test.

## D10 · 10 teams instead of "about 8" (user decision)
Context: CLAUDE.md says about 8 teams; teams.py has 10 (5 CLINC domain teams, Technical Service, Spare Parts, Warranty, Auto reply, Human agent). Decision (user): keep 10. Consequence: none for metrics; the team table is only used for display and routing.

## D11 M2 = word2vec mean-pooling + MLP (P2)
- Context: the spec asks for an own neural net on word vectors.
- Options: mean of word vectors (spec), LSTM over the sequence, fastText.
- Decision: gensim word2vec (dim 100, 30 epochs, trained on preprocessed train texts) -> mean vector -> MLP 100-256-n_classes, hand-written SGD loop (lr 0.1, momentum 0.9, batch 64, early stopping patience 15 on val accuracy).
- Consequence: tiny and fast (2.6 MB, 0.14 ms) but word order is lost, so it is weaker than M1 in-scope; lr/epochs/patience are my choices inside the spec.

## D12 BERT teacher -> DistilBERT student -> ONNX Runtime INT8
- Context: the homework wants a Transformer plus an efficient (smaller/faster) version of it.
- Options: quantize BERT directly; distill only; distill, then quantize (spec); torch.quantization.quantize_dynamic.
- Decision: M3 BERT fine-tuned (3 ep) -> M3s DistilBERT distilled (T=2, 5 ep, lr 5e-5 my choice) -> ONNX (legacy exporter, the default one fixed batch size to 1) -> onnxruntime quantize_dynamic QInt8 (M4). ONNX Runtime, because the PyTorch API is deprecated.
- Consequence: M3s alone shows the cost of distillation, M4 vs M3s the cost of quantization; M4 is the smallest and fastest Transformer row.
