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
