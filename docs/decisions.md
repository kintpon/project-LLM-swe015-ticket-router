# Decision records

## D1 · Why Use Case 2 (Customer Support Ticket Router)
Context: homework offers several use cases. Options: other use cases. Decision: ticket router. Consequence: labelled intent data (CLINC150) exists and the routing result is easy to demo.

## D2 · How to tick the plan checklist
Context: the plan PDF has checkboxes. Options: draw ticks on the PDF (needs reportlab) or keep a separate checklist file. Decision (user): separate file, PLAN_CHECKLIST.md. Consequence: no extra package; the PDF stays untouched.

## D3 · Python 3.12 venv with CUDA torch
Context: system Python is 3.14 and the machine has an RTX 3060. Options: 3.14 as is, 3.12 via uv. Decision: 3.12 venv, torch cu128. Consequence: BERT can likely be trained locally; latency is still measured on CPU.
