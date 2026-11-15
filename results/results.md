| Model | In-scope acc | OOS recall | Appliance acc (15 rows) | Threshold | Size (MB) | Latency mean (ms) | Latency p95 (ms) |
|---|---|---|---|---|---|---|---|
| M1 tf-idf + LogReg | 0.7980 | 0.9070 | 0.0000 | 0.45 | 40.9 | 0.43 | 0.61 |
| M2 word2vec + MLP | 0.6726 | 0.8770 | 0.1333 | 0.90 | 2.6 | 0.14 | 0.36 |
| M3 BERT-base (fp32) | 0.9094 | 0.8720 | 0.0000 | 0.25 | 438.4 | 61.43 | 70.90 |
| M3s DistilBERT student (fp32) | 0.9079 | 0.9100 | 0.0000 | 0.65 | 268.3 | 31.76 | 34.29 |
| M4 DistilBERT student INT8 (ONNX Runtime) | 0.8930 | 0.9320 | 0.0000 | 0.65 | 67.5 | 12.01 | 13.05 |

Hardware: 12th Gen Intel(R) Core(TM) i7-12700H (20 logical cores, 31.8 GB RAM), Windows-11-10.0.26200-SP0, Python 3.12.13, torch 2.11.0+cu128, onnxruntime 1.32.0; latency on CPU, 4 threads

Notes: size = all files needed at inference, tokenizer files excluded. Latency = one message at a time, preprocess -> route, 10 warm-up calls then 200 test messages (same for all models). Metrics are measured on the models alone (small-talk rules not applied). Appliance accuracy covers only 15 test rows (3 per appliance intent), so treat it with caution.
