| Model | In-scope acc | OOS recall | Appliance acc (30 rows) | Threshold | Size (MB) | Latency mean (ms) | Latency p95 (ms) |
|---|---|---|---|---|---|---|---|
| M1 tf-idf + LogReg | 0.7973 | 0.9110 | 0.0000 | 0.45 | 40.8 | 0.45 | 0.68 |

Hardware: 12th Gen Intel(R) Core(TM) i7-12700H (20 logical cores), Windows-11-10.0.26200-SP0, Python 3.12.13, torch 2.11.0+cu128, onnxruntime 1.32.0; latency on CPU, 4 threads

Notes: size = all files needed at inference, tokenizer files excluded. Latency = one message at a time, preprocess -> route, 10 warm-up calls then 200 test messages (same for all models). Metrics are measured on the models alone (small-talk rules not applied). Appliance accuracy covers only 30 test rows, so treat it with caution.
