"""Area D - Efficient model, step 2: export M3s to ONNX, then INT8 dynamic quantization (ONNX Runtime).
M4 runs with ONNX Runtime only; no PyTorch model is needed at inference (tokenizer aside)."""
import argparse

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import src.m3_bert as m3
from src.data import load_split
from src.evaluate import evaluate
from src.logutil import log


def export_onnx(folder, onnx_path):
    model = AutoModelForSequenceClassification.from_pretrained(folder).eval()
    tok = AutoTokenizer.from_pretrained(folder)
    enc = tok(["hello"], padding="max_length", max_length=m3.MAX_LEN, return_tensors="pt")
    args = (enc["input_ids"], enc["attention_mask"])
    kw = dict(input_names=["input_ids", "attention_mask"], output_names=["logits"], opset_version=17,
              dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}, "logits": {0: "batch"}})
    # legacy exporter (dynamo=False): the default one fixed the batch size to 1 and failed on bigger batches
    torch.onnx.export(model, args, str(onnx_path), dynamo=False, **kw)


def make_predict_proba(onnx_path, folder):
    """predict_proba on CPU with ONNX Runtime, 4 threads."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    tok = AutoTokenizer.from_pretrained(folder)

    def predict_proba(texts):
        enc = tok(m3.clean(texts), padding="max_length", truncation=True, max_length=m3.MAX_LEN, return_tensors="np")
        logits = session.run(None, {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    return predict_proba


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    smoke = ap.parse_args().smoke
    folder = m3.model_dir("m3s_distilbert", smoke)
    fp32_path, int8_path = folder.parent / "m3s_fp32.onnx", folder.parent / "m4_int8.onnx"
    export_onnx(folder, fp32_path)  # the un-quantized ONNX file is kept too
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    p_int8 = make_predict_proba(int8_path, folder)
    row, int8_probs = evaluate("M4", "DistilBERT student INT8 (ONNX Runtime)", p_int8, [int8_path], smoke)
    # sanity check: M4 and M3s should agree on >= 90% of test predictions
    test = load_split("test")
    fp32_probs = m3.make_predict_proba(folder)(test["text"].tolist())
    agree = float((fp32_probs.argmax(1) == int8_probs.argmax(1)).mean())
    msg = f"M4 smoke={smoke} agree_with_M3s={agree:.4f} fp32_onnx={fp32_path.stat().st_size / 1e6:.1f}MB int8={row['size_mb']:.1f}MB"
    print(msg)
    log("m4_quantize", msg)
