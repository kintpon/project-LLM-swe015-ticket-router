"""Area D - Efficient model, step 1: knowledge distillation. M3s = DistilBERT student (fp32) learns from the
fine-tuned BERT teacher (M3). loss = 0.5*CE + 0.5*T^2*KL(student/T || teacher/T), T = 2, 5 epochs."""
import argparse
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import src.m3_bert as m3
from src.evaluate import evaluate
from src.logutil import log

STUDENT = "distilbert/distilbert-base-uncased"
EPOCHS, LR, T = 5, 5e-5, 2.0  # student lr is my choice (not given in the spec)


def teacher_logits(train, smoke):
    """Teacher logits for the train set, computed once and cached."""
    cache = m3.model_dir("m3_train_logits.npy", smoke)
    if cache.exists():
        return torch.tensor(np.load(cache))
    folder = m3.model_dir("m3_bert", smoke)
    tok = AutoTokenizer.from_pretrained(folder)
    teacher = AutoModelForSequenceClassification.from_pretrained(folder).to(m3.DEVICE)
    logits = m3.logits_of(teacher, tok, train["text"].tolist())
    np.save(cache, logits.numpy())
    return logits


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    smoke = ap.parse_args().smoke
    folder = m3.model_dir("m3s_distilbert", smoke)
    train, val = m3.get_data(smoke)
    start = time.time()
    t_logits = teacher_logits(train, smoke)
    m3.set_seed()  # before new_model: the new classification head is random
    model, tok = m3.new_model(STUDENT)
    best = m3.train_model(model, tok, train, val, 1 if smoke else EPOCHS, LR, folder, t_logits, T)
    print(f"train time {time.time() - start:.0f} s on {m3.DEVICE}, best val acc {best:.4f}")
    del model
    row, _ = evaluate("M3s", "DistilBERT student (fp32)", m3.make_predict_proba(folder), m3.weight_files(folder), smoke)
    log("m4_distill", f"M3s smoke={smoke} in_scope={row['in_scope_acc']:.4f} oos_recall={row['oos_recall']:.4f}")
