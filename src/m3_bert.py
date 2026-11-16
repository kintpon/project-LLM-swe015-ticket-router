"""Area C - Transformer. M3 = bert-base-uncased fine-tuned (teacher). Also holds the training loop and
predict_proba helpers that the DistilBERT student (m4_distill.py) reuses."""
import argparse
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from config import MODELS_DIR, SEED
from src.data import label_names, load_split
from src.evaluate import evaluate
from src.logutil import log
from src.preprocess import preprocess

BERT = "google-bert/bert-base-uncased"
MAX_LEN, BATCH = 32, 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def model_dir(name, smoke=False):
    return (MODELS_DIR / "smoke" if smoke else MODELS_DIR) / name


def clean(texts):
    """Same preprocessing as every other model."""
    return [preprocess(t)[0] for t in texts]


def encode(tok, texts):
    return tok(clean(texts), padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")


def logits_of(model, tok, texts):
    """Logits for many texts (used on GPU for training-time evaluation and teacher caching)."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 128):
            enc = encode(tok, texts[i:i + 128]).to(DEVICE)
            out.append(model(**enc).logits.float().cpu())
    return torch.cat(out)


def val_accuracy(model, tok, val):
    pred = logits_of(model, tok, val["text"].tolist()).argmax(dim=1).numpy()
    return float((pred == val["label"].to_numpy()).mean())


def train_model(model, tok, train, val, epochs, lr, save_to, teacher_logits=None, T=2.0):
    """Hand-written loop. With teacher_logits: loss = 0.5*CE + 0.5*T^2*KL(teacher/T || student/T).
    F.kl_div(log q_student, p_teacher) computes KL(teacher || student): the standard Hinton distillation loss."""
    set_seed()
    model.to(DEVICE)
    texts, labels = train["text"].tolist(), torch.tensor(train["label"].to_numpy())
    steps = epochs * ((len(texts) + BATCH - 1) // BATCH)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    best = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(len(texts))
        total = 0.0
        for i in range(0, len(texts), BATCH):
            idx = order[i:i + BATCH]
            enc = encode(tok, [texts[j] for j in idx]).to(DEVICE)
            y = labels[idx].to(DEVICE)
            out = model(**enc).logits
            loss = F.cross_entropy(out, y)
            if teacher_logits is not None:
                t = teacher_logits[idx].to(DEVICE)
                kl = F.kl_div(F.log_softmax(out / T, dim=1), F.softmax(t / T, dim=1), reduction="batchmean")
                loss = 0.5 * loss + 0.5 * T * T * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        acc = val_accuracy(model, tok, val)
        log(save_to.name, f"epoch {epoch} loss {total / len(texts):.4f} val_acc {acc:.4f}")
        print(f"{save_to.name} epoch {epoch}: loss {total / len(texts):.4f}, val acc {acc:.4f}")
        if acc > best:  # keep the best epoch by validation accuracy
            best = acc
            model.save_pretrained(save_to)
            tok.save_pretrained(save_to)
    return best


def new_model(name):
    n = len(label_names())
    return AutoModelForSequenceClassification.from_pretrained(name, num_labels=n), AutoTokenizer.from_pretrained(name)


def get_data(smoke):
    train, val = load_split("train"), load_split("val")
    if smoke:
        train, val = train.sample(300, random_state=SEED), val.sample(300, random_state=SEED)
    return train, val


def make_predict_proba(folder):
    """CPU predict_proba from a saved fp32 model folder (4 threads)."""
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(folder)
    model = AutoModelForSequenceClassification.from_pretrained(folder).to("cpu").eval()

    def predict_proba(texts):
        enc = tok(clean(texts), padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            return torch.softmax(model(**enc).logits, dim=1).numpy()
    return predict_proba


def weight_files(folder):
    return [folder / "model.safetensors"]  # tokenizer files are excluded from the size


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    smoke = ap.parse_args().smoke
    folder = model_dir("m3_bert", smoke)
    train, val = get_data(smoke)
    start = time.time()
    set_seed()  # before new_model: the new classification head is random
    model, tok = new_model(BERT)
    best = train_model(model, tok, train, val, 1 if smoke else 3, 3e-5, folder)
    model_dir("m3_train_logits.npy", smoke).unlink(missing_ok=True)  # cached teacher logits are stale now
    print(f"train time {time.time() - start:.0f} s on {DEVICE}, best val acc {best:.4f}")
    del model
    row, _ = evaluate("M3", "BERT-base (fp32)", make_predict_proba(folder), weight_files(folder), smoke)
    log("m3_bert", f"M3 smoke={smoke} in_scope={row['in_scope_acc']:.4f} oos_recall={row['oos_recall']:.4f}")
