"""Area A - own neural network M2: word2vec (gensim) -> mean vector -> MLP 100-256-n_classes, hand-written SGD loop."""
import argparse
import random

import numpy as np
import torch
from gensim.models import KeyedVectors, Word2Vec
from torch import nn

from config import MODELS_DIR, SEED
from src.data import label_names, load_split
from src.evaluate import evaluate
from src.logutil import log
from src.preprocess import preprocess

DIM = 100
torch.set_num_threads(4)  # latency is measured with 4 threads


def paths(smoke=False):
    folder = MODELS_DIR / "smoke" if smoke else MODELS_DIR
    return folder / "m2_word2vec.kv", folder / "m2_mlp.pt"


def tokens(texts):
    return [preprocess(t)[0].split() for t in texts]


def embed(token_lists, wv):
    """Mean word vector per message; zero vector if no word is known."""
    out = np.zeros((len(token_lists), DIM), dtype=np.float32)
    for i, toks in enumerate(token_lists):
        vecs = [wv[t] for t in toks if t in wv]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return out


def make_net(n_classes):
    return nn.Sequential(nn.Linear(DIM, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, n_classes))


def accuracy(net, x, y):
    net.eval()
    with torch.no_grad():
        return (net(x).argmax(1) == y).float().mean().item()


def train(smoke=False, rows=None):
    """Train word2vec on the train texts, then the MLP with a manual SGD loop; keep the best epoch by val accuracy."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    train_df, val_df = load_split("train"), load_split("val")
    if smoke:
        train_df, val_df = train_df.sample(rows or 300, random_state=SEED), val_df.sample(100, random_state=SEED)
    train_tok, val_tok = tokens(train_df["text"]), tokens(val_df["text"])
    w2v = Word2Vec(train_tok, vector_size=DIM, window=5, min_count=1, epochs=5 if smoke else 30,
                   workers=1, seed=SEED)  # workers=1 keeps it reproducible
    x_train = torch.tensor(embed(train_tok, w2v.wv))
    y_train = torch.tensor(train_df["label"].values)
    x_val, y_val = torch.tensor(embed(val_tok, w2v.wv)), torch.tensor(val_df["label"].values)

    net = make_net(len(label_names()))
    opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    best_acc, best_state, bad_epochs = -1.0, None, 0
    for epoch in range(1 if smoke else 200):
        net.train()
        order = torch.randperm(len(x_train))
        total = 0.0
        for i in range(0, len(order), 64):
            idx = order[i:i + 64]
            opt.zero_grad()
            loss = loss_fn(net(x_train[idx]), y_train[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        val_acc = accuracy(net, x_val, y_val)
        log("m2", f"epoch {epoch + 1} loss {total / len(order):.4f} val_acc {val_acc:.4f}")  # mean loss of the epoch
        if val_acc > best_acc:
            best_acc, bad_epochs = val_acc, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= 15:  # early stopping
                break
    net.load_state_dict(best_state)
    kv_path, net_path = paths(smoke)
    kv_path.parent.mkdir(parents=True, exist_ok=True)
    w2v.wv.save(str(kv_path))
    torch.save(net.state_dict(), net_path)


def make_predict_proba(smoke=False):
    kv_path, net_path = paths(smoke)
    wv = KeyedVectors.load(str(kv_path))
    net = make_net(len(label_names()))
    net.load_state_dict(torch.load(net_path))
    net.eval()

    def predict_proba(texts):
        x = torch.tensor(embed(tokens(texts), wv))
        with torch.no_grad():
            return torch.softmax(net(x), dim=1).numpy()
    return predict_proba


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    smoke = ap.parse_args().smoke
    train(smoke)
    row, _ = evaluate("M2", "word2vec + MLP", make_predict_proba(smoke), paths(smoke), smoke)
    log("m2", f"M2 smoke={smoke} in_scope={row['in_scope_acc']:.4f} oos_recall={row['oos_recall']:.4f}")
