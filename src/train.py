"""
Trains both models on the GX10 and saves them to model.pt.

  1. Autoencoder  -- trained ONLY on "normal" data. High reconstruction
                     error later = "this doesn't look healthy".
  2. Classifier   -- trained on all labels. Names the fault.

    python train.py
"""
import numpy as np
import torch
import torch.nn as nn

from features import NOT_FAULTS, load_dataset

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class AutoEncoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 12))
        self.dec = nn.Sequential(nn.Linear(12, 64), nn.ReLU(), nn.Linear(64, d))

    def forward(self, x):
        return self.dec(self.enc(x))


class Classifier(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, k))

    def forward(self, x):
        return self.net(x)


def fit(model, X, target, loss_fn, epochs=60):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X), device=DEV)
        for i in range(0, len(X), 128):
            idx = perm[i:i + 128]
            loss = loss_fn(model(X[idx]), target[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def recon_error(ae, X):
    ae.eval()
    with torch.no_grad():
        return ((ae(X) - X) ** 2).mean(dim=1).cpu().numpy()


def main():
    print(f"device: {DEV}")
    X, y, fid, pos, aug = load_dataset("data")
    classes = sorted(set(y))
    assert "normal" in classes, "need at least one CSV labeled 'normal'"

    # last 20% of every recording is held out -- overlapping windows would leak otherwise.
    # Augmented copies are training-only; grading always uses untouched real data.
    val = (pos > 0.8) & ~aug
    tr = (pos <= 0.8)
    normal_tr = tr & (y == "normal")

    mu = X[normal_tr].mean(0)
    sd = X[normal_tr].std(0) + 1e-6
    Xs = torch.tensor((X - mu) / sd, dtype=torch.float32, device=DEV)
    yi = torch.tensor([classes.index(c) for c in y], device=DEV)

    # 1. autoencoder on healthy data only
    ae = AutoEncoder(X.shape[1]).to(DEV)
    fit(ae, Xs[normal_tr], Xs[normal_tr], nn.MSELoss())
    err = recon_error(ae, Xs)
    threshold = float(np.percentile(err[val & (y == "normal")], 99) * 1.5)

    # 2. classifier on everything
    clf = Classifier(X.shape[1], len(classes)).to(DEV)
    fit(clf, Xs[tr], yi[tr], nn.CrossEntropyLoss())
    clf.eval()
    with torch.no_grad():
        pred = clf(Xs).argmax(1).cpu().numpy()

    # report on held-out windows
    print(f"\nanomaly threshold: {threshold:.4f}")
    print(f"{'class':<16}{'flagged anomalous':>18}{'classified right':>18}")
    for c in classes:
        m = val & (y == c)
        flagged = (err[m] > threshold).mean()
        right = (pred[m] == classes.index(c)).mean()
        if c == "normal":
            note = "  (flagged % = false alarm rate)"
        elif c in NOT_FAULTS:
            note = "  (not a fault: shown as a notice, never an alarm)"
        else:
            note = "  (flagged % = detection rate)"
        print(f"{c:<16}{flagged:>17.0%}{right:>18.0%}{note}")

    torch.save({"ae": ae.state_dict(), "clf": clf.state_dict(), "mu": mu, "sd": sd,
                "threshold": threshold, "classes": classes, "d": X.shape[1]}, "model.pt")
    print("\nsaved model.pt")


if __name__ == "__main__":
    main()
