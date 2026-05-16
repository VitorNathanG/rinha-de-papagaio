"""
Step 3/4 — Train the "papagaio": a small MLP that maps a 14-dim vector to
P(this query would be denied by 5-NN over the references).

Target derivation:
    fraud_count = number of fraud labels among the 5 nearest neighbors
    target      = 1 if fraud_count >= 3 else 0     (rinha threshold is 0.6 -> >=3/5)

Loss: BCEWithLogitsLoss with pos_weight=POS_WEIGHT (default 3.0).
The rinha scoring weights FN at 3x FP, so we upweight the positive (fraud)
class during training to push the model toward calling things fraud when in
doubt.

Tunables (env vars):
    HIDDEN       hidden width      (default 64)
    DEPTH        number of hidden layers (default 3)
    LR           Adam learning rate (default 1e-3)
    EPOCHS       passes over train data (default 12)
    BATCH        SGD minibatch     (default 8192)
    POS_WEIGHT   positive class weight in BCE loss (default 3.0)
    SEED         shuffle seed      (default 42)

Outputs:
    data/model.pt
    data/train_indices.npy, data/val_indices.npy, data/test_indices.npy
"""
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"


class Papagaio(nn.Module):
    def __init__(self, hidden: int = 64, depth: int = 3):
        super().__init__()
        layers = [nn.Linear(14, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def main():
    seed = int(os.environ.get("SEED", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)

    refs = np.load(DATA_DIR / "references.npy")           # (N_all, 14)
    counts = np.load(DATA_DIR / "fraud_counts_k5.npy")     # (N,) — may be shorter
    N = int(counts.shape[0])
    refs = refs[:N]
    y = (counts >= 3).astype(np.float32)                   # 1 == fraud (deny)

    perm = np.random.default_rng(seed).permutation(N)
    n_train = int(0.80 * N)
    n_val = int(0.10 * N)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    np.save(DATA_DIR / "train_indices.npy", train_idx)
    np.save(DATA_DIR / "val_indices.npy", val_idx)
    np.save(DATA_DIR / "test_indices.npy", test_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")
    print(f"[train] N={N:,}  train={len(train_idx):,}  val={len(val_idx):,}  test={len(test_idx):,}")
    print(f"[train] target rate (fraud=1): {y.mean()*100:.2f}%")

    Xtr = torch.from_numpy(refs[train_idx]).to(device)
    ytr = torch.from_numpy(y[train_idx]).to(device)
    Xva = torch.from_numpy(refs[val_idx]).to(device)
    yva = torch.from_numpy(y[val_idx]).to(device)

    hidden = int(os.environ.get("HIDDEN", 64))
    depth = int(os.environ.get("DEPTH", 3))
    model = Papagaio(hidden=hidden, depth=depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model: hidden={hidden}  depth={depth}  params={n_params:,}")

    pos_weight = torch.tensor(float(os.environ.get("POS_WEIGHT", 3.0)), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=float(os.environ.get("LR", 1e-3)))

    epochs = int(os.environ.get("EPOCHS", 12))
    batch = int(os.environ.get("BATCH", 8192))
    print(f"[train] epochs={epochs}  batch={batch}  lr={opt.defaults['lr']}  "
          f"pos_weight={pos_weight.item()}")

    t0 = time.time()
    best_val = float("inf")
    best_state = None
    n = Xtr.shape[0]
    for ep in range(epochs):
        model.train()
        order = torch.randperm(n, device=device)
        tot, count = 0.0, 0
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = Xtr[idx], ytr[idx]
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * xb.shape[0]
            count += xb.shape[0]
        train_loss = tot / count

        model.eval()
        with torch.no_grad():
            v_logits = model(Xva)
            val_loss = loss_fn(v_logits, yva).item()
            v_pred = (torch.sigmoid(v_logits) >= 0.5)
            v_true = (yva >= 0.5)
            acc = (v_pred == v_true).float().mean().item()
        elapsed = time.time() - t0
        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            marker = " *"
        print(f"[train] ep {ep+1:>2}/{epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={acc*100:.2f}%  elapsed={elapsed:.0f}s{marker}")

    model.load_state_dict(best_state)
    torch.save(
        {"state_dict": model.state_dict(), "hidden": hidden, "depth": depth},
        DATA_DIR / "model.pt",
    )
    print(f"[train] saved data/model.pt (best val_loss={best_val:.4f})")


if __name__ == "__main__":
    main()
