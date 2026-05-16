"""
Step 3.5 — Train the 3-class router ("box classifier").

Predicts P(A-Legit), P(A-Fraud), P(B) from a 14-dim query vector. At inference
the routing rule is:
    if P(B) > tau:  → slow path (exact k=5 over references)
    else:           → return argmax(P(A-Legit), P(A-Fraud)) directly

This collapses the two-stage architecture (separate router + Box-A predictor)
into a single small model. The classifier sees Box-B examples in training, so
its uncertainty at the decision boundary is calibrated by data — unlike the
Box-A-only MLP, which has no training signal to be uncertain there.

Loss: cross-entropy with inverse-frequency class weights. Box B is ~3.5% of
the data, so without weighting the model collapses to "predict A every time"
and ignores the class that matters for routing. Inverse-frequency weights are
the simplest fix.

Tunables (env vars):
    HIDDEN   hidden width       (default 32)
    DEPTH    number of hidden layers (default 2)
    LR       Adam learning rate (default 1e-3)
    EPOCHS   training epochs    (default 12)
    BATCH    SGD minibatch      (default 8192)
    SEED     split + init seed  (default 42)

Outputs:
    data/router.pt
    data/router_train_indices.npy
    data/router_val_indices.npy
    data/router_test_indices.npy
"""
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"


class Router(nn.Module):
    def __init__(self, hidden: int = 32, depth: int = 2, num_classes: int = 3):
        super().__init__()
        layers = [nn.Linear(14, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main():
    seed = int(os.environ.get("SEED", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)

    refs = np.load(DATA_DIR / "references.npy")
    box = np.load(DATA_DIR / "box_labels.npy")
    N = int(len(box))

    perm = np.random.default_rng(seed).permutation(N)
    n_train = int(0.80 * N)
    n_val = int(0.10 * N)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    np.save(DATA_DIR / "router_train_indices.npy", train_idx)
    np.save(DATA_DIR / "router_val_indices.npy", val_idx)
    np.save(DATA_DIR / "router_test_indices.npy", test_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[router] device: {device}")
    print(f"[router] N={N:,}  train={len(train_idx):,}  val={len(val_idx):,}  test={len(test_idx):,}")

    Xtr = torch.from_numpy(refs[train_idx]).to(device)
    ytr = torch.from_numpy(box[train_idx]).to(device).long()
    Xva = torch.from_numpy(refs[val_idx]).to(device)
    yva = torch.from_numpy(box[val_idx]).to(device).long()

    class_counts = np.bincount(box[train_idx], minlength=3)
    # Inverse-frequency weighting; divide by num_classes so the average weight is 1.
    weights = class_counts.sum() / (class_counts.clip(min=1) * 3)
    weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"[router] class counts (train): A-L={class_counts[0]:,}  "
          f"A-F={class_counts[1]:,}  B={class_counts[2]:,}")
    print(f"[router] class weights:        A-L={weights[0]:.3f}  "
          f"A-F={weights[1]:.3f}  B={weights[2]:.3f}")

    hidden = int(os.environ.get("HIDDEN", 32))
    depth = int(os.environ.get("DEPTH", 2))
    model = Router(hidden=hidden, depth=depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[router] model: hidden={hidden}  depth={depth}  params={n_params:,}")

    loss_fn = nn.CrossEntropyLoss(weight=weights_t)
    lr = float(os.environ.get("LR", 1e-3))
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = int(os.environ.get("EPOCHS", 12))
    batch = int(os.environ.get("BATCH", 8192))
    print(f"[router] epochs={epochs}  batch={batch}  lr={lr}")

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
            v_pred = v_logits.argmax(dim=1)
            acc = (v_pred == yva).float().mean().item()
            recalls = []
            for c in range(3):
                mask = (yva == c)
                if mask.sum() == 0:
                    recalls.append(float("nan"))
                else:
                    recalls.append(((v_pred == c) & mask).float().sum().item()
                                   / mask.sum().item())
        elapsed = time.time() - t0
        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            marker = " *"
        print(f"[router] ep {ep+1:>2}/{epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"acc={acc*100:5.2f}%  "
              f"recall=[A-L={recalls[0]*100:5.2f}% "
              f"A-F={recalls[1]*100:5.2f}% "
              f"B={recalls[2]*100:5.2f}%]"
              f"  elapsed={elapsed:.0f}s{marker}")

    model.load_state_dict(best_state)
    torch.save(
        {"state_dict": model.state_dict(), "hidden": hidden, "depth": depth},
        DATA_DIR / "router.pt",
    )
    print(f"[router] saved data/router.pt (best val_loss={best_val:.4f})")


if __name__ == "__main__":
    main()
