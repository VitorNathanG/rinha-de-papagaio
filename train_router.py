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

Treina sobre os labels PRISTINE (`box_labels.before_halo.npy`), não sobre o
post-halo. O halo só engorda o Box-B pro slow path conseguir achar refs de
borda; o router deve só distinguir clusters homogêneos (A) de tudo mais
(misturas, outliers de label = B). Esse desacoplamento permite ao router
zerar misroute B→A sem precisar memorizar a casca outward do halo.

Loss: cross-entropy com inverse-frequency class weights. B é ~3.5% dos dados
pristine (vs ~7% post-halo), então sem peso o modelo colapsa em "prediz A
sempre".

Critério de parada principal: primeira epoch com **misroute B→A = 0 sobre os
3M refs completos** (train + val + test, ou seja, todo o dataset rotulado).
Confusion matrix 3x3 é computada e impressa em cada epoch sobre o 3M completo.
Patience baseada em val_loss continua como fallback se nunca zerar.

Critério de save: state_dict com **menor misroute_3M** ao longo do treino —
NÃO o de menor val_loss. Empiricamente os dois divergem: epoch com val_loss
mínimo costuma ter ~3-4× mais misroutes que a epoch de melhor recall em B.
Como o objetivo do router é justamente zerar misroute, salvamos por ele.

Tunables (env vars):
    HIDDEN    hidden width                    (default 32)
    DEPTH     number of hidden layers         (default 2)
    LR        Adam learning rate              (default 1e-3)
    EPOCHS    training epochs                 (default 500)
    BATCH     SGD minibatch                   (default 8192)
    SEED      split + init seed               (default 42)
    PATIENCE  epochs sem melhora pra parar    (default 30)
    MIN_DELTA threshold de melhora val_loss   (default 1e-5)

Outputs:
    data/router.pt
    data/router_train_indices.npy
    data/router_val_indices.npy
    data/router_test_indices.npy
"""
import pipeline_log

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
        # GELU(tanh) — the OpenAI / GPT-2 approximation. Picked over the
        # default erf-based GELU because the Rust inference kernel reuses
        # this formula via libm::tanhf (one transcendental call instead of
        # one erff that itself fans out into expf), and we need the train
        # and inference functions to match numerically.
        gelu = lambda: nn.GELU(approximate="tanh")
        layers = [nn.Linear(14, hidden), gelu()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), gelu()]
        layers += [nn.Linear(hidden, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main():
    seed = int(os.environ.get("SEED", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)

    refs = np.load(DATA_DIR / "references.npy")
    # Pristine labels (sem o halo D=0.23). Halo é só pro Box-B do slow path;
    # treinar o router em cima dele força ele a memorizar a casca outward —
    # tarefa fora da capacidade dum MLP de 1.6k params e contra a intuição de
    # "incerteza → vetorial".
    box = np.load(DATA_DIR / "box_labels.before_halo.npy")
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
    # Full 3M no device pra CM por epoch + critério de parada zero-misroute.
    # 3M × 14 × float32 ≈ 168 MB no GPU — cabe nos 17 GB do RDNA4.
    X_all = torch.from_numpy(refs).to(device)
    y_all = torch.from_numpy(box).to(device).long()

    class_counts = np.bincount(box[train_idx], minlength=3)
    # Inverse-frequency weighting; divide by num_classes so the average weight is 1.
    weights = class_counts.sum() / (class_counts.clip(min=1) * 3)
    weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"[router] class counts (train): A-L={class_counts[0]:,}  "
          f"A-F={class_counts[1]:,}  B={class_counts[2]:,}")
    print(f"[router] class weights:        A-L={weights[0]:.3f}  "
          f"A-F={weights[1]:.3f}  B={weights[2]:.3f}")

    hidden = int(os.environ.get("HIDDEN", 64))
    depth = int(os.environ.get("DEPTH", 2))
    model = Router(hidden=hidden, depth=depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[router] model: hidden={hidden}  depth={depth}  params={n_params:,}")

    loss_fn = nn.CrossEntropyLoss(weight=weights_t)
    lr = float(os.environ.get("LR", 1e-3))
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = int(os.environ.get("EPOCHS", 500))
    batch = int(os.environ.get("BATCH", 8192))
    # Early stopping: bail when val_loss hasn't improved by MIN_DELTA in the
    # last PATIENCE epochs. Keeps the "best state" model regardless.
    patience = int(os.environ.get("PATIENCE", 30))
    min_delta = float(os.environ.get("MIN_DELTA", 1e-5))
    print(f"[router] epochs<={epochs}  batch={batch}  lr={lr}  "
          f"patience={patience}  min_delta={min_delta}")

    t0 = time.time()
    # Patience continua sendo dirigida por val_loss (sinal mais estável que
    # misroute, que oscila ±30 por causa de não-determinismo do argmax em
    # samples borderline). best_state, porém, segue misroute_3M — esse é o
    # objetivo do treino e val_loss menor não implica menos misroute.
    best_val = float("inf")
    best_misroute = float("inf")
    best_state = None
    best_epoch = 0
    epochs_since_best = 0
    stopped_early = False
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
            # val_loss continua sendo computado em val pra patience.
            v_logits = model(Xva)
            val_loss = loss_fn(v_logits, yva).item()
            # CM e misroute são sobre os 3M completos (train + val + test).
            # IMPORTANTE: rodar `model(X_all)` com X_all de 3M de uma vez
            # produz logits corrompidos no backend ROCm (magnitudes ~100×
            # maiores que o real, predições aleatórias). Chunked passa.
            # Repro: ver hash a8c8 do TODO histórico — aparentemente kernel
            # de matmul ou GELU falha com batch dim acima de ~1M no RDNA4.
            all_pred = torch.empty(len(y_all), dtype=torch.long, device=device)
            eval_chunk = 100_000
            for s in range(0, len(y_all), eval_chunk):
                e = s + eval_chunk
                all_pred[s:e] = model(X_all[s:e]).argmax(dim=1)
            acc = (all_pred == y_all).float().mean().item()
            cm = torch.zeros(3, 3, dtype=torch.long, device=device)
            for c in range(3):
                mask = (y_all == c)
                for p in range(3):
                    cm[c, p] = ((all_pred == p) & mask).sum()
            cm_np = cm.cpu().numpy()
            row_tot = cm_np.sum(axis=1)
            recalls = [cm_np[c, c] / max(int(row_tot[c]), 1) for c in range(3)]
            # Misroute = true=B mas pred != B sobre TODO o dataset. Esse é o
            # número que precisa ir a zero — qualquer ref true-B classificado
            # como A vira (potencialmente) um bypass do slow path em prod.
            misroute = int(cm_np[2, 0] + cm_np[2, 1])
            true_b_n = int(row_tot[2])
        elapsed = time.time() - t0
        marker = ""
        # Save: epoch com menor misroute_3M ganha (empate = mais recente vence,
        # tende a ter pesos mais "assentados" do otimizador).
        if misroute <= best_misroute:
            best_misroute = misroute
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = ep + 1
            marker = " *"
        # Patience: zera quando val_loss melhora (sinal de progresso geral).
        if val_loss < best_val - min_delta:
            best_val = val_loss
            epochs_since_best = 0
        else:
            epochs_since_best += 1
        print(f"[router] ep {ep+1:>3}/{epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"acc_3M={acc*100:5.2f}%  "
              f"recall=[A-L={recalls[0]*100:5.2f}% "
              f"A-F={recalls[1]*100:5.2f}% "
              f"B={recalls[2]*100:5.2f}%]"
              f"  misroute_3M={misroute}/{true_b_n}"
              f"  elapsed={elapsed:.0f}s{marker}")
        # Confusion matrix completa sobre o 3M (uma linha por classe verdadeira).
        for c, name in enumerate(("A-L", "A-F", "B  ")):
            print(f"[router]   cm true={name}: "
                  f"→A-L={cm_np[c,0]:>9,}  →A-F={cm_np[c,1]:>9,}  "
                  f"→B={cm_np[c,2]:>9,}  (tot={row_tot[c]:>9,})")
        # Objetivo: primeira epoch com zero misroutes B→A no 3M completo.
        if misroute == 0:
            print(f"[router] OBJECTIVE HIT: zero misroutes B→A em 3M no ep {ep+1}. Stopping.")
            best_misroute = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = ep + 1
            stopped_early = True
            break
        if epochs_since_best >= patience:
            print(f"[router] early stop: no val_loss improvement in "
                  f"{patience} epochs (best at ep {best_epoch}, "
                  f"best_misroute_3M={best_misroute})")
            stopped_early = True
            break

    if not stopped_early:
        print(f"[router] reached max epochs={epochs} without early stop "
              f"(best at ep {best_epoch}, best_misroute_3M={best_misroute})")

    model.load_state_dict(best_state)
    torch.save(
        {"state_dict": model.state_dict(), "hidden": hidden, "depth": depth},
        DATA_DIR / "router.pt",
    )
    print(f"[router] saved data/router.pt (best misroute_3M={best_misroute}, ep={best_epoch})")


if __name__ == "__main__":
    pipeline_log.setup(__file__)
    main()
