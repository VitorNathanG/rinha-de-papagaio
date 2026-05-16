"""
Step 4/4 — Evaluate the papagaio on the held-out test split.

For each decision threshold, computes the rinha confusion matrix and
simulates score_det using the formula from AVALIACAO.md:

    E         = 1*FP + 3*FN + 5*Err
    epsilon   = E / N
    failures  = FP + FN + Err

    if failures/N > 0.15:   score_det = -3000   (cut)
    else:                   score_det = K*log10(1/max(eps, 0.001)) - beta*log10(1+E)

Here Err = 0 because this is an offline evaluation (no HTTP errors).

Outputs:
    data/results.json   threshold sweep with confusion matrix and score_det
    stdout              human-readable summary
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

from train import Papagaio

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"


def score_det(tp, tn, fp, fn, errs=0, K=1000.0, beta=300.0, eps_min=0.001, tx_corte=0.15):
    n = tp + tn + fp + fn + errs
    if n == 0:
        return {"score_det": 0.0, "cut": False, "E": 0, "epsilon": 0.0, "failure_rate": 0.0,
                "rate_component": 0.0, "absolute_penalty": 0.0}
    E = fp * 1 + fn * 3 + errs * 5
    failures = fp + fn + errs
    eps = E / n
    failure_rate = failures / n
    if failure_rate > tx_corte:
        return {"score_det": -3000.0, "cut": True, "E": E, "epsilon": eps,
                "failure_rate": failure_rate, "rate_component": None, "absolute_penalty": None}
    rate = K * math.log10(1.0 / max(eps, eps_min))
    pen = -beta * math.log10(1.0 + E)
    return {"score_det": rate + pen, "cut": False, "E": E, "epsilon": eps,
            "failure_rate": failure_rate, "rate_component": rate, "absolute_penalty": pen}


def main():
    refs = np.load(DATA_DIR / "references.npy")
    counts = np.load(DATA_DIR / "fraud_counts_k5.npy")
    test_idx = np.load(DATA_DIR / "test_indices.npy")

    ckpt = torch.load(DATA_DIR / "model.pt", weights_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Papagaio(hidden=ckpt["hidden"], depth=ckpt["depth"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    Xt = torch.from_numpy(refs[test_idx]).to(device)
    y_true_fraud = (counts[test_idx] >= 3)                  # bool — true label says deny
    expected_approved = ~y_true_fraud

    with torch.no_grad():
        probs = torch.sigmoid(model(Xt)).cpu().numpy()

    def eval_at(thr: float):
        pred_deny = probs >= thr
        pred_appr = ~pred_deny
        tp = int(np.sum(y_true_fraud & pred_deny))           # fraud correctly denied
        tn = int(np.sum(expected_approved & pred_appr))      # legit correctly approved
        fp = int(np.sum(expected_approved & pred_deny))      # legit denied
        fn = int(np.sum(y_true_fraud & pred_appr))           # fraud approved
        return tp, tn, fp, fn

    n_test = int(len(test_idx))
    fr = float(y_true_fraud.mean())
    print(f"[eval] test size: {n_test:,}    test fraud rate: {fr*100:.2f}%")
    print(f"[eval] model parameters: {n_params:,}")
    print()

    rows = []
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    for thr in thresholds:
        tp, tn, fp, fn = eval_at(thr)
        s = score_det(tp, tn, fp, fn)
        rows.append({"threshold": thr, "tp": tp, "tn": tn, "fp": fp, "fn": fn, **s})

    best = max(rows, key=lambda r: r["score_det"])

    print(f"{'thr':>5} {'TP':>8} {'TN':>8} {'FP':>8} {'FN':>8}"
          f" {'fail%':>7} {'eps':>8} {'rate':>7} {'pen':>7} {'score_det':>10}")
    print("-" * 88)
    for r in rows:
        mark = "  <- best" if r is best else ""
        rate = r["rate_component"] if r["rate_component"] is not None else float("nan")
        pen = r["absolute_penalty"] if r["absolute_penalty"] is not None else float("nan")
        print(f"{r['threshold']:>5.2f} {r['tp']:>8,} {r['tn']:>8,} {r['fp']:>8,} {r['fn']:>8,}"
              f" {r['failure_rate']*100:>6.2f}% {r['epsilon']:>8.4f}"
              f" {rate:>7.0f} {pen:>7.0f} {r['score_det']:>10.2f}{mark}")

    out = {
        "n_test": n_test,
        "test_fraud_rate": fr,
        "n_params": n_params,
        "rows": rows,
        "best": best,
        "max_possible_score_det": 3000.0,
        "min_possible_score_det": -3000.0,
    }
    (DATA_DIR / "results.json").write_text(json.dumps(out, indent=2))
    print()
    print(f"[eval] wrote {DATA_DIR / 'results.json'}")
    print(f"[eval] BEST score_det = {best['score_det']:.2f}  at threshold {best['threshold']}")
    print()
    print("[eval] interpreting this number:")
    print("       score_det >= 2500 -> destilation viable, p99 dominates final score")
    print("       1500..2500        -> viable, but a hybrid (fall back to exact on uncertain)")
    print("                            could push detection score higher")
    print("       < 1500            -> destilation alone leaves too much detection score")
    print("                            on the table; reconsider the approach")
    print("       NOTE: real rinha test queries are NOT the references themselves, so")
    print("             real performance will be somewhat WORSE than this leave-one-out")
    print("             upper bound. Treat this as the optimistic ceiling.")


if __name__ == "__main__":
    main()
