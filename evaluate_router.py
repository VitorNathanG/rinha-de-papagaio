"""
Step 4.5 — Evaluate the 3-class router.

Three reports:

1. 3x3 box confusion matrix (true box class vs predicted class).
2. Misroute analysis — among true B test points, how many are predicted as A?
   Those queries bypass the slow path and rely on the direct argmax prediction;
   that is the safety-critical number for this architecture.
3. End-to-end rinha confusion + score_det at several P(B) thresholds, assuming
   the slow path is the rinha oracle (i.e., exact k=5 over the references,
   correct by definition). This gives the optimistic upper bound for this
   architecture. Real slow path will differ slightly.

The k=5 ground truth comes from fraud_counts_k5.npy: deny iff count >= 3.

Outputs:
    data/router_results.json
    stdout            human-readable report
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from train_router import Router

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

A_LEGIT = 0
A_FRAUD = 1
B = 2


def score_det(tp, tn, fp, fn, errs=0, K=1000.0, beta=300.0, eps_min=0.001, tx_corte=0.15):
    n = tp + tn + fp + fn + errs
    if n == 0:
        return {"score_det": 0.0, "cut": False, "E": 0, "epsilon": 0.0,
                "failure_rate": 0.0, "rate_component": 0.0, "absolute_penalty": 0.0}
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
    box = np.load(DATA_DIR / "box_labels.npy")
    counts_k5 = np.load(DATA_DIR / "fraud_counts_k5.npy")
    test_idx = np.load(DATA_DIR / "router_test_indices.npy")

    ckpt = torch.load(DATA_DIR / "router.pt", weights_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Router(hidden=ckpt["hidden"], depth=ckpt["depth"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    Xt = torch.from_numpy(refs[test_idx]).to(device)
    y_box = box[test_idx]                              # 0/1/2
    y_fraud_count = counts_k5[test_idx]
    is_fraud_truth = (y_fraud_count >= 3)              # True = should deny
    expected_approved = ~is_fraud_truth

    with torch.no_grad():
        logits = model(Xt)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    pred = probs.argmax(axis=1)
    n_test = len(test_idx)

    print(f"[eval] test size: {n_test:,}")
    print(f"[eval] model parameters: {n_params:,}")
    print()

    # --- 1. 3x3 box confusion matrix --------------------------------------
    print("[eval] 3x3 box confusion matrix (rows = true class, columns = predicted):")
    header = f"{'':>14}" + "".join(f"{name:>12}" for name in ("pred A-L", "pred A-F", "pred B"))
    print(header + f"{'(row total)':>14}")
    rows_str = []
    for true_c, name in [(A_LEGIT, "true A-L"), (A_FRAUD, "true A-F"), (B, "true B")]:
        row_total = int((y_box == true_c).sum())
        cells = []
        for pred_c in (A_LEGIT, A_FRAUD, B):
            n_cell = int(np.sum((y_box == true_c) & (pred == pred_c)))
            cells.append(n_cell)
        line = f"{name:>14}" + "".join(f"{c:>12,}" for c in cells) + f"{row_total:>14,}"
        print(line)
        rows_str.append(line)
    print()

    # --- 2. Misroute analysis ---------------------------------------------
    true_b_mask = (y_box == B)
    pred_a_mask = (pred != B)
    misroute_mask = true_b_mask & pred_a_mask
    misroute_n = int(misroute_mask.sum())
    true_b_n = int(true_b_mask.sum())
    print("[eval] misroute analysis (true B but predicted as A — bypasses slow path):")
    print(f"        misroute count:        {misroute_n:,} / {true_b_n:,} true-B "
          f"({100*misroute_n/max(true_b_n,1):.2f}%)")
    # Of the misroutes, how many would the direct prediction get right anyway?
    if misroute_n > 0:
        direct_approved_mis = (pred[misroute_mask] == A_LEGIT)
        truth_approved_mis = expected_approved[misroute_mask]
        correct_anyway = int(np.sum(direct_approved_mis == truth_approved_mis))
        print(f"        of misroutes, correct anyway by direct argmax: "
              f"{correct_anyway:,} ({100*correct_anyway/misroute_n:.2f}%)")
        print(f"        of misroutes, wrong (actual rinha errors):     "
              f"{misroute_n - correct_anyway:,} ({100*(misroute_n-correct_anyway)/misroute_n:.2f}%)")
    print()

    # --- 3. End-to-end rinha sweep over tau --------------------------------
    def eval_with_tau(tau: float):
        route_slow = probs[:, B] > tau
        # If routed fast, output is A_LEGIT vs A_FRAUD argmax (ignoring P(B))
        fast_pred_legit = probs[:, A_LEGIT] >= probs[:, A_FRAUD]
        # final approved: slow path = truth; fast path = predicted A_LEGIT means approve
        final_approved = np.where(route_slow, expected_approved, fast_pred_legit)
        tp = int(np.sum(is_fraud_truth & ~final_approved))     # fraud denied
        tn = int(np.sum(expected_approved & final_approved))   # legit approved
        fp = int(np.sum(expected_approved & ~final_approved))  # legit denied
        fn = int(np.sum(is_fraud_truth & final_approved))      # fraud approved
        return tp, tn, fp, fn, int(route_slow.sum())

    rows = []
    print("[eval] P(B) threshold sweep (slow path = rinha oracle, perfect):")
    print(f"{'tau':>5} {'slow %':>8} {'TP':>8} {'TN':>8} {'FP':>8} {'FN':>8}"
          f" {'fail%':>7} {'eps':>8} {'rate':>7} {'pen':>7} {'score_det':>10}")
    print("-" * 95)
    for tau in [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70]:
        tp, tn, fp, fn, n_slow = eval_with_tau(tau)
        s = score_det(tp, tn, fp, fn)
        slow_rate = n_slow / n_test
        rows.append({"tau": tau, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                     "n_slow": n_slow, "slow_rate": slow_rate, **s})
        rate = s["rate_component"] if s["rate_component"] is not None else float("nan")
        pen = s["absolute_penalty"] if s["absolute_penalty"] is not None else float("nan")
        print(f"{tau:>5.2f} {100*slow_rate:>7.2f}% {tp:>8,} {tn:>8,} {fp:>8,} {fn:>8,}"
              f" {s['failure_rate']*100:>6.2f}% {s['epsilon']:>8.4f}"
              f" {rate:>7.0f} {pen:>7.0f} {s['score_det']:>10.2f}")

    best = max(rows, key=lambda r: r["score_det"])
    print()
    print(f"[eval] BEST score_det = {best['score_det']:.2f} at tau = {best['tau']}")
    print(f"       slow path used: {100*best['slow_rate']:.2f}% of queries")
    print()
    print("[eval] interpretation:")
    print("       this is the OPTIMISTIC ceiling. It assumes the slow path returns")
    print("       the exact rinha k=5 verdict (which it does by definition only if")
    print("       you actually run k=5 over the full 3M references). Any subset")
    print("       slow path will trade off some score_det for less memory / latency.")
    print()
    print("       a routing rate of e.g. 5% means 5% of queries go to slow path —")
    print("       at 900 RPS that's 45 RPS hitting the slower handler.")

    out = {
        "n_test": n_test,
        "n_params": n_params,
        "rows": rows,
        "best": best,
        "max_possible_score_det": 3000.0,
        "min_possible_score_det": -3000.0,
        "notes": [
            "Optimistic ceiling: assumes slow path is perfect (== rinha oracle).",
            "Misroute = true B predicted as A. Those queries bypass the slow",
            "  path and rely on the direct argmax(P(A-L), P(A-F)) prediction.",
            "Leave-one-out optimistic: real queries may have different",
            "  distribution from the references used as queries here.",
        ],
    }
    (DATA_DIR / "router_results.json").write_text(json.dumps(out, indent=2))
    print(f"\n[eval] wrote {DATA_DIR / 'router_results.json'}")


if __name__ == "__main__":
    main()
