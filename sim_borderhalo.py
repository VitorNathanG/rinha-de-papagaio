"""
For each candidate threshold D, build a "border halo" extension of Box-B:
    new_box_b = Box-B(k=25)  UNION  { ref in A : nearest_opp_dist(ref) < D }

Then replay our 5 confirmed mismatches against the new subset (brute-force
k=5) and report:
    - how many of the 5 are now correctly classified
    - projected score (fp/fn weighted as in test.js: E = fp*1 + fn*3)
"""
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
DATA = ROOT / "data"


def vectorize(req):
    """Same logic as backend/src/vectorize.rs."""
    tx, cust, merch, term = req["transaction"], req["customer"], req["merchant"], req["terminal"]
    last = req.get("last_transaction")
    requested = datetime.fromisoformat(tx["requested_at"].replace("Z", "+00:00"))
    if last:
        last_ts = datetime.fromisoformat(last["timestamp"].replace("Z", "+00:00"))
        mins = (requested - last_ts).total_seconds() / 60
        km_last = last["km_from_current"]
    else:
        mins, km_last = -1, -1
    mcc_risk = {"5411": .15, "5812": .30, "5912": .20, "5944": .45,
                "7801": .80, "7802": .75, "7995": .85, "4511": .35,
                "5311": .25, "5999": .50}.get(merch["mcc"], .50)

    def c(x):
        return max(0.0, min(1.0, x))

    v = np.zeros(14, dtype=np.float32)
    v[0] = c(tx["amount"] / 10000)
    v[1] = c(tx["installments"] / 12)
    v[2] = c((tx["amount"] / cust["avg_amount"]) / 10)
    v[3] = requested.hour / 23.0
    v[4] = requested.weekday() / 6.0
    if mins < 0:
        v[5] = v[6] = -1.0
    else:
        v[5] = c(mins / 1440)
        v[6] = c(km_last / 1000)
    v[7] = c(term["km_from_home"] / 1000)
    v[8] = c(cust["tx_count_24h"] / 20)
    v[9] = float(term["is_online"])
    v[10] = float(term["card_present"])
    v[11] = 0.0 if merch["id"] in cust["known_merchants"] else 1.0
    v[12] = mcc_risk
    v[13] = c(merch["avg_amount"] / 10000)
    return v


def main():
    print("[sim] loading refs / labels / box / opp ...", flush=True)
    refs = np.load(DATA / "references.npy").astype(np.float32)
    labels = np.load(DATA / "labels.npy")
    box = np.load(DATA / "box_labels.npy")
    opp = np.load(DATA / "nearest_opp_dist.npy")
    in_A = (box != 2)

    mismatches = json.loads((DATA / "mismatches.json").read_text())
    queries = [(m, vectorize(m["request"])) for m in mismatches]

    THRESHOLDS = [None, 0.18, 0.20, 0.22, 0.25, 0.30, 0.40]

    print(f"{'D':>6}  {'Box-B':>8}  {'5-NN fixes':>12}  {'fp':>3}  {'fn':>3}  "
          f"{'E':>3}  {'score':>6}", flush=True)
    print("-" * 60, flush=True)

    for D in THRESHOLDS:
        if D is None:
            new_box = (box == 2)
            label_str = "—"
        else:
            new_box = (box == 2) | (in_A & (opp < D))
            label_str = f"{D:.3f}"

        idx = np.where(new_box)[0]
        refs_sub = refs[idx]
        labels_sub = labels[idx]

        fixes = 0
        fp = 0
        fn = 0
        for m, v in queries:
            d2 = np.sum((refs_sub - v) ** 2, axis=1)
            top5 = np.argpartition(d2, 5)[:5]
            count = int(labels_sub[top5].sum())
            approved = count < 3
            expected = m["expected_approved"]
            if approved == expected:
                fixes += 1
            else:
                if expected and not approved:
                    fp += 1
                else:
                    fn += 1

        # Rinha scoring (test.js): E = fp*1 + fn*3 + errs*5; abs_penalty = -300*log10(1+E)
        # rate_component saturates at 3000 (max), so detection = 3000 - 300*log10(1+E)
        # provided failure_rate < 0.15 (we always are).
        E = fp * 1 + fn * 3
        det = 3000 - 300 * math.log10(1 + E) if E >= 0 else 3000
        # p99 already saturates at 3000 → total = 3000 + det
        total = round(3000 + det)
        print(f"  {label_str:>5}  {len(idx):>8,}  {fixes:>2}/{len(queries):>2} fixed  "
              f"{fp:>3}  {fn:>3}  {E:>3}  {total:>6}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
