# Papagaio — distilling 5-NN into a small model

A feasibility study for an alternative approach to the **Rinha de Backend 2026**
fraud-detection challenge. Instead of running k-NN over the 3M-vector
reference set at request time, we train a small parametric model that
*imitates* the k-NN's verdict — the "papagaio" (parrot) that learned the
oracle's answers by heart.

This document captures the approach, the empirical findings that shaped it,
and the open work.

---

## Table of contents

1. [The challenge in one paragraph](#the-challenge-in-one-paragraph)
2. [The alternative: distill the oracle](#the-alternative-distill-the-oracle)
3. [Empirical finding that shapes everything](#empirical-finding-that-shapes-everything)
4. [Two-box architecture](#two-box-architecture)
5. [Inference routing — the critical question](#inference-routing--the-critical-question)
6. [Why we believe this beats a pure ANN](#why-we-believe-this-beats-a-pure-ann)
7. ["Why not X" — alternatives we considered](#why-not-x--alternatives-we-considered)
8. [Status](#status)
9. [Open questions and risks](#open-questions-and-risks)
10. [Bail-out signals](#bail-out-signals)
11. [Repo layout](#repo-layout)

---

## The challenge in one paragraph

The Rinha 2026 task receives a credit-card transaction, vectorizes it into 14
normalized dimensions, finds the 5 nearest neighbors in a fixed reference set
of 3,000,000 labeled vectors, and computes `fraud_score = fraud_count_5 / 5`.
If `fraud_score >= 0.6`, the transaction is denied. The submitted service runs
inside a hard budget — **1 CPU and 350 MB RAM** across all containers,
handling up to **900 RPS peak** — and scoring heavily rewards low p99
latency (each 10× improvement is worth +1000 points up to +3000 at p99 ≤ 1 ms,
balanced against a detection score that punishes false negatives at 3× false
positives and HTTP errors at 5×).

The "standard" play is some flavor of ANN index (HNSW, IVF, VP-tree). Those
work, but they all involve carrying the 3M vectors (or a derived structure) in
memory and doing a similarity search per request.

## The alternative: distill the oracle

The 5-NN over a fixed reference set is a **deterministic function** of the
query: R^14 → {approve, deny}. Anything deterministic can be approximated by
a parametric model.

The plan:

1. **Offline** — run the slow oracle (brute-force 5-NN over the 3M) on a huge
   sample of query vectors. The result is a labeled set of (query, verdict)
   pairs.
2. **Train** a small MLP to predict the verdict from the query alone.
3. **Deploy** — ship only the model weights. No reference vectors, no index
   in the container. Inference is a few microseconds of matmul.

In ML lingo this is **knowledge distillation** of a non-parametric classifier
into a parametric one. The trick is that the offline phase has no time budget
(the references are fixed, available at build time), so we can spend hours
generating training labels and the runtime never pays for it.

Trade-offs:
- **Pro** — latency dominated by matmul; saturates `score_p99` at +3000.
- **Pro** — memory dominated by model weights; trivially within budget.
- **Con** — `score_det` depends on how well the MLP imitates k-NN. Boundary
  errors (false approvals / false denials) eat directly into the score.

This study exists to measure whether the detection trade-off is acceptable.

## Empirical finding that shapes everything

We ran leave-one-out k-NN over all 3M references at k=5 and k=25 (GPU
brute-force, ~5 min on an RX 9070 XT). For each reference vector we counted
how many of its k nearest neighbors (excluding itself) are labeled fraud.

### k=5 distribution

```
0/5:  1,951,240  (65.04%)   pure legit cluster
1/5:     16,476  ( 0.55%)
2/5:     33,005  ( 1.10%)
3/5:     32,810  ( 1.09%)
4/5:     16,337  ( 0.54%)
5/5:    950,132  (31.67%)   pure fraud cluster

pure (count==0 or count==5):  96.71%
non-pure:                      3.28%
edge zone (count==2 or 3):     2.19%
```

### k=25 distribution

```
0/25:   1,947,958  (64.93%)
1..24/25 combined:   105,472  ( 3.52%)   peaks at 12/25 = 16,109 and 13/25 = 16,079
25/25:    946,570  (31.55%)

pure (count==0 or count==25):  96.48%
non-pure:                       3.52%
```

### Interpretation

1. **The data is strongly clustered.** Two large, homogeneous regions
   separated by a thin transition zone.
2. **The clustering is real, not an artifact of small k.** Going from k=5 to
   k=25 only converts 0.23% of points from "pure" to "non-pure" — the
   boundary is genuinely thin in input space.
3. **The non-pure points at k=25 concentrate at the 50/50 mark (12-13/25),**
   confirming a real decision surface rather than a gradient.
4. **96.5% of the problem is trivial.** Any reasonable classifier — even
   logistic regression — handles cluster interiors with near-perfect accuracy.
5. **The contest is decided in the 3.5% boundary.** This is where the errors
   come from, and where scoring punishes you.

This shapes a two-track strategy rather than a single-model one.

## Two-box architecture

We split the 3M reference set by **structure**, not just by label.

### Box A — bulk clusters (~96.5%)

References where the point's own label agrees fully with its k=25
neighborhood:

- `label == legit` AND `count_25 == 0`
- `label == fraud` AND `count_25 == 25`

These are points deep inside their cluster. An MLP trained only on them gets
a clean signal — no boundary noise — and learns the cluster regions of input
space well.

Box A handles the **fast path** at inference. Input: the 14-dim query
vector. Output: a single probability. A few microseconds.

### Box B — boundary, outliers, and a spatial halo (~3.5% + halo)

Everything that isn't Box A:

1. **Mixed neighborhoods** — `count_25 in 1..24`. Pure boundary points.
2. **Label outliers** — points whose label disagrees with a fully homogeneous
   neighborhood:
   - `label == fraud` AND `count_25 == 0` — fraud isolated in legit territory.
   - `label == legit` AND `count_25 == 25` — legit isolated in fraud territory.
3. **Spatial halo** — for each Box B point, include its m closest reference
   neighbors (even if those neighbors are themselves Box A). The halo gives
   the boundary handler local context, and widens the spatial footprint of
   Box B so routing (next section) is more forgiving.

Box B is handled by a **slower but more careful mechanism**. Candidates we
haven't picked between yet:

- Brute-force k=5 over Box B (after halo, ~150k × 14 floats ≈ 8 MB,
  estimated ~1.5 ms per query at 1 CPU).
- A second, larger MLP trained specifically on Box B targets.
- A small ANN index (HNSW or VP-tree) over Box B.

Choice depends on validation results.

## Inference routing — the critical question

A query at request time does not have a label. We can't apply
`label == count` directly. We need a proxy.

### Why model confidence alone isn't enough

The Box A MLP is trained only on cluster interiors and never sees ambiguous
examples. Its confidence is calibrated where the answer is obvious. **At the
boundary in feature space**, the MLP can be confidently wrong because it has
no training signal to be uncertain there. Routing solely on `prob` would miss
the cases that matter most.

### The fix: distance to Box B

Box B is a *spatial* structure — the references at or near the decision
boundary, plus the halo. A query close to any Box B point is close to the
boundary; a query far from all of them is in cluster interior.

```
[query 14-dim]
    │
    ▼
[KD-tree over Box B + halo, ~150k points]
    │
    ├── nearest_distance > radius ─► [Box A MLP]   ─► response
    │
    └── nearest_distance ≤ radius ─► [slow path]   ─► response
```

The KD-tree lookup is O(log N) with cheap distance comps — microseconds. The
routing cost is essentially zero.

### Why this works geometrically

A natural worry: "what if a Box B point is *inside* Box A territory and the
Box A model says fraud/legit confidently when the truth disagrees?"

That scenario happens near the boundary — but **"near the boundary" means
spatially close to a Box B point**, which the routing catches. The halo
widens Box B's footprint so a query slightly off the exact boundary still
falls within the halo radius.

The remaining failure mode: a query in the *deep interior* of a cluster that
the Box A MLP gets wrong anyway. In principle possible, but if the model is
well-trained on the clean Box A signal, queries in cluster interiors have
their true k=5 verdict agree with the cluster label — so MLP and oracle
align there.

### Picking the radius

Empirical. The plan:

1. Build Box A model + Box B + halo.
2. Generate a holdout of queries with ground-truth k=5 verdicts.
3. For each query, compute `(error_of_box_A, distance_to_nearest_box_B)`.
4. Plot error rate vs. distance. Expect a curve where error is high for small
   distance and drops fast as distance grows.
5. Pick the smallest radius where error is below the score budget.

A wider radius routes more queries to the slow path: better detection but
worse p99. The optimum depends on slow-path latency.

## Why we believe this beats a pure ANN

Quantitatively:

- **p99** — distillation runs ~0.1 ms vs. ~1-5 ms for ANN. Difference of
  +1000 to +2000 on `score_p99`.
- **score_det** — comparable to ANN if routing is reliable; degrades smoothly
  if we trade some routing precision for speed.
- **Memory** — model is a few MB vs. ANN index of ~50-200 MB. Frees budget
  for other services and reduces cache thrash.

Qualitatively:

- Heavy work happens at build time. The runtime is simple and predictable.
- The split mirrors the underlying data structure rather than fighting it.

## "Why not X" — alternatives we considered

**Why not use HNSW / IVF / VP-tree?** They're perfectly valid; they're what
most submissions will use. The reason to try something else: with 1 CPU
shared between LB + ≥2 API replicas, the per-request budget is ~1-2 ms.
ANN puts the work *at* request time; distillation moves it *out* of request
time. The p99 ceiling matters a lot in this scoring scheme.

**Why not train one big MLP on everything?** The boundary points are 3.5% of
the data; in normal training they'd be drowned out by the easy 96.5%. Loss
optimizers don't naturally spend capacity where it matters. Splitting
forces the architecture to allocate effort by region.

**Why not memorize the 3M as a lookup table?** They aren't (query, answer)
pairs — they're labeled vectors. The k=5 answer for an arbitrary query is a
function of the 5 nearest neighbors, which requires a search. We *can* derive
3M (query, answer) pairs via leave-one-out (which is what `label.py` does),
and that's our training set. But the runtime cannot do constant-time lookup
on a continuous 14-dim space without an index — that's literally what ANN is.
Distillation is the part that lets us *skip* the index.

**Why not random forest / GBDT instead of MLP?** Plausible, especially given
14 well-bounded numeric features. We picked MLP because (a) GPU training is
trivial in PyTorch, (b) inference vectorizes nicely with a small matmul,
(c) the bimodal data structure suggests the decision surface is mostly
piecewise-flat, which any reasonable model class handles. GBDT is a fallback
if MLP can't reach the target accuracy.

**Why leave-one-out on the references instead of synthetic queries?** It's
the highest-quality training signal we have (real k=5 verdicts on 3M points),
and the references span the actual data distribution. The risk is that real
test queries come from a slightly different distribution. We plan to
cross-check on synthetic queries derived from `example-payloads.json`.

## Status

### Built

- `prepare.py` — decompresses `references.json.gz` into compact `numpy`
  arrays (3M × 14 float32 + 3M bool). Idempotent.
- `label.py` — GPU brute-force k-NN, chunked top-K merge for memory
  efficiency, fp16 with fp32 fallback, parameterized by K. Outputs
  `fraud_counts_k{K}.npy`. We have k=5 and k=25 for all 3M.
- `train.py` — binary MLP with BCE + `pos_weight=3` (matches the FN/FP cost
  asymmetry). Adam, ~12 epochs, configurable hidden size and depth.
- `evaluate.py` — threshold sweep over the held-out test split with the exact
  `score_det` formula from the rinha rules.
- `run.py` / `run.sh` — orchestration with uv-managed deps; targets
  ROCm 7.2 (RDNA4) by default.

### Not yet built

- Box A / Box B partition script (combines `labels.npy` and
  `fraud_counts_k25.npy`).
- Spatial halo expansion.
- KD-tree or alternative spatial index over Box B + halo.
- Slow-path handler.
- Holdout-based radius tuning (`validate_subset.py`).
- End-to-end p99 + `score_det` benchmark in a real container with the rinha
  test harness.

### Numbers we have so far

- Reference set: 3,000,000 vectors, 33.31% fraud labels.
- Box A candidate size: ~96.5% of references (~2.9M points).
- Box B candidate size: ~3.5% of references (~105k points) before halo.
- Edge zone at k=5 (count in {2, 3}): 2.19% of references.

## Open questions and risks

- **Distribution shift between training and test.** Leave-one-out uses
  references as queries; the actual rinha test uses synthetic payloads.
  Mitigation: validate on `example-payloads.json` and synthetic queries.
- **Halo size.** Top-m spatial neighbors per Box B point, or a fixed radius?
  Trade-off between Box B memory footprint and routing reliability.
- **Slow-path latency at scale.** At 900 RPS and a 3.5% boundary,
  the slow path handles ~30 RPS. Each request must finish in time to keep
  p99 reasonable. Brute-force over ~150k × 14 should fit in ~1.5 ms;
  that's the operating budget.
- **MLP overconfidence at boundary.** If routing fails (query lands in Box A
  but is actually near boundary), the MLP can be confidently wrong.
  Distance-based routing is the safety net; we need to validate it catches
  enough cases.
- **The training target uses k=5 leave-one-out, not synthetic queries.**
  References-as-queries is an *optimistic* upper bound — real queries may sit
  in regions the references don't cover well.

## Bail-out signals

If any of the following hold after the next round of measurement, abandon
distillation and ship a plain ANN:

- Box B + halo doesn't catch ≥ 99% of the queries that the Box A MLP gets
  wrong.
- Offline `score_det` on a representative holdout is < 1500.
- After a complete build, the rinha test gives meaningfully worse
  `score_det` than a baseline HNSW submission would.

## Repo layout

```
papagaio/
├── README.md            this file
├── pyproject.toml       uv project config, ROCm 7.2 torch by default
├── uv.lock              pinned deps (committed)
├── run.sh               install uv + sync deps + run pipeline
├── run.py               orchestrator: prepare → label → train → evaluate
├── prepare.py           step 1 — decompress references to numpy
├── label.py             step 2 — leave-one-out k-NN on GPU
├── train.py             step 3 — train the papagaio MLP
├── evaluate.py          step 4 — confusion matrix + simulated score_det
├── data/                .gitignored — generated artifacts
│   ├── references.npy         (3M × 14 float32)
│   ├── labels.npy             (3M bool)
│   ├── fraud_counts_k5.npy    (3M uint8, values 0..5)
│   └── fraud_counts_k25.npy   (3M uint8, values 0..25)
└── logs/                .gitignored — timestamped run logs
```

## Reading order for new contributors

1. The rinha challenge docs in `../rinha-de-backend-2026/docs/en/` — at
   minimum `VECTOR_SEARCH.md`, `DETECTION_RULES.md`, and `EVALUATION.md`.
2. This file.
3. `prepare.py` and `label.py` — the data layer.
4. `train.py` and `evaluate.py` — the distillation layer.
5. Whatever scripts get added next for Box A/B partition and routing.
