# Papagaio

Submissão para a [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026) —
detecção de fraude por 5-NN sobre 3M vetores de referência, budget de **1 CPU / 350 MB** e meta de
**p99 ≤ 1 ms**.

Em vez de carregar um índice ANN sobre os 3M refs em runtime, este projeto **destila** o
classificador 5-NN num MLP minúsculo (14 → 32 → 32 → 3, 1.635 params, 6.5 KB) que roteia ~3-7%
das queries para um slow path IVF sobre um subset curado (Box-B, ~213k refs). O resto é resolvido
direto pelo argmax do router.

## Resultado

```
Score:    6000 / 6000   (TP=24037 TN=30023 FP=0 FN=0 errors=0)
p99:      0.13 ms (test oficial rinha) / 0.15 ms (peak.js local)
Memory:   ~270 MB / 350 MB budget
```

## Quickstart

Pré-requisitos: `uv`, GPU (ROCm 7.2 ou CUDA — ajustar `pyproject.toml`), Docker, `k6`. O dataset
e o harness oficial são esperados em `../rinha-de-backend-2026/` (repo irmão).

```bash
# 1. Pipeline offline (gera data/router_weights.bin + data/box_b_*.bin)
uv run python prepare.py
K=25 uv run python label.py
uv run python partition.py
uv run python nearest_opp.py
D=0.23 uv run python border_halo.py
uv run python export_box_b.py
uv run python train_router.py
uv run python export_router.py

# 2. Stack de produção
docker compose up --build

# 3. Test oficial
k6 run /projects/rinha-de-backend-2026/test/test.js
```

Detalhes operacionais completos em [docs/08 — Pipeline de dados](./docs/08-pipeline-de-dados.md).

## Documentação

Os docs estão organizados em ordem de leitura recomendada — cada um se sustenta sozinho, mas a
sequência conta uma história coerente.

### Por que esse approach faz sentido

- **[01 — Visão geral](./docs/01-visao-geral.md)**: o desafio, as restrições, por que escolhemos
  destilação em vez de ANN puro, visão de alto nível da arquitetura e dos trade-offs.

- **[02 — Estrutura dos dados](./docs/02-estrutura-dos-dados.md)**: descobertas empíricas que
  motivam tudo. 96.5% dos refs vivem em vizinhanças totalmente homogêneas; a fronteira de classe
  é fina e real (pico em 50/50). Existe um **noise floor BORDERLINE** (~3% de refs com label
  aleatório) que define quanto score é fisicamente alcançável.

### Decisões arquiteturais

- **[03 — Partição e border halo](./docs/03-particao-e-halo.md)**: divide os 3M refs em A-Legit /
  A-Fraud / B (com base em `count_k25`). O bug do "inward purity" — refs no contorno externo de
  cluster puro são missados se a partição usa só k-NN do próprio ref. A solução: **nearest-opp
  halo**, que computa para cada ref a distância ao ref de label oposto mais próximo e promove
  para B aqueles abaixo de um threshold (D=0.23).

- **[04 — Router MLP](./docs/04-router-mlp.md)**: o classificador de 3 classes que substitui
  router + Box-A model separados. Class weights inverse-frequency (sem isso o modelo colapsa em
  "predict A trivially"), GELU(approximate="tanh"), early stopping. Recall_B na convergência
  ≈ 99.93%.

- **[05 — Slow path IVF](./docs/05-slow-path-ivf.md)**: o kernel IVF (NLIST=256, NPROBE=16) que
  cabe em L2 do Mac Mini target. Layout CSR (refs sorted por cluster, padded para 16 floats/linha
  = 1 cache line). AVX2+FMA distance, fixed-size insertion sort para top-5, software prefetch
  8 ahead. Por que não brute force, HNSW, ou VP-tree.

- **[06 — Backend Rust](./docs/06-backend-rust.md)**: hyper 1.x direto (sem axum, sem serde),
  byte parser zero-alloc, respostas pré-renderizadas como `&'static [u8]`, mmap + page touch,
  UNIX domain socket entre nginx e APIs, `current_thread` tokio. A jornada de p99 de 0.30 → 0.15 ms.

### Guardrails

- **[07 — Guardrails numéricos](./docs/07-guardrails-numericos.md)**: quatro pontos onde a
  numérica do treino tem que casar exatamente com a do runtime. **round4** no vectorize (match
  com o data generator), **GELU(tanh) coupling** entre train e infer, **Padé[7/6] tanh inline**
  vs `libm::tanhf`, shape do router congelada em hidden=32 / depth=2 em ambos os lados. Cada um
  foi descoberto por um bug que custou pontos.

### Operação

- **[08 — Pipeline de dados](./docs/08-pipeline-de-dados.md)**: o que cada script Python faz,
  em que ordem, com quais env vars. Inclui o reset do zero, localização de arquivos gerados,
  e tempo estimado de cada etapa.

- **[09 — Debug e profiling](./docs/09-debug-e-profiling.md)**: workflow de diagnóstico de
  mismatch via `find_mismatches.py` + `analyze_mismatches.py`. Microbenches isolados
  (`ivf_bench`, `bench`), sweep de parâmetros (`run_sweep.sh`), profiling com perf + flamegraph
  (`profile.sh`), e os testes k6 (peak.js, profile.js, test.js oficial).

### História

- **[10 — Experimentos descartados](./docs/10-experimentos-descartados.md)**: o que tentamos e
  abandonamos. Geometric halo, frequency-weighted sampling, prune-from-full, axum, serde_json,
  multi-thread tokio dentro do container, custom LB, HNSW. Para cada um, **por que não
  funcionou**.

- **[11 — Evolução do score](./docs/11-evolucao-do-score.md)**: cronologia commit-a-commit do
  score (5700 → 5910 → 6000) e do p99 (0.30 → 0.15 ms). Por que 5700 parecia um plateau e por
  que não era.

## Estrutura do repo

```
papagaio/
├── README.md                ← este arquivo
├── CLAUDE.md                ← orientação para o Claude Code
├── LATENCY_BACKLOG.md       ← otimizações de latência pendentes (priorizadas por ROI)
├── docs/                    ← documentação técnica (este link)
│
├── pyproject.toml           ← uv project (torch ROCm 7.2 default)
├── uv.lock
├── run.sh / run.py          ← pipeline antigo (feasibility study; não cobre produção)
├── docker-compose.yml       ← LB + 2 réplicas, 1.0 CPU / 270 MB total
├── nginx.conf               ← stream mode + UDS upstream
│
├── prepare.py               ← step 1: references.json.gz → numpy
├── label.py                 ← step 2: GPU leave-one-out k-NN
├── partition.py             ← step 3: labels + counts → 3-class box
├── nearest_opp.py           ← step 4: GPU dist até label oposto
├── border_halo.py           ← step 5: promove refs A → B no contorno
├── export_box_b.py          ← step 6: k-means + serializa binários
├── train_router.py          ← step 7: treina MLP de 3 classes
├── export_router.py         ← step 8: serializa pesos para Rust
├── sim_borderhalo.py        ← calibração offline do D do halo
├── find_mismatches.py       ← varre test-data.json contra backend
├── analyze_mismatches.py    ← caracteriza cada mismatch
│
├── backend/                 ← API Rust (hyper direto, byte parser, AVX2)
│   ├── Cargo.toml
│   ├── Dockerfile
│   └── src/
│       ├── main.rs          ← hyper service, mmap loader, UDS/TCP listener
│       ├── vectorize.rs     ← byte parser → [f32; 16] (com round4)
│       ├── router.rs        ← MLP forward + Padé tanh + softmax
│       └── slow_path.rs     ← kernel IVF (AVX2 + FMA)
│
├── bench/                   ← microbench brute-force k=5 (ground truth p/ ivf_bench)
├── ivf_bench/               ← microbench do kernel IVF isolado
├── sweep_ivf.py             ← sweep NLIST × NPROBE (builda variantes)
├── run_sweep.sh             ← roda ivf_bench sobre o grid
├── profile.sh               ← carga sustentada + perf + flamegraph
│
├── test/
│   ├── peak.js              ← smoke 6s @ 900 RPS, p50/p90/p99/p99.9
│   └── profile.js           ← carga sustentada para profile (default 10k RPS)
│
├── data/                    ← .gitignored — todos os artefatos gerados
└── logs/                    ← .gitignored — logs do run.sh
```

## Repos relacionados

- `../rinha-de-backend-2026/` — spec, dataset, test harness oficial. Referenciado por **caminho
  relativo** (não é submodule). `docs/en/DETECTION_RULES.md`, `DATASET.md`, `EVALUATION.md` são
  ground truth.

## Língua

A documentação e os comentários do código estão em **português**, com termos técnicos preservados
em inglês (`mmap`, `cache line`, `k-NN`, `softmax`, `Padé`, etc.). Identificadores, env vars e
strings que outras ferramentas consomem **não são traduzidos**.
