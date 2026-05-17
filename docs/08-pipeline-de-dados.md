# 08 — Pipeline de dados

Esta seção é a referência operacional: o que cada script faz, em que ordem rodar, e quais env
vars controlam o comportamento. Para o "por que" de cada decisão, veja os docs específicos linkados.

## Ordem obrigatória do pipeline de produção

```
prepare.py
    │
    ▼
label.py (K=25)                          ← K=5 também é útil para sanidade
    │
    ▼
partition.py                             ← combina labels + counts_k25
    │
    ▼
nearest_opp.py                           ← GPU, ~5 min
    │
    ▼
border_halo.py (D=0.23)                  ← promove A → B
    │
    ▼
export_box_b.py (NLIST=256)              ← k-means + serialização
    │
    ▼
train_router.py                          ← MLP de 3 classes
    │
    ▼
export_router.py                         ← pesos → router_weights.bin
```

**Não existe orquestrador único** para essa sequência. `run.py` / `run.sh` cobrem só o pipeline
antigo (`prepare → label → train → evaluate`), que serviu para a feasibility study inicial e
está stale em relação ao stack de produção.

## prepare.py

**Função**: descompacta `references.json.gz` (do repo irmão da rinha) em arrays numpy compactos.

```bash
uv run python prepare.py
```

Inputs:

- `../rinha-de-backend-2026/resources/references.json.gz` (default, override com `REFERENCES_GZ`)

Outputs:

- `data/references.npy` — `(N, 14) float32`, ~168 MB para N=3M
- `data/labels.npy` — `(N,) bool`, True = fraud

Idempotente — pula se os outputs já existem. Tempo: ~30 s para 3M entries.

## label.py

**Função**: brute-force k-NN leave-one-out em GPU, retorna a contagem de fraud labels entre os
k vizinhos mais próximos. Usado tanto para sanity check (K=5) quanto como input da partição (K=25).

```bash
K=25 uv run python label.py     # essencial para a partição
K=5  uv run python label.py     # opcional, sanity
```

Env vars principais:

| Var      | Default | Significado                                            |
|----------|---------|--------------------------------------------------------|
| `K`      | 5       | Número de vizinhos por query (1..65535)                |
| `BATCH`  | 2048    | Queries por chunk (controla VRAM de saída)             |
| `CHUNK`  | 16384   | Refs por chunk interno (controla peak distance matrix) |
| `SUBSET` | 0       | Processar só os primeiros N (0 = todos)                |
| `FP32`   | -       | Se setado, força fp32 (default é fp16)                 |
| `FORCE`  | -       | Sobrescreve output existente                           |

Output: `data/fraud_counts_k{K}.npy` — `(N,) uint8` (ou `uint16` se K > 255), valores em `0..K`.

Tempo (RX 9070 XT, ROCm 7.2):

- K=5 sobre 3M: ~4 min 20 s
- K=25 sobre 3M: ~4 min 22 s

Algoritmo: chunked top-K merge. Cada outer batch mantém um running top-K `(BATCH, K)`; os refs
são processados em tiles `(BATCH, CHUNK)` mergeados via `torch.topk` na concatenação. Detalhes
em [02 — Estrutura dos dados](./02-estrutura-dos-dados.md).

## partition.py

**Função**: combina `labels.npy` e `fraud_counts_k25.npy` em uma classificação 3-class.

```bash
uv run python partition.py
```

Env vars:

| Var     | Default | Significado                          |
|---------|---------|--------------------------------------|
| `FORCE` | -       | Sobrescreve `box_labels.npy` existente |

Output: `data/box_labels.npy` — `(N,) uint8`, valores 0/1/2.

Classes (ver [03 — Partição e halo](./03-particao-e-halo.md)):

- 0 = **A-Legit** — `label == legit` AND `count_k25 == 0`
- 1 = **A-Fraud** — `label == fraud` AND `count_k25 == 25`
- 2 = **B** — qualquer outra coisa

Idempotente. Tempo: < 1 s.

## nearest_opp.py

**Função**: para cada ref, computa a distância euclidiana até o ref *de label oposto* mais
próximo. Métrica direta de "está perto da fronteira de classe?".

```bash
uv run python nearest_opp.py
```

Env vars:

| Var            | Default | Significado                                  |
|----------------|---------|----------------------------------------------|
| `BATCH`        | 2048    | Outer batch de queries por kernel launch     |
| `INNER`        | 131072  | Inner chunk do "other class" (controla VRAM) |
| `SKIP_COMPUTE` | -       | Pula a parte de compute, só faz o sweep      |

Output: `data/nearest_opp_dist.npy` — `(N,) float32`, distância euclidiana raw.

Tempo: ~5 min em GPU. Brute-force chunked nos dois eixos para manter VRAM intermediária < 1 GB.

## border_halo.py

**Função**: promove para Box-B todos os refs em A cujo `nearest_opp < D`. Idempotente — guarda
um snapshot da partição original em `box_labels.before_halo.npy`.

```bash
D=0.23 uv run python border_halo.py    # re-tuna trocando D
```

Env vars:

| Var | Default | Significado                                     |
|-----|---------|-------------------------------------------------|
| `D` | 0.23    | Threshold de promoção (refs com opp < D viram B) |

Inputs:

- `data/box_labels.npy` (output do partition.py — usado só na primeira execução)
- `data/nearest_opp_dist.npy` (output do nearest_opp.py)

Outputs:

- `data/box_labels.npy` — atualizado com promoções aplicadas
- `data/box_labels.before_halo.npy` — snapshot pré-halo (criado na primeira execução)

Re-rodar com D diferente lê do snapshot, não acumula promoções. Tempo: < 1 s.

Calibração: o default `D=0.23` foi escolhido com `sim_borderhalo.py`, que faz sweep de D contra
os 5 mismatches conhecidos do `test-data.json`. D=0.23 é o menor valor que recupera todos os 11
refs faltando no top-5 NN.

## export_box_b.py

**Função**: roda k-means sobre o subset Box-B, sorta refs por cluster, e escreve os 4 binários
que o backend Rust mmaps na subida.

```bash
NLIST=256 ITER=20 uv run python export_box_b.py
```

Env vars:

| Var     | Default | Significado                              |
|---------|---------|------------------------------------------|
| `NLIST` | 256     | Número de clusters do IVF                |
| `ITER`  | 20      | Iterações de Lloyd's k-means             |
| `SEED`  | 42      | Seed do init aleatório do k-means        |

Inputs:

- `data/references.npy`
- `data/labels.npy`
- `data/box_labels.npy` (filtra apenas onde `box == 2`)

Outputs:

- `data/box_b_refs.bin` — `(N_B, 16) float32`, padded, sorted por cluster
- `data/box_b_labels.bin` — `(N_B,) uint8`, sorted na mesma ordem
- `data/box_b_ivf_centroids.bin` — `(NLIST, 16) float32`, padded
- `data/box_b_ivf_offsets.bin` — `(NLIST+1,) uint32`, CSR offsets *em linhas*

Tempo: ~30 s a 1 min (k-means em numpy CPU). Em ROCm GPU é mais rápido mas o paralelismo do
numpy já é suficiente. Detalhes do layout em [05 — Slow path IVF](./05-slow-path-ivf.md).

## train_router.py

**Função**: treina o MLP de 3 classes (14 → 32 → 32 → 3).

```bash
uv run python train_router.py
EPOCHS=100 HIDDEN=32 DEPTH=2 uv run python train_router.py
```

Env vars:

| Var         | Default | Significado                                         |
|-------------|---------|-----------------------------------------------------|
| `HIDDEN`    | 32      | Largura das camadas escondidas (***NÃO MUDE — o export e o Rust assumem 32***) |
| `DEPTH`     | 2       | Número de camadas escondidas (***NÃO MUDE — idem***) |
| `LR`        | 1e-3    | Adam learning rate                                  |
| `EPOCHS`    | 500     | Cap de epochs (early stopping geralmente para muito antes) |
| `BATCH`     | 8192    | SGD minibatch                                       |
| `PATIENCE`  | 30      | Epochs sem melhora antes do early stop              |
| `MIN_DELTA` | 1e-5    | Tolerance de "melhora" no val_loss                  |
| `SEED`      | 42      | Seed de split + init                                |

Inputs:

- `data/references.npy`
- `data/box_labels.npy`

Outputs:

- `data/router.pt` — checkpoint do PyTorch (state_dict + metadata)
- `data/router_train_indices.npy` / `router_val_indices.npy` / `router_test_indices.npy` —
  split fixo (80/10/10, seed 42)

Tempo: ~1-2 min em GPU (~30s por epoch × ~40 epochs até early stop). Em CPU multiplica por ~10.

Métricas durante o treino:

```
ep  39/500  train_loss=0.0014  val_loss=0.0013  acc=99.98%
            recall=[A-L=99.99% A-F=99.99% B=99.93%]  elapsed=42s *
```

O `*` no final marca o melhor val_loss até aquele epoch — o state salvo no `.pt` é o do `*` mais
recente, não o último epoch. Mais detalhes em [04 — Router MLP](./04-router-mlp.md).

## export_router.py

**Função**: serializa os pesos do `router.pt` em raw binary little-endian para o loader Rust.

```bash
uv run python export_router.py
```

Sem env vars relevantes. Asserta que `hidden=32` e `depth=2` (o Rust assume essa shape).

Output: `data/router_weights.bin` — **1635 floats × 4 bytes = 6540 bytes**.

Ordem do flat array (importante!):

```
w1 (32, 14)   — 448 floats   (row-major: out × in)
b1 (32,)      —  32 floats
w2 (32, 32)   — 1024 floats
b2 (32,)      —  32 floats
w3 (3, 32)    —  96 floats
b3 (3,)       —   3 floats
              ────
              1635 floats total
```

O Rust loader em `backend/src/router.rs::load_weights` lê nessa exata ordem. Qualquer mudança de
shape ou ordem **quebra o backend silenciosamente** (sem panic, mas com vereditos lixo).

## Scripts auxiliares (não no pipeline crítico)

### `sim_borderhalo.py`

Sandbox de calibração: para um sweep de valores de D, monta um Box-B candidato e replaya os 5
mismatches conhecidos contra ele em brute-force, reportando fp/fn projetados. Foi usado para
escolher D=0.23 — o menor D que recupera todos os 11 refs faltando.

Útil para re-calibrar se o dataset mudar.

### `evaluate.py` / `evaluate_router.py`

Avaliação offline sobre split held-out. Computam confusion matrix e simulam o `score_det` com a
fórmula da rinha. Não substituem o test oficial do k6, mas servem para validar mudanças do
pipeline sem subir Docker.

### `find_mismatches.py` / `analyze_mismatches.py`

Workflow de debug por mismatch. Ver [09 — Debug e profiling](./09-debug-e-profiling.md).

### Pipeline antigo (run.py / run.sh)

`run.py` orquestra `prepare → label → train → evaluate`. Esse era o pipeline da feasibility study
inicial; foi mantido por compatibilidade mas **não produz os artefatos que o backend de produção
precisa**. Use os scripts individuais listados acima.

### Outras explorações descartadas

- `halo.py` — geometric halo (k-NN dos pontos de B). Descartado por não mover o noise floor.
- `sample_subset.py` — frequency-weighted sampling.
- `prune_subset.py` — top-N de uso global.
- `ivf.py` — IVF em GPU (versão de pesquisa, não usado em produção).

Detalhes do que foi tentado e por que foi descartado em
[10 — Experimentos descartados](./10-experimentos-descartados.md).

## Reset / re-build do zero

Se você quer regenerar tudo do zero:

```bash
rm -rf data/*.npy data/*.bin data/*.pt data/box_labels.before_halo.npy

uv run python prepare.py            # ~30s
K=25 uv run python label.py         # ~5 min em GPU
uv run python partition.py          # < 1s
uv run python nearest_opp.py        # ~5 min em GPU
D=0.23 uv run python border_halo.py # < 1s
uv run python export_box_b.py       # ~1 min
uv run python train_router.py       # ~2 min em GPU
uv run python export_router.py      # < 1s

# Validar
docker compose up --build -d
k6 run test/peak.js
```

Total: ~15 min em GPU + setup do Docker. Em CPU pode levar várias horas (label.py e
nearest_opp.py são os gargalos).

## Localização dos arquivos gerados

Tudo cai em `data/` (que está no `.gitignore`):

```
data/
├── references.npy                 168 MB
├── labels.npy                       3 MB
├── fraud_counts_k5.npy              3 MB
├── fraud_counts_k25.npy             3 MB
├── nearest_opp_dist.npy            12 MB
├── box_labels.npy                   3 MB
├── box_labels.before_halo.npy       3 MB
├── box_b_refs.bin                  13.6 MB     ← lido pelo backend
├── box_b_labels.bin               213 KB       ← lido pelo backend
├── box_b_ivf_centroids.bin         16 KB       ← lido pelo backend
├── box_b_ivf_offsets.bin            1 KB       ← lido pelo backend
├── router.pt                        ~10 KB
├── router_weights.bin              6.5 KB      ← lido pelo backend
├── router_train_indices.npy
├── router_val_indices.npy
└── router_test_indices.npy
```

Os 5 arquivos marcados como "lido pelo backend" são os únicos que o container precisa em runtime.
O `docker-compose.yml` monta `data/` read-only nos containers.
