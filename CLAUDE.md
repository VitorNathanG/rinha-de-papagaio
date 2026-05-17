# CLAUDE.md

Este arquivo orienta o Claude Code (claude.ai/code) quando ele estiver trabalhando neste repositório.

## O que é este repo

"Papagaio" é uma submissão para a **Rinha de Backend 2026** (detecção de fraude por 5-NN sobre 3M
vetores de referência, budget rígido de 1 CPU / 350 MB para LB + ≥2 réplicas da API, p99 ≤ 1 ms
para score máximo). A spec oficial do desafio, o dataset, o test data e o harness k6 vivem no repo
irmão `../rinha-de-backend-2026/` (NÃO é um submódulo — é referenciado por caminho relativo a
partir de `prepare.py`, `test/peak.js`, `test/profile.js`, `find_mismatches.py`, etc.). Sempre
consulte `../rinha-de-backend-2026/docs/en/` para as regras canônicas (DETECTION_RULES.md,
DATASET.md, EVALUATION.md).

Em vez de rodar k-NN sobre os 3M refs em tempo de request, esta submissão destila o problema num
MLP "router" minúsculo (14 → 32 → 32 → 3) mais um slow path IVF (NN aproximado) sobre um
subconjunto curado de ~212k refs chamado **Box-B**. Score de produção atual: **6000/6000**,
p99 ≈ 0.15 ms.

## Arquitetura em alto nível

O repo é composto de **duas metades fortemente acopladas**:

1. **Pipeline offline (Python, GPU via PyTorch ROCm)** — produz três artefatos que o backend lê na
   subida: `data/router_weights.bin` (MLP de 6.5 KB), `data/box_b_*.bin` (refs ordenados por
   cluster + labels + centroides IVF + offsets CSR). Todo o trabalho pesado acontece aqui, uma
   única vez.

2. **Backend HTTP online (Rust, `backend/`)** — hyper 1.x direto (sem axum, sem crate de JSON),
   tokio em `current_thread`, parser byte-a-byte manual, kernel de distância AVX2+FMA, arquivos de
   índice via mmap, socket UNIX como upstream para o nginx em modo `stream`. Tudo no hot path é
   zero-alloc.

### Fluxo de inferência por request

```
POST /fraud-score
  → vectorize::vectorize       (byte scan → [f32; 16], com round4 igual ao data-generator)
  → router::infer              (MLP de 1635 params → P(A-Legit), P(A-Fraud), P(B))
  → se P(B) > 0.5: slow_path::ivf_k5  (IVF sobre Box-B → fraud count 0..5)
    senão:         argmax(P(A-Legit) vs P(A-Fraud)) → count 0 ou 5
  → responde com um dos 6 corpos JSON pré-renderizados como &'static [u8]
```

### Ordem do pipeline (offline, sequencial na primeira execução)

```
prepare.py        descompacta references.json.gz → data/references.npy + labels.npy
label.py          leave-one-out k-NN na GPU. RODAR DUAS VEZES: K=5 (sanidade) e
                  K=25 (usado pela partição). Saída: data/fraud_counts_k{K}.npy.
partition.py      labels + fraud_counts_k25 → data/box_labels.npy
                  (3 classes: 0=A-Legit, 1=A-Fraud, 2=B)
nearest_opp.py    GPU: distância até o ref mais próximo de label oposto →
                  data/nearest_opp_dist.npy
border_halo.py    promove refs A próximos da fronteira de classe para Box-B
                  (D=0.23 default). Lê box_labels.before_halo.npy se existir; a
                  primeira execução cria esse snapshot. Idempotente: pode rodar
                  de novo com outro D sem promover duas vezes.
export_box_b.py   k-means (NLIST=256) sobre Box-B → refs/labels/centroides
                  padded + offsets CSR, ordenados por cluster.
train_router.py   MLP de 3 classes com cross-entropy balanceada por classe,
                  GELU(approximate="tanh"), early stopping (PATIENCE=30,
                  MIN_DELTA=1e-5) → data/router.pt
export_router.py  serializa os 1635 pesos f32 → data/router_weights.bin
```

`run.py` / `run.sh` orquestram `prepare → label → train → evaluate` (pipeline antigo do estudo de
viabilidade); o caminho de produção usa a sequência completa acima. **Não há orquestrador
end-to-end que cubra partition → nearest_opp → border_halo → export** — esses passos são rodados
manualmente, na ordem descrita.

## Comandos comuns

### Offline (Python, gerenciado pelo uv)

```bash
# Setup inicial: instala uv, cria .venv, sincroniza torch+ROCm
./run.sh                                # roda prepare→label→train→evaluate completo
SUBSET=100000 BATCH=128 ./run.sh        # smoke test em CPU / GPU pequena

# Steps individuais (depois que as deps já estão sincronizadas):
uv run python prepare.py
K=25 uv run python label.py             # K controla a largura da vizinhança
uv run python partition.py
uv run python nearest_opp.py
D=0.23 uv run python border_halo.py     # re-tuning do threshold do halo
NLIST=256 uv run python export_box_b.py
uv run python train_router.py
uv run python export_router.py
```

A GPU usa ROCm 7.2 por padrão (RDNA4). Troque a URL do index em `pyproject.toml` para CUDA/CPU; os
comentários daquele arquivo listam as URLs alternativas. `triton-rocm` está como dep direta porque
o uv só honra `[tool.uv.sources]` para deps diretas.

### Backend (Rust)

```bash
# Build / run local (depois que o pipeline offline já produziu data/)
cd backend && cargo build --release
./backend/target/release/papagaio-api   # ouve em 0.0.0.0:9999 por padrão

# Stack Docker (LB + 2 réplicas, exatamente no formato da submissão da rinha)
docker compose up --build

# Profile (line-table debug symbols, perf, flamegraph)
./profile.sh                             # escreve profile-out/{perf-report.txt,flame.svg}
```

Env vars principais consumidas pelo backend: `BIND_ADDR` (`host:port` ou `/caminho/para.sock`),
`TOKIO_WORKERS` (default 1 = current_thread), `NPROBE` (default 16), `REFS_PATH`, `LABELS_PATH`,
`CENTROIDS_PATH`, `OFFSETS_PATH`, `WEIGHTS_PATH`.

### Testes / benches

```bash
# Smoke de latência (ramp 1s + 5s @ 900 RPS, percentis completos)
k6 run test/peak.js

# Teste oficial da rinha (54.100 entradas, ramp 2 min até 900 RPS, escreve results.json)
k6 run /projects/rinha-de-backend-2026/test/test.js

# Carga sustentada para profile (default 10000 RPS / 30s; ajuste RATE / DURATION)
RATE=30000 DURATION=15s k6 run test/profile.js

# Microbench só do kernel IVF (sem HTTP, sem router), lê os arquivos de índice via mmap
cd ivf_bench && cargo run --release

# Sweep (NLIST × NPROBE) sobre o grid do IVF
./run_sweep.sh                           # ≈ 6 NLIST × 7 NPROBE combinações

# Microbench brute-force k=5 sobre Box-B (harness mais antigo)
cd bench && cargo run --release

# Diagnóstico por request, contra o test set oficial
uv run python find_mismatches.py        # POST em todas as 54.100 entradas; escreve data/mismatches.json
uv run python analyze_mismatches.py     # explica cada mismatch (miss em Box-B? aproximação IVF? round4?)
```

## Guardrails numéricos críticos

Esses pontos são sutis e já causaram regressões — não mude um lado sem mudar o outro.

- **`vectorize.rs` arredonda o vetor de 14 dims para 4 casas decimais** antes do router (veja
  `backend/src/vectorize.rs:97-99` e o commit `d6c4081`). O data generator da rinha faz o mesmo
  (`data-generator/main.c:735,774`) antes de calcular `expected_approved`, então empates no rank-5
  vs rank-6 do brute k-NN são desempatados do mesmo jeito. Tirar isso traz de volta verdict flips
  em queries de empate apertado (ex.: entrada 5472 do test set).

- **A ativação do router é `nn.GELU(approximate="tanh")`** — fórmula tanh do OpenAI/GPT-2, não a
  default baseada em erf. A inferência no Rust (`router::gelu` + o `tanh_approx` Padé inline em
  `backend/src/router.rs`) tem que casar com a fórmula do treino bit a bit. Se você re-treinar com
  outra ativação, atualize também o kernel Rust.

- **A shape do router está congelada em hidden=32, depth=2** em `export_router.py` (assert) e em
  `backend/src/router.rs` (constantes `H=32`, `D_IN=14`, `D_OUT=3` em compile time). Mudar a shape
  do MLP exige atualizar os dois lados.

- **Refs e centroides são padded para 16 floats por linha** (uma cache line de 64 bytes) para que
  o kernel AVX2 de distância carregue-os com dois `_mm256_loadu_ps`. Os exportadores Python
  escrevem esse padding; o kernel Rust depende dele.

- **Os refs e labels em `box_b_*.bin` estão ordenados pela atribuição de cluster IVF**, com
  `box_b_ivf_offsets.bin` como offsets CSR de linha. O slow path lê cada cluster sondado como uma
  slice contígua — sem indireção por cluster.

## Coisas que parecem refatoráveis mas não são

- **Duas cópias de `slow_path.rs`**: uma em `backend/src/` (produção) e outra inline em
  `ivf_bench/src/main.rs` (bench). É proposital — o crate do bench fica independente do crate do
  backend. Se você mexer no kernel, mexa nos dois — o `ivf_bench` não importa de `backend`.

- **`run.py` / `run.sh` só cobrem o pipeline antigo de viabilidade** (prepare → label → train →
  evaluate). O pipeline de produção bifurca em `label.py` (com K=25) para partition → nearest_opp →
  border_halo → export_box_b → train_router → export_router. Não existe um orquestrador único do
  caminho de produção; os passos são rodados manualmente como listado acima.

## Contexto do score (interpretando mudanças)

O 6000/6000 atual corresponde a 0 fp / 0 fn contra `test-data.json`. Commits anteriores
estacionaram em **5700/6000** por causa da classe BORDERLINE no dataset de origem (~3% dos refs
têm label aleatório; veja o commit `89dde9a`). Fechar os 5 últimos verdict gaps exigiu (a) o fix
do round4 no vectorize e (b) o border halo via nearest-opp em D=0.23. Se uma mudança trouxer
falhas de volta, espere que elas se concentrem em queries de fronteira —
`find_mismatches.py` + `analyze_mismatches.py` identificam se a causa é uma aproximação do IVF, um
miss no subset Box-B, ou drift numérico.

Para priorização de otimizações puramente de latência, veja `LATENCY_BACKLOG.md`. Itens que mexem
em score (k-NN exato sobre os 3M completos) estão fora de escopo desse backlog.
