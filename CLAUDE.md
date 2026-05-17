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
MLP "router" pequeno (14 → 64 → 64 → 3, ~5,3k params) mais um slow path IVF (NN aproximado) sobre
um subconjunto curado de ~212k refs chamado **Box-B**. Score de produção atual: **6000/6000**,
p99 ≈ 0.16 ms.

## Arquitetura em alto nível

O repo é composto de **duas metades fortemente acopladas**:

1. **Pipeline offline (Python, GPU via PyTorch ROCm)** — produz três artefatos que o backend lê na
   subida: `data/router_weights.bin` (MLP de 20.8 KB / 5315 f32), `data/box_b_*.bin` (refs
   ordenados por cluster + labels + centroides IVF + offsets CSR). Todo o trabalho pesado acontece
   aqui, uma única vez.

2. **Backend HTTP online (Rust, `backend/`)** — hyper 1.x direto (sem axum, sem crate de JSON),
   tokio em `current_thread`, parser byte-a-byte manual, kernel de distância AVX2+FMA, arquivos de
   índice via mmap, socket UNIX como upstream para o nginx em modo `stream`. Tudo no hot path é
   zero-alloc.

### Fluxo de inferência por request

```
POST /fraud-score
  → vectorize::vectorize       (byte scan → [f32; 16], com round4 igual ao data-generator)
  → router::infer              (MLP de 5315 params → P(A-Legit), P(A-Fraud), P(B))
  → se P(B) > 0.5: slow_path::ivf_k5  (IVF sobre Box-B → fraud count 0..5)
    senão:         argmax(P(A-Legit) vs P(A-Fraud)) → count 0 ou 5
  → responde com um dos 6 corpos JSON pré-renderizados como &'static [u8]
```

### Ordem do pipeline (offline, sequencial na primeira execução)

```
prepare.py        descompacta references.json.gz → data/references.npy + labels.npy
label.py          leave-one-out k-NN na GPU (K=25, usado pela partição). Saída:
                  data/fraud_counts_k25.npy. (O run com K=5 é só sanidade — não
                  faz parte do pipeline de produção orquestrado pelo Makefile.)
partition.py      labels + fraud_counts_k25 → data/box_labels.before_halo.npy
                  (3 classes: 0=A-Legit, 1=A-Fraud, 2=B). Pristine; nunca é
                  sobrescrito pelo halo.
nearest_opp.py    GPU: distância até o ref mais próximo de label oposto →
                  data/nearest_opp_dist.npy
border_halo.py    promove refs A próximos da fronteira de classe para Box-B
                  (D=0.23 default). Lê box_labels.before_halo.npy + nearest_opp,
                  escreve data/box_labels.npy limpo (sem mutação in-place;
                  rodar de novo com outro D nunca dobra-promove).
export_box_b.py   k-means (NLIST=512 em produção) sobre Box-B → refs/labels/
                  centroides padded + offsets CSR, ordenados por cluster. Tanto
                  os arquivos f32 quanto os mirrors .i16.bin (scale=10000).
train_router.py   MLP de 3 classes com cross-entropy balanceada por classe,
                  GELU(approximate="tanh"), early stopping (PATIENCE=30,
                  MIN_DELTA=1e-5). Lê box_labels.before_halo.npy (pristine,
                  sem halo) — o router só precisa distinguir clusters
                  homogêneos de tudo mais; o halo é só pro slow path.
                  Save state por menor misroute B→A no 3M (não val_loss).
                  → data/router.pt
export_router.py  serializa os 5315 pesos f32 → data/router_weights.bin
```

`make artifacts` orquestra o pipeline completo de produção (8 passos acima)
com hiperparâmetros congelados (K=25, D=0.23, NLIST=512, SEED=42, HIDDEN=64,
DEPTH=2, ITER=1000) e deps por arquivo — só re-roda o que ficou stale.
`make artifacts-clean` apaga `data/*` com confirmação. `run.py` / `run.sh`
são do pipeline ANTIGO de viabilidade (prepare → label → train → evaluate)
e ficam só por compatibilidade — NÃO usar pra reproduzir a imagem oficial.

## Comandos comuns

### Offline (Python, gerenciado pelo uv)

```bash
# Pipeline completo de produção (reproduz os 5 arquivos da imagem oficial):
make artifacts                          # ~1h: prepare → label → ... → export_router

# Re-rodar incrementalmente: Make sabe só o que ficou stale. Ex.: mexer em
# train_router.py re-roda só train + export_router (segundos).

# Steps individuais (override de env funciona normal):
NLIST=1024 NPROBE=32 uv run python sweep_ivf.py  # estudo separado, fora do make
uv run python find_mismatches.py        # roda contra o backend pra validar 6000/6000
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

### Submissão à rinha (Makefile)

A submissão usa uma imagem monolítica `vitornathan/rinha-de-papagaio:latest` no Docker Hub
(binário + 5 artefatos i16 do índice embutidos, ~31 MB). A branch órfã `submission` contém só
`docker-compose.yml` (com o `nginx.conf` inline via `configs.content`) e `info.json`. O engine
da rinha clona o repo, faz `git checkout submission` e `docker compose up`. O mapeamento
`participants/VitorNathanG.json → repo + id` vive no upstream `zanfranceschi/rinha-de-backend-2026`
(já mergeado; `id=papagaio`).

Targets relevantes:

```bash
make build       # build local da imagem papagaio-api:latest
make up          # sobe o stack (lb + 2 réplicas) com healthcheck no /ready
make test        # roda o k6 oficial (test.js, 2 min ramp até 900 RPS)
make deploy      # build + tag + push pro REGISTRY_IMAGE no Docker Hub
make issue       # `gh issue create rinha/test $(SUBMISSION_ID)` no upstream
```

**`make deploy` e `make issue` são ações de blast-radius alto** — empurram artefato pro registry
público e abrem uma issue visível atrelada ao usuário no repo de terceiros. **Nunca rode esses
dois targets de forma proativa.** Espere o usuário pedir explicitamente ("deploy", "submete",
"manda pra rinha", "abre a issue"). Mesmo após uma mudança aparentemente pronta, pare em `make
test` ou `make build` e pergunte (ou só reporte que está pronto pra deploy).

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

# Microbench só do kernel do router (sem HTTP, sem IVF). Reusa stress_queries.bin
# se existir, senão gera queries random reproduzíveis com SEED. BATCH amortiza
# overhead do Instant::now (~25 ns) — use BATCH=1000 pra ler latência pura.
cd router_bench && cargo run --release   # default: 100k queries, BATCH=1

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

- **A shape do router está congelada em hidden=64, depth=2** em `export_router.py` (assert) e em
  `backend/src/router.rs` (constantes `H=64`, `D_IN=14`, `D_OUT=3` em compile time). O bench
  isolado `router_bench/src/main.rs` duplica essas mesmas constantes (kernel proposital duplicado,
  igual ao `ivf_bench` vs `slow_path`). Mudar a shape do MLP exige atualizar os três lados.

- **Refs e centroides são padded para 16 valores por linha** (32 bytes em int16 = meia cache line;
  duas refs por linha). O kernel AVX2 carrega cada ref com um `_mm256_loadu_si256` e calcula a
  distância via `_mm256_sub_epi16` + `_mm256_madd_epi16` + `_mm256_cvtepi32_ps`. Os exportadores
  Python escrevem o padding zerado; o kernel Rust depende dele.

- **Os refs/centroides em produção são int16 com scale=10000** (`box_b_refs.i16.bin`,
  `box_b_ivf_centroids.i16.bin`). Como `vectorize` aplica round4, todo input está no grid
  `k/10000`, e a quantização é bit-exato nas refs. Centroides (médias do k-means) ganham ±0.5
  unidade de ruído por dim, mas isso é sub-raio-do-cluster e não muda ranking nprobe. Os arquivos
  f32 (`box_b_refs.bin`, `box_b_ivf_centroids.bin`) continuam sendo escritos como mirror — o
  `bench/` brute-force e ferramentas antigas ainda os lêem.

- **A escolha de NLIST=512 é dominada pela constraint de recall top-5, não pelo custo balance.**
  A math do IVF prevê `nlist* ≈ √(nprobe·N) ≈ 1845` (para nprobe=16, N=212k). Empiricamente,
  NLIST≥1024 introduz 1-3 fn no test set oficial porque queries borderline têm os 5 vizinhos
  espalhados por múltiplos clusters menores que nprobe não cobre. NLIST=512 + nprobe=16 mantém
  6000/6000 e é o ponto de menor latência sem mismatches. Documentado no sweep do commit do i16
  kernel.

- **O k-means do `export_box_b.py` roda até convergência total** (zero point reassignments entre
  iters). `ITER` é apenas safety cap (default 1000); convergência típica em ~100 iters. Ler isso
  como "rode o número certo" — não diminua o cap pra acelerar.

- **Os refs e labels em `box_b_*.bin` estão ordenados pela atribuição de cluster IVF**, com
  `box_b_ivf_offsets.bin` como offsets CSR de linha. O slow path lê cada cluster sondado como uma
  slice contígua — sem indireção por cluster.

## Coisas que parecem refatoráveis mas não são

- **Duas cópias de `slow_path.rs`**: uma em `backend/src/` (produção) e outra inline em
  `ivf_bench/src/main.rs` (bench). É proposital — o crate do bench fica independente do crate do
  backend. Se você mexer no kernel, mexa nos dois — o `ivf_bench` não importa de `backend`.

- **Duas cópias de `router.rs`**: mesma justificativa. `backend/src/router.rs` (produção) e
  `router_bench/src/main.rs` (bench) têm kernels idênticos (Padé tanh, GELU, infer, softmax) e
  constantes `H`/`D_IN`/`D_OUT` duplicadas em compile time. Mudou um, mude o outro.

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
