# 09 — Debug e profiling

Esta seção documenta o ferramental que usamos para diagnosticar falhas (mismatches contra
ground-truth) e medir performance (latência, throughput, perfil de CPU).

## Workflow de mismatch diagnosis

Sempre que o score cai, a pergunta é: **quais entries do test set estamos errando, e por quê?**

Dois scripts standalone respondem isso: `find_mismatches.py` (encontra) e `analyze_mismatches.py`
(caracteriza).

### find_mismatches.py

Faz POST em cada uma das 54.100 entries do `test-data.json` e registra as que discordam do
`expected_approved`.

```bash
# backend tem que estar rodando (Docker ou nativo)
uv run python find_mismatches.py
```

Pontos sutis:

1. **Compact JSON**. Usa `json.dumps(..., separators=(',', ':'))` (sem espaços). O byte parser
   do backend espera o formato exato que o `JSON.stringify` do k6 emite — sem espaços ao redor de
   `:` e `,`. O `json.dumps` default do Python adiciona espaços e quebra a busca por keys.

2. **Keep-alive single connection**. Sustenta ~18k req/s contra o backend local através de uma
   única conexão. Suficiente para varrer 54k entries em ~3 segundos.

3. **POST em série**. Não paraleliza — interessa só a corretude. Se o backend é determinístico
   (e é), a ordem não importa.

Output: `data/mismatches.json` — lista com `index, expected, actual, fraud_score, original_request`
para cada disagreement.

### analyze_mismatches.py

Para cada entry em `data/mismatches.json`:

1. Rebuilda o query vector em Python espelhando `backend/src/vectorize.rs`.
2. Roda dois brute-force k=5 em numpy:
   - **(a)** sobre os 3M refs completos (ground truth que o data generator usou).
   - **(b)** sobre o subset Box-B (o que o slow path do backend efetivamente vê).
3. Reporta o overlap dos top-5 entre os dois.
4. Para cada ref do top-5 verdadeiro (em 3M), marca se está em Box-B, ou se está em Box-A com
   `count_k25` específico, para localizar **por que** ele não foi incluído.

Output: texto formatado pra leitura humana, com algo do tipo:

```
=== entry 5472 ===
expected: fraud
actual:   legit (count=2)

Top-5 in full 3M:                    Top-5 in Box-B subset:
  d=0.23485  ref=12345  label=fraud    d=0.23485  ref=12345  label=fraud
  d=0.23491  ref=67890  label=fraud    d=0.23491  ref=67890  label=fraud
  d=0.23502  ref=703996 label=legit    d=0.23502  ref=703996 label=legit
  d=0.23508  ref=11122  label=legit    d=0.23508  ref=11122  label=legit
  d=0.23509  ref=1707961 label=fraud   d=0.23510  ref=44455  label=legit  ← perdido!
                                       (1707961 está em A-Fraud, count_k25=25)

Box-B miss: 1707961 está em A-Fraud, count_k25=25. Border halo D atual=0.23
acima de nearest_opp_dist[1707961]=0.187 → halo capturaria com D >= 0.19.
```

(Output exato muda por versão — formato exato em `analyze_mismatches.py:80-180`.)

### Os três motivos clássicos de mismatch

Após várias rodadas de debug, aprendemos a categorizar mismatches em três buckets:

1. **Box-B subset miss** — o top-5 verdadeiro inclui refs que estão em A. Solução: ampliar Box-B
   via halo (ver [03 — Partição e halo](./03-particao-e-halo.md)).

2. **Round4 tie-break** — o top-5 do backend bate com o brute Python *exceto* no rank-5/rank-6
   que estão dentro de 1e-5 de distância. Solução: `round4` no vectorize (ver
   [07 — Guardrails numéricos](./07-guardrails-numericos.md)).

3. **IVF approximation** — o brute Box-B acerta mas o IVF perde por aproximação. Mitigação:
   subir `NPROBE` ou rebuild com `NLIST` menor. **Não observado em produção** com NLIST=256,
   NPROBE=16.

Para qualquer mismatch novo, rodar `analyze_mismatches.py` é o primeiro passo. A diferença dos
três casos é diagnosticável em < 1 segundo de análise.

## Microbench: ivf_bench

`ivf_bench/` é um Rust binary que exercita só o kernel `ivf_k5` — sem HTTP, sem tokio, sem byte
parser. Permite medir latência **pura do kernel** sem o ~50-100 µs de overhead end-to-end.

```bash
cd ivf_bench && cargo build --release
./target/release/ivf-bench
```

Env vars:

```
REFS         caminho do refs.bin       (default data/box_b_refs.bin)
LABELS       caminho do labels.bin     (default data/box_b_labels.bin)
CENTROIDS    caminho dos centroides    (default data/box_b_ivf_centroids.bin)
OFFSETS      caminho dos offsets       (default data/box_b_ivf_offsets.bin)
QUERIES      query set                 (default data/stress_queries.bin, 100k queries)
GROUND_TRUTH counts pré-computadas     (default data/rust_results.bin, opcional)
NPROBE       16
N_QUERIES    20000   (cap do sample; controla resolução de p99)
WARMUP       1000    (queries timed mas descartadas; prime de caches)
NLIST_LABEL  ""      (label cosmético; echoed na primeira coluna)
```

Procedimento:

1. mmap + page-touch de todos os inputs.
2. Warm-up pass (untimed) de 1000 queries.
3. Replay timed de N queries, `Instant::now()` por query.
4. Emite uma linha tab-separated:

```
nlist_label  nprobe  samples  p50_us  p90_us  p99_us  rps  recall_pct
```

Baseline no dev box (Ryzen 9 9900X, NLIST=256, NPROBE=16):

```
256   16   20000   4.7   5.1   5.9   207000   100.0
```

Recall é calculado contra `data/rust_results.bin` (output do `bench/` brute-force).

## Sweep: NLIST × NPROBE

`sweep_ivf.py` + `run_sweep.sh` automatizam a varredura.

### sweep_ivf.py

Para cada `NLIST` em `{64, 128, 256, 512, 1024, 2048}` (configurável), roda Lloyd's k-means até
convergência ou MAX_ITER. Escreve um diretório por NLIST sob `data/ivf_sweep/nlist{N}/`:

```
data/ivf_sweep/nlist256/
├── refs.bin
├── labels.bin
├── centroids.bin
├── offsets.bin
└── meta.json       ← cluster stats + convergência
```

Run uma vez:

```bash
NLISTS="64,128,256,512,1024,2048" .venv/bin/python sweep_ivf.py
```

### run_sweep.sh

Build da bench binary + loop sequencial sobre o produto cartesiano `NLIST × NPROBE`. Pinned com
`taskset -c 0` para reduzir variância:

```bash
./run_sweep.sh
```

Env vars:

```
NLISTS="64 128 256 512 1024 2048"      space-separated
NPROBES="1 2 4 8 16 32 64"             space-separated
N_QUERIES=20000                        samples por combo
WARMUP=1000                            untimed priming
PIN_CPU=0                              core para taskset
REBUILD=1                              força re-run do sweep_ivf.py
```

Output: tabela fixed-width no stdout:

```
nlist   nprobe   samples   p50_us   p90_us   p99_us   rps        recall_%
------  -------  --------  -------  -------  -------  --------   --------
512     16       20000     4.73     5.12     5.89     208123     100.00
512     8        20000     2.91     3.21     3.66     312050      99.86
256     16       20000     5.10     5.42     6.02     192100     100.00
...
```

Foi assim que `NLIST=256, NPROBE=16` virou o default.

## Microbench: bench/

`bench/` é o brute-force k=5 puro (sem IVF), mantido como baseline de comparação. Lê o mesmo
`data/box_b_refs.bin` (mas ignora o particionamento por cluster) e produz `data/rust_results.bin`
— usado como ground truth pelo `ivf_bench`.

```bash
cd bench && cargo run --release
```

Tempo no dev box (Ryzen): ~76 µs por query (105k refs, antes do halo) → ~150 µs estimado no Mac
Mini Haswell (extrapolando pela razão de bandwidth DRAM).

## Profile: profile.sh

Sustained-load profiling do backend nativo (não em Docker — interessa o PID e zero overhead de
container).

```bash
./profile.sh
RATE=30000 DURATION=15s ./profile.sh   # tune
```

Env vars principais:

```
K6_BIN=/tmp/k6                  caminho do binary k6
RATE=10000                      RPS sustentado
DURATION=30s
TOKIO_WORKERS=6                 (override do default current_thread para profile multi-core)
SAMPLE_FREQ=999                 hz do perf record
SAMPLE_DURATION=20              segundos de amostragem
WARMUP=5                        seg de carga antes de começar a amostrar
```

Output em `profile-out/`:

```
backend.log       stdout/stderr do backend Rust
k6.log            resumo do k6
perf.data         amostras raw do perf
perf-report.txt   top-symbols text report
flame.svg         flamegraph interativo (abre em qualquer browser)
```

Dependências: `perf` (linux-tools), `inferno-flamegraph` e `inferno-collapse-perf` (instale com
`cargo install inferno`).

### Build com debug symbols

O `profile.sh` builda com `cargo build --profile profiling`, que herda `release` mas mantém
`debug = "line-tables-only"` e `strip = false`. Isso garante que `perf` resolve símbolos do
hot path (incluindo símbolos LTO-inlined) sem perder a otimização de release.

### Interpretando o output

O flamegraph SVG é o golden source. Hotspots típicos no estado atual (commit `04ca1a3`):

```
8.5%   handle::closure (hyper service_fn, dispatcher inlined)
7.5%   tokio runtime (epoll, executor)
7%     kernel/network (futex, recv, send)
4%     hyper HTTP1 (parse, write_head)
3%     memchr/vectorize (byte parser do payload)
~0%    slow_path::ivf_k5 (invisível: só ~3.5% das queries entram)
```

A 50k RPS o perfil muda — kernel e tokio runtime crescem. Detalhes de "o que ainda dá pra
melhorar" em `LATENCY_BACKLOG.md`.

## Smoke test: peak.js

Quick 6s test (ramp 1s + 5s @ 900 RPS) com percentis completos.

```bash
k6 run test/peak.js
```

Output:

```
✓ tp_count.....: 24037
✓ tn_count.....: 30023
✓ fp_count.....: 0
✓ fn_count.....: 0
✓ error_count..: 0

http_req_duration.....: avg=170µs  min=98µs  med=147µs
                        p(90)=219µs p(95)=247µs p(99)=327µs
                        p(99.9)=572µs max=2.8ms
```

Diferente do `test/test.js` oficial:

- Ramp de 1s (vs 2 min) — só smoke.
- Reporta p50/p90/p99/p99.9/max (oficial só p99).
- Roda em loop com modulo (vs 54k entries).
- NÃO escreve `results.json` (não substitui o teste oficial para scoring).

Reutiliza o `test-data.json` canônico do repo irmão por caminho relativo —
`../rinha-de-backend-2026/test/test-data.json`.

## Sustained load: profile.js

Constant-arrival-rate sem ramp, configurável.

```bash
RATE=30000 DURATION=15s k6 run test/profile.js
PREVUS=600 MAXVUS=3000 RATE=50000 k6 run test/profile.js
```

Usado pelo `profile.sh` por baixo. Default 10k RPS / 30s.

## Test oficial: rinha-de-backend-2026/test/test.js

A *fonte de verdade* para scoring é o test oficial — ramping-arrival-rate 0 → 900 RPS em 120s,
54.100 entries, escreve `results.json` que vai pro ranking.

```bash
k6 run /projects/rinha-de-backend-2026/test/test.js
```

Output (relevante para o score):

```
data_received................: ...
data_sent....................: ...
fp_count.....................: 0
fn_count.....................: 0
http_req_duration............: p(99)=148µs
...
```

E `results.json` no current dir:

```json
{
  "tp_count": 24037, "tn_count": 30023,
  "fp_count": 0, "fn_count": 0,
  "error_count": 0,
  "p99": 0.148,
  "score": 6000.0
}
```

## Checklist antes de declarar "score regrediu"

1. Backend está usando a versão recém-buildada? (`docker compose up --build`)
2. `data/router_weights.bin` e `data/box_b_*.bin` estão sincronizados? (re-export se mexeu no
   pipeline)
3. `find_mismatches.py` reporta as mesmas N entries como mismatch?
4. `analyze_mismatches.py` confirma a categoria do mismatch (Box-B miss / round4 / IVF approx)?
5. Microbench `ivf_bench` ainda reporta recall ≥ 99.95%?
