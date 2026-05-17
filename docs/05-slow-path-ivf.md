# 05 — Slow path IVF

Queries roteadas para a classe B precisam de um k=5 sobre o subset Box-B. Esta seção documenta o
slow path — por que IVF e não brute force, como o índice é montado, e como o kernel AVX2 explora
o layout para ficar L2-warm.

## A motivação: cache do Mac Mini

A máquina de teste oficial da rinha é um **Mac Mini Late 2014** (Haswell, 2.6 GHz, 8 GB DDR3-1600,
**4 MB L3**). Esse L3 minúsculo é o que dita a escolha de IVF.

Working set do slow path com Box-B + halo:

```
212.733 refs × 16 floats × 4 B = 13.6 MB
```

Brute force varre todos os 13.6 MB por query. Mas:

- L3 do Mac Mini = 4 MB → working set não cabe.
- DRAM efetiva ≈ 15 GB/s → varrer 13.6 MB leva ~900 µs.
- A 32 RPS de slow path (3.5% × 900 RPS) isso é trivial em throughput, mas **catastrófico em
  p99** (cada miss para DRAM é um pico).

Num dev box com 32 MB de L3 (Ryzen 9 9900X), brute force fica L3-resident e roda em ~76 µs — por
isso o problema é invisível em desenvolvimento e só aparece no target.

IVF cabe o per-query working set em L2.

## Como o IVF é estruturado

**Inverted File index**: o dataset é particionado em `nlist` clusters via k-means. Cada cluster
é definido por um centroide. Para um query:

1. Compute distância do query a *cada centroide* (varredura linear pequena).
2. Pegue os `nprobe` clusters mais próximos.
3. Compute distância exata só aos refs *desses* clusters.
4. Sort top-5 globalmente.

A aproximação acontece em (2-3): se o vizinho verdadeiro estiver num cluster não-sondado, ele é
perdido. Com `nlist=256` e `nprobe=16`, a recall medida contra brute-force ground-truth é
**100%** nos `stress_queries.bin` que usamos no microbench. Em queries borderline da rinha o
recall é um pouco menor — mas ainda dentro do que o halo + round4 já cobrem.

### Parâmetros default

```
NLIST=256          (≈ √N para N≈105k; após halo N≈212k mas ainda funciona)
NPROBE=16
ITER=20            (Lloyd's k-means até convergência)
SEED=42
```

`NLIST=256` foi escolhido olhando o sweep em `sweep_ivf.py` + `run_sweep.sh`. Findings do sweep:

```
nlist=512, nprobe=16:   p50 4.73 µs, p99 5.89 µs, recall 100%
nlist=256, nprobe=16:   p50 ~5 µs,   p99 ~6 µs,   recall 100%  ← escolhido
nlist=512, nprobe=8:    p50 ~3 µs,   p99 3.66 µs, recall 99.86%
nlist=2048:             centroid scan vira gargalo (2 µs só varrendo centroides)
```

`nlist=256` ganha em uma vantagem que o microbench não mede: footprint dos centroides
(`256 × 64 B = 16 KB`) cabe perfeitamente em **L1** (32 KB típico), enquanto `nlist=512` (32 KB)
fica no limite.

## Working set por query

Com `nlist=256`, `nprobe=16`, e ~415 refs/cluster (após halo):

```
Centroides:           256 × 64 B = 16 KB     ← cabe em L1
Refs sondados:    16 × 415 × 64 B = 425 KB   ← cabe em L2 (256 KB típico do Haswell? na verdade 2-8 MB)
```

L2 do Haswell tem 256 KB por core mais 4 MB de L3. 425 KB sai do L2 estrito mas cabe no L3, e o
**streaming pattern** (varredura linear de refs contíguos) é o caso ideal para prefetcher
hardware. Em prática, com software prefetch 8 refs à frente, a CPU mantém ~50% das misses
absorvidas pelo prefetcher.

Compare com brute (13.6 MB): IVF tem **32× menos** working set por query.

## Layout binário (CSR)

`export_box_b.py` escreve 4 arquivos:

```
box_b_refs.bin            f32 × N × 16    (refs ordenados por cluster, padded)
box_b_labels.bin          u8  × N         (labels na mesma ordem)
box_b_ivf_centroids.bin   f32 × nlist × 16
box_b_ivf_offsets.bin     u32 × (nlist+1) (offsets CSR em *linhas*, não bytes)
```

Dois detalhes importantes:

### 1. Ordenação por cluster

Após o k-means, refs e labels são re-ordenados via `np.argsort(assignments, kind="stable")`.
Resultado: todos os refs do cluster 0 vêm primeiro (contíguos), depois cluster 1, etc.

Em runtime, escanear um cluster é **uma slice contígua** (`refs[offsets[c] .. offsets[c+1]]`).
Sem cluster→ref indirection. Sem scatter de memória. O hardware prefetcher adora.

### 2. Padding para 16 floats por linha

Cada ref ocupa exatamente 64 bytes (1 cache line):

```
[v0, v1, ..., v13, 0.0, 0.0]   ← 14 dims + 2 floats de padding
 \__________________________/
            64 B
```

O kernel AVX2 carrega um ref em **dois ymm registers** com dois `_mm256_loadu_ps`, sem precisar
de máscaras ou loads parciais. As 2 dims de padding zeradas geram `(0 - 0)² = 0` na soma de
quadrados — não afetam o resultado.

Os centroides têm o mesmo padding pela mesma razão.

## O kernel AVX2

`backend/src/slow_path.rs::ivf_k5`. Estrutura:

```rust
unsafe fn ivf_k5(query, centroids, offsets, refs, labels, nprobe) -> u8 {
    let q0 = _mm256_loadu_ps(query.as_ptr());
    let q1 = _mm256_loadu_ps(query.as_ptr().add(8));

    // FASE 1: top-nprobe centroides via fixed-size insertion sort
    let mut top_c_d   = [INF; MAX_NPROBE];
    let mut top_c_idx = [0u32; MAX_NPROBE];
    for c in 0..nlist {
        let d = distance(q0, q1, cptr.add(c * 16));
        insert_top_centroid(&mut top_c_d, &mut top_c_idx, d, c);
    }

    // FASE 2: scan dos clusters sondados, top-5 global
    let mut best_d   = [INF; 5];
    let mut best_idx = [0u32; 5];
    for p in 0..nprobe {
        let cluster = top_c_idx[p];
        let start = offsets[cluster];
        let end   = offsets[cluster + 1];
        // varredura linear contígua com prefetch 8 ahead
        for i in start..end-PREFETCH_AHEAD {
            _mm_prefetch::<{ _MM_HINT_T0 }>(rptr.add((i+PREFETCH_AHEAD) * 16));
            let d = distance(q0, q1, rptr.add(i * 16));
            insert_top_k_refs(&mut best_d, &mut best_idx, d, i);
        }
        // tail sem prefetch
        for i in end-PREFETCH_AHEAD..end {
            ...
        }
    }

    // count = soma dos labels do top-5
    let mut count: u16 = 0;
    for i in 0..5 { count += labels[best_idx[i]] as u16; }
    count as u8
}
```

### O hot inner loop

```rust
unsafe fn distance(q0: __m256, q1: __m256, r_ptr: *const f32) -> f32 {
    let r0 = _mm256_loadu_ps(r_ptr);
    let r1 = _mm256_loadu_ps(r_ptr.add(8));
    let diff0 = _mm256_sub_ps(q0, r0);
    let diff1 = _mm256_sub_ps(q1, r1);
    let sq0   = _mm256_mul_ps(diff0, diff0);
    let sum   = _mm256_fmadd_ps(diff1, diff1, sq0);   // FMA: (diff1 * diff1) + sq0
    // horizontal sum de 8 floats para escalar
    ...
}
```

Custo por ref: 2 loads, 2 subs, 1 mul, 1 fma, ~4 instruções de horizontal sum. ~10-12 ciclos por
ref no Haswell, latência dominada pelo horizontal sum.

### Insertion sort fixo

`insert_top_k_refs` e `insert_top_centroid` são insertion sorts em arrays de tamanho fixo (5 e
até 64). Cada chamada custa O(K) no pior caso mas o caso comum é O(1) (a maioria dos refs não
está no top). Não há heap, não há malloc, tudo na stack.

### Prefetch 8 ahead

`_mm_prefetch::<_MM_HINT_T0>` traz a próxima cache line para L1 com 8 refs de antecedência. O
hardware prefetcher pega o padrão linear sozinho, mas o explicit prefetch ajuda em cluster
boundaries onde o stride muda.

## mmap + page touch

`backend/src/main.rs::mmap_static` carrega cada arquivo binário via `memmap2::Mmap`, depois faz
`Box::leak` para obter uma slice `'static`. O leak é one-time (vive a vida do processo) e mantém
o arquivo no page cache do kernel em vez de duplicar em heap.

`touch_pages` lê 1 byte por página (4 KB stride) na subida para forçar todos os page faults
*antes* da primeira request. Sem isso, as primeiras 200-300 queries pagariam demand-paging custo
(latência ~50 µs cada em vez de ~5 µs).

`madvise(WillNeed)` é setado também, mas é só hint — o touch síncrono é o que garante.

## Aproximação cost

Validado contra os 54.060 queries do `test-data.json` da rinha:

```
fp:  0  (zero)
fn:  0  (zero)
```

Após o halo D=0.23 + round4. O IVF não causa nenhum verdict flip — todos os cenários onde ele
*poderia* errar são compensados pelo halo (que pegou os refs externos que iam faltar) e pelo
fato de que `recall` em queries borderline ainda é >99.95% mesmo se cair de 100% nominal.

## Estrutura dos binários (formato esperado pelo backend)

```
box_b_refs.bin
└── 212.733 × 64 B = 13.6 MB
    │ row 0: [f32 × 14 || 0.0 × 2]   ← cluster 0, ref 0
    │ row 1: [f32 × 14 || 0.0 × 2]   ← cluster 0, ref 1
    │ ...
    │ row N: [f32 × 14 || 0.0 × 2]   ← cluster nlist-1, último ref

box_b_labels.bin
└── 212.733 × 1 B = 213 KB
    │ label[i] ∈ {0, 1}             ← na mesma ordem dos refs

box_b_ivf_centroids.bin
└── 256 × 64 B = 16 KB
    │ row 0: [f32 × 14 || 0.0 × 2]   ← centroide do cluster 0

box_b_ivf_offsets.bin
└── (256 + 1) × 4 B = 1.028 B
    │ offsets[0]   = 0
    │ offsets[1]   = nrefs_cluster_0
    │ offsets[2]   = nrefs_cluster_0 + nrefs_cluster_1
    │ ...
    │ offsets[256] = N (total de refs)
```

Validações em runtime (em `main.rs::load_state`):

```rust
assert_eq!(refs.len(), labels.len() * 16);           // refs e labels casam em count
assert_eq!(centroids.len(), nlist * 16);             // centroids e offsets casam em nlist
assert!((1..=64).contains(&nprobe));                 // NPROBE no range válido
```

## Microbench isolado

`ivf_bench/` é um Rust binary standalone que exercita só o `ivf_k5` — sem HTTP, sem tokio, sem
parser. Lê os mesmos binários mmap'd via env vars e replaya queries de `data/stress_queries.bin`
(100k queries pré-geradas).

Baseline no dev box (Ryzen 9 9900X, NLIST=256, NPROBE=16):

```
p50 = 4.7 µs   p99 = 5.9 µs   207k QPS   recall = 100%
```

`run_sweep.sh` roda esse bench para o produto cartesiano de `NLIST × NPROBE` automatizado, com
`taskset -c 0` pinned para reduzir variância. Ver
[09 — Debug e profiling](./09-debug-e-profiling.md) para uso.

## Pegadinha: o kernel está duplicado

`backend/src/slow_path.rs` (produção) e `ivf_bench/src/main.rs` (bench) têm cópias **idênticas e
independentes** do kernel `ivf_k5`. Deliberado — o crate do bench não importa do crate do
backend. Se você mexer no kernel, mexa nos dois lugares.
