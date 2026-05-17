# 10 — Experimentos descartados

Documentar o que **não** funcionou é tão útil quanto documentar o que funcionou — evita repetir
arquitetural dead-ends e dá contexto para decisões aparentemente arbitrárias.

## Subset selection: três estratégias para Box-B + halo

A pergunta era: além dos refs de fronteira pura (count_k25 in 1..24), quais refs adicionais
precisam estar no slow path subset para que k=5 sobre o subset bata com k=5 sobre os 3M?

Testamos três estratégias antes de chegar no nearest-opp halo que vingou.

### A — Geometric halo (k-NN dos refs de B)

`halo.py`. Para cada ref em B, pegue os `m` vizinhos mais próximos (em qualquer classe), dedup, e
adicione ao Box-B.

Resultados:

| Halo size | A refs adicionados | Slow-path agreement |
|-----------|--------------------|--------------------|
| m=5       | 1.305               | 94.4% (idêntico ao baseline sem halo) |
| m=10      | 2.939               | 94.4% (também idêntico) |

**Por que falhou**: Box-B é uma camada fina e densa. Os k-NN dos pontos de B são predominantemente
**outros pontos de B** — não os refs de A na borda externa do cluster que estávamos tentando
pegar. Geometric halo adiciona pontos redundantes (já estavam em B) e perde os que importam.

Lição: a métrica de proximidade tem que ser **opposite-label**, não absoluta.

### B — Frequency-weighted sampling

`sample_subset.py`. Geração intensiva:

1. Sample 2M queries aleatórias do espaço (ou de payloads sintéticos).
2. Para cada query, filtra via router. Se router → B, ignora (já pego). Se router → A, computa
   o top-5 verdadeiro em 3M.
3. Conta quantas vezes cada **ref em A** aparece no top-5 dessas queries roteadas a A.
4. Weight bonus para refs que aparecem em queries de count `{2, 3}` (i.e., queries cujo veredito
   depende do label do ref): peso 1.0 se count ∈ {2,3}, 0.1 caso contrário.
5. Sort A refs por score ponderado, sweep "adicionar top-N do ranking".

Curva resultante:

| Top-N adicionado | Slow-path agreement |
|------------------|--------------------|
| +500             | 95.3%               |
| +5.000           | 98.15%              |
| +10.000          | 98.98%              |
| +50.000          | 98.98%              |
| (assintótico)    | ~98.98%             |

O assintótico de ~99% (vs 100%) **é o noise floor BORDERLINE**: 1.02% das queries vai depender
de refs BORDERLINE no top-5, e nenhuma estratégia pode arrumar isso (labels são aleatórios por
definição).

**Por que (parcialmente) falhou**: chegou no asymptote mas **caro de calibrar**. Precisava de
2M queries sintéticas + um critério de peso a-priori (count ∈ {2,3}). Quando o nearest-opp halo
mostrou que dava pra chegar nos mesmos refs com uma métrica geométrica direta, abandonamos esse
caminho.

### C — Prune from full

`prune_subset.py`. Inversão da B: começa com 3M refs, ordena por usage (via sampling como em B),
mantém top-N.

Resultados:

| Subset N | Slow-path agreement |
|----------|--------------------|
| Box-B (105k)         | 94.4% (in-distribution test) |
| Top 100k (pruned)    | 75.6% (in-distribution) |
| Top 100k (pruned)    | 98.2% (out-of-distribution sampled) |

**Por que falhou catastroficamente**: distribution shift. A amostragem usada para ranquear
"usage" tinha distribuição diferente das queries reais do `test-data.json` da rinha. Refs que
nunca foram tocados em sampling random eram exatamente os refs que queries in-distribution mais
tocavam. Pruning collapsa a 75.6% (vs 94.4% do Box-B baseline) na distribuição que importa.

Lição: **pruning otimizado para uma distribuição quebra catastroficamente em outra**. Útil como
diagnóstico ("17k de Box-B refs nunca são tocados por router-routed queries"), perigoso em
produção.

### D — nearest-opp border halo (o que vingou)

`nearest_opp.py` + `border_halo.py`. Métrica direta: distância ao ref de label oposto mais
próximo. Promove A → B para todos os refs com `nearest_opp < D`.

Detalhes em [03 — Partição e halo](./03-particao-e-halo.md). Resumindo:

- D = 0.23: 105k → 213k refs em Box-B.
- Recupera **todos os 11 refs faltantes** nos 5 mismatches conhecidos.
- 6000/6000 contra `test-data.json`.

Por que funcionou: a métrica é geométrica (não depende de sampling), local (cada ref tem o seu
nearest_opp independentemente), e captura exatamente o conceito de "está na fronteira de classe"
sem proxies.

## Backend e infra

### A — axum

Baseline original. Profile mostrou ~25-29% de samples em handler dispatching do axum (closure
boxing, extractor machinery, layered middleware). hyper direto via `service_fn` colapsa tudo num
único symbol via LTO.

Trade-off: perde ergonomia (router declarativo, extractors, layered middleware). Para uma API
com **uma única rota**, axum é puro overhead.

### B — serde_json

Profile mostrou ~5% do CPU em allocator (`_int_malloc`, `_int_free`, `cfree`, `memmove`).
Substituído por byte-scan manual em `vectorize.rs` — o data generator emite campos em ordem
determinística, então `memmem` para localizar keys + `memchr` para delimitadores é mais rápido
que qualquer parser genérico.

Trade-off: parser é frágil — qualquer reordenação de campos no payload quebra. Mitigado pelo fato
de que o data generator é nosso ground truth (não suportamos clientes arbitrários).

### C — libm::tanhf

Padé[7/6] inline em vez de `libm::tanhf`. Ver [07 — Guardrails numéricos](./07-guardrails-numericos.md).
`libm::tanhf` em pure-Rust fan-outs para `expm1f` internamente, somando ~5% do CPU. O Padé é
~metade dos ciclos.

Trade-off: precisão Padé é 2e-7 (vs ~ulp do libm). Bem abaixo do que afeta vereditos.

### D — multi-thread tokio dentro do container 0.425 CPU

Profile mostrou que o scheduler de multi-thread tokio (work-stealing, epoch counters) era custo
puro num container que mal saturava 1 worker. `current_thread` é literalmente mais rápido nesse
regime.

Trade-off: zero — não há cenário onde multi-thread ganha dentro do budget da rinha.

### E — Custom Rust LB

Avaliado contra `nginx stream`. nginx stream é `splice()`-based TCP forwarding puro; uma versão
custom em Rust salvaria 1-3 µs no máximo. Custo: ~200 linhas + Dockerfile separado + testes.
Payback ≤ 5 µs por request. **Não vale.**

### F — Hand-rolled HTTP/1.1 parser

Em `LATENCY_BACKLOG.md` como Tier 1 item 6. Esperado -4 a -5% CPU, ~5-10 µs no p99. Custo:
~1-2 dias, perde edge cases do hyper (chunked transfer, weird Connection headers). Não foi feito
porque já estamos em 6000/6000.

### G — io_uring (tokio-uring / glommio)

Em `LATENCY_BACKLOG.md` como Tier 1 item 7. Esperado -5% CPU a 50k RPS, ~zero ganho a 900 RPS.
Custo: rewrite do runtime. Trade-off ruim para a carga da rinha.

## Slow path

### A — Brute force em produção

Funcionava no dev box (105k refs / 6.75 MB cabe em L3 de 32 MB do Ryzen). No Mac Mini target
(L3 = 4 MB), brute streaming de DRAM é ~450 µs por query. Substituído por IVF.

### B — HNSW

Considerado mas não implementado. HNSW tem footprint maior em memória (cada ref armazena lista
de neighbors em múltiplos níveis) — ~50-200 MB para 200k refs. Mais que o budget total de
350 MB. IVF com NLIST=256 + offsets CSR tem footprint ~13.6 MB total — duas ordens de magnitude
menor.

Trade-off: HNSW geralmente tem recall melhor que IVF no mesmo budget de compute, mas o budget
aqui é memória, não compute. IVF vence por footprint.

### C — VP-tree / KD-tree

Não testado. Em 14 dimensões, a maldição da dimensionalidade tipicamente degrada VP/KD-trees
para perto de linear scan — perdem para IVF no mesmo regime.

### D — Two-stage: confidence-based routing

A versão original do design (no README inicial) era:

1. Router router binário → P(B).
2. Box-A model → P(fraud | A).

Substituído pelo classificador único de 3 classes. Razões em
[04 — Router MLP](./04-router-mlp.md). Resumindo: um único forward, calibração de incerteza no
contorno por dados, pesos compartilhados.

## Treino do router

### A — Sem class weighting

Box-B é ~3.5% do dataset. Cross-entropy sem weighting colapsa o modelo para "predict A
trivialmente, ignore B". Recall_B fica próximo de 0. Inverse-frequency weighting é a fix mínima
viável.

### B — pos_weight binário (do README original)

A formulação binária inicial usava `BCEWithLogitsLoss + pos_weight=3.0` (reflete FN sendo 3× pior
que FP no score_det). Funcionou para uma classificação fraud/legit, mas não acomoda a partição
em 3 classes. CrossEntropy com weights cobre o caso 3-class corretamente.

### C — GBDT / Random forest

Considerado em vez de MLP. Pros: features bounded e numéricas, decision surface piecewise-flat
"naturalmente". Cons: tempo de inferência por árvore é cache miss bait, e o número de árvores
necessárias para ~99.93% recall em B passa de 100. MLP de 1635 params + ~5 µs de matmul vence em
latência por uma ordem de magnitude.

### D — Modelo único sobre tudo

Discutido no [02 — Estrutura dos dados](./02-estrutura-dos-dados.md). Loss média é dominada
pelos 96.5% triviais; o modelo não aprende o contorno. Partition + router específico é
estritamente melhor.

## O que ficou aberto (não testado)

Coisas que poderiam funcionar mas não foram exploradas:

- **Synthetic queries do C generator como training data para router/Box-B**. Atualmente
  treinamos com leave-one-out das references; queries reais vêm de uma distribuição
  potencialmente diferente. Risco: leaking do test set.
- **Distillation hierárquica**: um router rapido seguido de um classificador de boundary
  mais largo. Provavelmente overkill — 6000/6000 já está saturado.
- **Quantização do router** (int8 weights). 6.5 KB já é desprezível; quantizar economiza ~3 KB
  sem ganho relevante.
- **Halo via k=100 / k=500 purity**. Testado durante calibração e descartado: refs no contorno
  externo de clusters *grandes* não são pegos por k mais largo (todos os k vizinhos ainda são do
  mesmo cluster até k muito alto). nearest_opp é estritamente melhor.
