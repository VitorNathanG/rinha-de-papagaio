# 01 — Visão geral

## O desafio

A [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026) pede uma API de
detecção de fraude que recebe uma transação JSON, vetoriza em 14 dimensões normalizadas, encontra
os **5 vizinhos mais próximos** num dataset fixo de **3.000.000 vetores rotulados**, e responde
`approved = fraud_score < 0.6` onde `fraud_score = fraud_count_5 / 5`.

Restrições da submissão:

- **1 CPU e 350 MB de RAM** somando *todos* os containers (LB + APIs).
- **Mínimo 2 réplicas de API**, distribuídas em round-robin por um load balancer dedicado.
- Network mode `bridge` (sem `host`, sem `privileged`).
- Imagens `linux-amd64` em registry público.

Pontuação (`-6000 .. +6000`):

- `score_p99`: cada 10× de melhora vale +1000 pontos, satura em +3000 quando p99 ≤ 1 ms.
- `score_det`: combina taxa de erro ponderada (FN > FP, HTTP errors > tudo) com penalidade
  absoluta.

A spec canônica vive em `../rinha-de-backend-2026/docs/en/` — em particular `DETECTION_RULES.md`,
`DATASET.md`, `EVALUATION.md` e `VECTOR_SEARCH.md`.

## Duas estratégias plausíveis

### ANN puro (o que a maioria das submissões faz)

Carrega o dataset de 3M vetores num índice em memória (HNSW, IVF, VP-tree) e roda k-NN por
request. Funciona, mas:

- **Memória**: índice tem ~50-200 MB de footprint, comendo metade do budget.
- **Latência**: cada request paga ~1-5 ms de busca, deixando pouca margem do 1 ms p99.
- **CPU**: trabalho concentrado no hot path da request.

### Destilação (a abordagem deste repo)

A função 5-NN sobre um dataset fixo é **determinística**: `R^14 → {0..5}`. Logo, é aproximável por
um modelo paramétrico. O plano:

1. **Offline** — rode o oráculo brute-force uma vez (~5 min em GPU) para gerar (query, verdict).
2. **Treine** um MLP pequeno que imita o veredito.
3. **Em produção** — embarque só os pesos. Inferência vira matmul de microssegundos.

Em ML isso é **knowledge distillation** de um classificador não-paramétrico (k-NN) num
paramétrico (MLP). O custo offline é gratuito — o runtime nunca paga por isso.

## Por que destilação funciona aqui

A escolha não é arbitrária — depende de uma propriedade empírica do dataset. Quando rodamos k-NN
leave-one-out sobre os 3M refs (k=5 e k=25), descobrimos que **96.5% dos pontos vivem em
vizinhanças totalmente homogêneas** (todos os 25 vizinhos do mesmo label) e só **3.5% ficam na
fronteira de classe**. Os detalhes estão em [02 — Estrutura dos dados](./02-estrutura-dos-dados.md).

Isso significa que:

- 96.5% do problema é trivial — qualquer classificador decente acerta com 100%.
- O jogo é decidido nos 3.5% de fronteira.
- Vale a pena ter um modelo que dá fast-path para o cluster interior e um slow-path mais
  cuidadoso para a fronteira.

Essa é a base da **arquitetura two-box** (Box-A = clusters interiores, Box-B = fronteira) descrita
em [03 — Partição e halo](./03-particao-e-halo.md).

## Visão de alto nível do runtime

```
                      ┌─── nginx (stream mode, UDS upstream) ───┐
   client ──9999──>   │       round-robin TCP forwarding         │
                      └────────────┬───────────────┬─────────────┘
                                   │               │
                          ┌────────▼─────┐ ┌──────▼────────┐
                          │  papagaio-   │ │  papagaio-    │
                          │     api1     │ │     api2      │
                          └────────┬─────┘ └──────┬────────┘
                                   │              │
   POST /fraud-score ────────────> │              │
                                   │              │
   vectorize::vectorize  (round4, [f32; 16])
        │
        ▼
   router::infer          (MLP 14→32→32→3, ~1635 params)
        │
        ├── P(B) > 0.5 ──> slow_path::ivf_k5  (IVF sobre Box-B, ~212k refs)
        │
        └── senão       ──> argmax(P(A-Legit), P(A-Fraud)) → fraud_count 0 ou 5
                                   │
                                   ▼
                       respond &'static [u8] pré-renderizada (zero-alloc)
```

Detalhamento dos componentes:

- **Router** (1635 params, 6.5 KB) → [04 — Router MLP](./04-router-mlp.md)
- **Slow path** (IVF sobre Box-B) → [05 — Slow path IVF](./05-slow-path-ivf.md)
- **Backend** (Rust, hyper direto, byte parser) → [06 — Backend Rust](./06-backend-rust.md)
- **Guardrails numéricos** (round4, GELU coupling) → [07 — Guardrails numéricos](./07-guardrails-numericos.md)

## Resultado atual

Contra o `test/test.js` oficial da rinha (54.100 queries, ramp 2 min até 900 RPS):

```
score:                 6000 / 6000
score_p99:             3000  (saturado; p99 < 1 ms)
score_det:             3000
p99:                   0.15 ms
TP / TN / FP / FN:     24037 / 30023 / 0 / 0
HTTP errors:           0
```

Working set total na memória:

| Componente                  | Tamanho  |
|-----------------------------|----------|
| Box-B refs (padded 16f)     | 13.6 MB  |
| Box-B labels                | 213 KB   |
| Centroides IVF + offsets    | 17 KB    |
| Router weights              | 6.5 KB   |
| Rust binary (release)       | ~5 MB    |
| **Total container memory**  | ~270 MB  |

Para a jornada cronológica de score e p99 (5700 → 6000), veja
[11 — Evolução do score](./11-evolucao-do-score.md).

## Trade-offs honestos

**Pró**:

- p99 dominado por matmul + AVX2 brute → satura `score_p99` em +3000.
- Memória dominada pelos pesos do MLP + 13.6 MB de Box-B → folgado dentro de 350 MB.
- Trabalho pesado offline, runtime previsível.

**Contra**:

- `score_det` depende de quão bem o MLP imita o k-NN. Erros na fronteira pesam direto.
- Coupling forte entre pipeline de treino e código de inferência (ver
  [07 — Guardrails numéricos](./07-guardrails-numericos.md)).
- Re-treinar exige rodar o pipeline offline inteiro de novo (~10 min em GPU + ~5 min de
  k-means + setup).
