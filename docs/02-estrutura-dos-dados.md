# 02 — Estrutura dos dados

A geometria do dataset é o que torna a destilação viável. Esta seção documenta as descobertas
empíricas que moldaram cada decisão arquitetural posterior.

## O dataset

`references.json.gz` no repo da rinha contém **3.000.000 vetores de 14 dimensões** rotulados como
`fraud` ou `legit`. As 14 dimensões são features normalizadas em `[0, 1]` (com sentinelas `-1` em
duas posições — ver `prepare.py`), descritas em detalhe em
`../rinha-de-backend-2026/docs/en/DETECTION_RULES.md`.

Distribuição de labels:

```
fraud: 33.31%
legit: 66.69%
```

A normalização e os limiares (`MAX_AMOUNT=10000`, `MAX_KM=1000`, etc.) estão replicados em
`backend/src/vectorize.rs` para que a vetorização runtime case com a do data generator.

## Distribuição k=5 leave-one-out

Para cada ref `q` no dataset, computamos seus 5 vizinhos mais próximos *excluindo `q`* e contamos
quantos dos 5 são `fraud`. Em GPU (brute-force, fp16, ~5 min num RX 9070 XT) com `label.py`:

```
0/5:  1.951.240  (65.04%)   ← cluster puro legit
1/5:     16.476  ( 0.55%)
2/5:     33.005  ( 1.10%)
3/5:     32.810  ( 1.09%)
4/5:     16.337  ( 0.54%)
5/5:    950.132  (31.67%)   ← cluster puro fraud

puros (count==0 ou count==5):  96.71%
não-puros (qualquer entre):     3.28%
zona de empate (count in {2,3}): 2.19%
```

## Distribuição k=25 leave-one-out

A mesma medida com `k=25` (rodar `K=25 uv run python label.py`):

```
 0/25:   1.947.958  (64.93%)
 1..24/25 (somados):  105.472  ( 3.52%)   ← picos em 12/25 e 13/25
25/25:    946.570  (31.55%)

puros (count==0 ou count==25):  96.48%
não-puros:                       3.52%
```

## Três insights estruturais

### 1. O clustering é estrutural, não um artefato de k pequeno

Ir de `k=5` para `k=25` converte só **0.23%** dos pontos de "puro" para "não-puro". Se a
homogeneidade fosse ilusão de vizinhança pequena, ampliar a vizinhança quebraria muitos clusters
— mas não quebra. A fronteira de classe é genuinamente **fina no espaço de input**.

### 2. A fronteira é uma superfície real, não um gradiente de incerteza

Os pontos não-puros em `k=25` se concentram no marco exato dos **50/50** (12/25 = 16.109 pontos,
13/25 = 16.079). Isso é assinatura de uma **superfície de decisão genuína** atravessada por
pontos, e não de um continuum onde a incerteza cresce gradativamente.

### 3. 96.5% do problema é trivial

Qualquer classificador razoável — até regressão logística — acerta no interior dos clusters. A
contest é decidida nos 3.5% de fronteira. Modelos que tentam aprender o problema "como um todo"
acabam alocando capacidade no lugar errado: a loss média é dominada pelos clusters puros e a
fronteira fica subdimensionada.

Isso motiva a **partição by structure** (ver
[03 — Partição e halo](./03-particao-e-halo.md)) — separar os 96.5% triviais dos 3.5% críticos e
dar tratamento dedicado a cada um.

## A pegadinha: a classe BORDERLINE

O data generator da rinha (`data-generator/main.c:737-740`) tem uma classe rara de refs chamada
**BORDERLINE**, cujos labels são atribuídos **uniformemente ao acaso**:

```c
if (profile == BORDERLINE) {
    label = (rand_double(&rng) < 0.5) ? "legit" : "fraud";
}
```

Estimativa: ~3% dos refs do dataset são BORDERLINE. Qualquer veredito k=5 cujo top-5 toque um ref
BORDERLINE é, na prática, um coin flip — e isso vale tanto para o oráculo brute-force quanto para
qualquer modelo. **Isso é o noise floor** do desafio.

### Impacto no score

Por bastante tempo o melhor score alcançado foi **5700/6000** (5 verdict gaps de 54.060 = ~0.01%
de erro). Esses 5 mismatches estavam concentrados em queries que tocavam refs BORDERLINE no
top-5. Aparentemente isso era um teto físico.

Não era — só *parecia* ser. O salto para 6000/6000 veio de dois bugs *nossos* que foram
mascarados pelo noise floor:

1. **round4 mismatch** no vetorizador (queries não estavam sendo arredondadas como o data
   generator faz antes do brute-force) → fechou 1 mismatch.
2. **Inward purity** na partição (refs no contorno externo de cluster puro estavam em Box-A em
   vez de Box-B) → fechou 4 mismatches.

Ambos descritos em [07 — Guardrails numéricos](./07-guardrails-numericos.md) e
[03 — Partição e halo](./03-particao-e-halo.md).

**Lição**: "score plateau" pode ser teto real, mas pode ser teto disfarçando bugs que produzem
erro do mesmo tamanho do noise. Vale investigar os mismatches individualmente
([09 — Debug e profiling](./09-debug-e-profiling.md)) antes de declarar fim de jogo.

## Performance da labeling em GPU

`label.py` faz brute-force k-NN em fp16 com chunked top-K merge:

```
GPU:       AMD Radeon RX 9070 XT (17.1 GB, ROCm 7.2)
k=5:       ~4 m 20 s    (~11.500 q/s)
k=25:      ~4 m 22 s    (~11.500 q/s)
```

A versão chunked mantém um running top-K `(BATCH, K)` e processa refs em tiles `(BATCH, CHUNK)`,
mergeando via `topk` na concatenação. A versão naive materializaria a matriz `(BATCH, N)` inteira
e estouraria VRAM para BATCH > 512.

fp16 é o default — diverge do fp32 em ~0.5% das queries, todas off-by-one no count, bem abaixo do
noise floor. `FP32=1` força o caminho exato.

O termo `||q||²` é omitido na distância porque é constante por linha e não muda o ranking. Ver
`label.py:117-127` para o kernel completo.
