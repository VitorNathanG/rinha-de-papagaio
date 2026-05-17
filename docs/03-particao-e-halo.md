# 03 — Partição em boxes e border halo

A descoberta empírica de que **96.5% dos refs ficam em vizinhanças homogêneas** (ver
[02 — Estrutura dos dados](./02-estrutura-dos-dados.md)) motiva uma decomposição estrutural do
dataset em três classes mutuamente exclusivas. Esta seção descreve a partição, o slow path que
opera sobre ela, e o bug do "inward purity" que motivou o border halo.

## A partição em três classes

`partition.py` lê `data/labels.npy` e `data/fraud_counts_k25.npy` e classifica cada um dos 3M
refs em uma de três classes:

| Classe       | Definição                                                  | Significado |
|--------------|------------------------------------------------------------|-------------|
| **A-Legit** (0) | `label == legit`  AND  `count_k25 == 0`                | Interior de cluster legit (todos os 25 vizinhos também legit) |
| **A-Fraud** (1) | `label == fraud`  AND  `count_k25 == 25`               | Interior de cluster fraud (todos os 25 vizinhos também fraud) |
| **B** (2)        | qualquer outra coisa                                   | Vizinhança mista OU label outlier |

A classe **B** captura duas situações distintas:

1. **Mixed neighborhoods** — `count_k25 in 1..24`. Pontos genuinamente na fronteira.
2. **Label outliers** — refs cujo label discorda da vizinhança totalmente homogênea
   (ex.: `label == fraud` num cluster onde `count_k25 == 0`). Tipicamente refs BORDERLINE
   (label aleatório do data generator).

Tamanho antes do halo:

```
A-Legit (0):  1.951.240   (65.04%)
A-Fraud (1):    950.132   (31.67%)
B       (2):    105.504   ( 3.52%)   ← slow path
```

Por que **k=25** e não k=5? Largura de vizinhança maior:

- Faz outliers caírem em B em vez de virarem ruído dentro de A.
- Não é a métrica usada em produção (rinha sempre é k=5), serve só para a partição.
- O delta `k=5 → k=25` em refs não-puros é só 0.23% — sinal de que k=25 não está exagerando.

## Por que partition por estrutura, e não por label

Uma alternativa óbvia seria treinar um único modelo gigante em tudo. Não funciona bem por dois
motivos:

1. **Otimização de loss**: o BCE médio é dominado pelos clusters puros (96.5% dos dados). Loss
   gradients alocam capacidade onde já temos boa solução, ignorando os 3.5% que decidem o jogo.

2. **Calibração de incerteza**: um modelo treinado em tudo aprende a ser confiantemente correto
   no interior — e *confiantemente errado* no contorno, porque não tem sinal de treino para ser
   incerto lá. Routing baseado só em confidence falha justamente onde mais importa.

A partição força o problema a ser tratado por região:

- Para refs em A, basta saber o argmax de `P(A-Legit) vs P(A-Fraud)`. O classificador é trivial.
- Para refs em B, fazemos k=5 exato (ou aproximado via IVF) sobre o subset Box-B. O slow path tem
  cuidado, mas roda em ≤ 5% das requests.

A junção é feita pelo **router** (ver [04 — Router MLP](./04-router-mlp.md)) que prediz
diretamente as 3 classes a partir do query vector.

## O slow path e o subset Box-B

Em runtime, queries roteadas para B fazem k=5 exato — não sobre os 3M refs, mas sobre **só os
refs que pertencem a Box-B**. A intuição:

- Se um query é interior de cluster legit, seus 5 vizinhos quase certamente são todos `A-Legit`
  (count=0).
- Se um query é interior de cluster fraud, idem, todos `A-Fraud` (count=5).
- Se um query está na fronteira, seus 5 vizinhos têm grande chance de também serem refs de
  fronteira → estão em B.

A premissa é que, dado um query roteado a B, **fazer k=5 sobre Box-B aproxima muito bem o k=5
sobre os 3M completos**. Vamos ver que essa premissa é quase verdadeira — e o "quase" motiva o
border halo.

## O bug: inward purity vs outward boundary

A definição de A-Legit / A-Fraud é uma **checagem inward**: olha para os 25 vizinhos *do
próprio ref* e exige unanimidade. Mas há uma situação em que essa checagem falha:

> Um ref X pode ter os 25 vizinhos próprios *todos legit* (logo classificado como A-Legit),
> *e mesmo assim* ser o vizinho mais próximo de um query borderline que orbita o cluster
> pela parte de fora.

Geometricamente, X está na **borda externa do cluster legit**, encarando o espaço onde queries
borderline existem. Inward purity diz "X está rodeado de legit". Outward boundary diz "X é o ref
mais próximo de muitos queries que estão fora do cluster".

### O sintoma

O slow path procura no Box-B só. Se X (que é A-Legit) é o top-1 NN verdadeiro de um query
borderline, o slow path nunca acha X — ele acha o next-nearest *em Box-B*, que pode ser outro
ref legit (ok) ou um ref fraud (verdict flip).

Nos primeiros runs em prod, ficamos com **5 mismatches deterministicos** contra `test-data.json`
(3 FP + 2 FN). Analisando case-by-case com `analyze_mismatches.py`
([09 — Debug e profiling](./09-debug-e-profiling.md)), descobrimos que:

- 4 dos 5 eram exatamente isso: o top-5 verdadeiro em 3M incluía 1+ refs em A-Legit ou A-Fraud
  que faltavam no Box-B.
- O 5º era outro bug (round4 mismatch — ver [07 — Guardrails numéricos](./07-guardrails-numericos.md)).

## A solução: border halo via nearest-opp

Em vez de tentar redefinir "puro" com k maior (k=100, k=500 — testados e não pegam refs de
borda externa em clusters grandes), introduzimos uma métrica direta:

> **`nearest_opp[i]`** = distância euclidiana de `refs[i]` ao ref *de label oposto* mais próximo.

Computado em GPU via `nearest_opp.py` — brute-force chunked (queries × outros-de-label-oposto)
em fp16, ~5 min no RX 9070 XT. Output: `data/nearest_opp_dist.npy`, 12 MB f32.

Esse número é a métrica genuína de "está perto da fronteira de classe?". Um ref com `nearest_opp`
pequeno está geograficamente perto de algum ref do label oposto — exatamente os refs que queremos
em Box-B.

### A promoção

`border_halo.py` lê `box_labels.npy` (output do `partition.py`) e promove para Box-B todos os
refs em A cujo `nearest_opp < D`:

```python
in_A = (box_base != 2)
halo_mask = in_A & (opp < D)
box_new[halo_mask] = 2
```

Default D = 0.23. Esse valor foi calibrado com `sim_borderhalo.py`, que faz um sweep de D, monta
um Box-B candidato para cada D, e replaya os 5 mismatches conhecidos em brute force. D = 0.23 é o
**menor threshold que recupera todos os 11 refs faltando no top-5 NN** dos 5 mismatches.

### Idempotência

`border_halo.py` é re-executável com um D diferente sem aplicar o halo duas vezes. O truque:

- Primeira execução: snapshota `box_labels.npy` para `box_labels.before_halo.npy`.
- Execuções seguintes: leem da snapshot, ignoram o `box_labels.npy` atual (provavelmente já tem
  halo aplicado).

Isso permite re-tuning de D sem ter que re-rodar `partition.py`.

## Tamanho do Box-B com halo D=0.23

```
Sem halo:    105.504 refs  (3.52% do dataset)
Com D=0.23:  212.733 refs  (7.09% do dataset)

Refs promovidos A → B:    107.229
  - de A-Legit:           ~95k
  - de A-Fraud:           ~12k
```

A assimetria reflete que o cluster legit é maior e tem fronteira mais extensa.

Working set do slow path:

```
Box-B refs (padded a 16 floats × 4 B):  13.6 MB   (213k × 64 B)
Box-B labels:                            213 KB
```

13.6 MB não cabe no L3 de 4 MB do Mac Mini target — por isso o slow path não é brute force, mas
sim IVF. Ver [05 — Slow path IVF](./05-slow-path-ivf.md).

## Efeito no score

Combinado com o fix do round4:

```
Sem halo, sem round4:  5700/6000  (3 FP + 2 FN)
+ round4:              5910/6000  (2 FP + 1 FN — derruba 2 mismatches)
+ halo D=0.23:         6000/6000  (0 FP + 0 FN)
```

p99 essencialmente inalterado (212k vs 105k não muda nada porque o IVF mantém o per-query
working set fixo via nprobe). O cluster size médio sobe de 206 → 415 mas nprobe=16 ainda cabe em
L2.
