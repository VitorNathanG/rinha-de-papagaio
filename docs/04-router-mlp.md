# 04 — Router MLP

O router é um MLP pequeno (14 → 64 → 64 → 3, **5.315 parâmetros**, 20,8 KB) que prediz
diretamente a classe do box (A-Legit / A-Fraud / B) de um query vector. Ele substitui o que
seriam dois modelos separados (um "router binário" + um "classificador interno") por um único
classificador de 3 classes.

## Por que três classes e não router + classificador separados

A formulação ingênua seria:

```
1. router    P(B) ∈ [0,1]   →  se P(B) > τ, slow path
2. modelo A  P(fraud|x) ∈ [0,1]  →  se P(fraud) > 0.5, count = 5; senão count = 0
```

Dois modelos = dois conjuntos de pesos, dois forwards por query, e a calibração do router é
treinada *isoladamente* (não sabe nada sobre confiança do modelo A).

Colapsando em um único classificador de 3 classes:

```
P(A-Legit), P(A-Fraud), P(B)  ← softmax
```

- Um único forward por query.
- O modelo vê exemplos de B durante o treino, então sua **incerteza no contorno é calibrada por
  dados** — diferente do "modelo A puro" que nunca veria refs de B em treino e seria
  confiantemente errado lá.
- Pesos compartilhados nas camadas escondidas — as features que distinguem cluster legit de
  cluster fraud são as mesmas que distinguem ambos de B.
- Roteamento (`P(B) > 0.5`) e classificação interna (`argmax(P(A-Legit), P(A-Fraud))`) saem do
  mesmo softmax, sem inconsistência possível.

## Shape e ativação

```python
nn.Sequential(
    nn.Linear(14, 64),
    nn.GELU(approximate="tanh"),
    nn.Linear(64, 64),
    nn.GELU(approximate="tanh"),
    nn.Linear(64, 3),
)
```

- **Entrada**: 14 dims (o query vetorizado, mesmo formato dos refs).
- **Escondidas**: 64 unidades × 2 camadas. Por que 64? Com pristine box labels (sem halo), a
  fronteira é muito mais limpa e 32 unidades já chegavam a recall_B ≈ 99.93% — mas a evidência
  empírica do treino atual mostra que dobrar pra 64 reduz o floor de misroutes B→A no 3M de ~31
  pra ~24 (5,3× mais params, ainda <1% do p99 do backend). Trade-off favorável.
- **Saída**: 3 classes via softmax.
- **Ativação**: GELU com `approximate="tanh"` (a fórmula GPT-2/OpenAI). Detalhes da escolha em
  [07 — Guardrails numéricos](./07-guardrails-numericos.md).

Tamanho final em bytes: `5315 × 4 = 21.260`. Cabe folgado em poucas páginas de memória.

## Treino

`train_router.py` cuida do treino. Setup:

- **Split**: 80% train / 10% val / 10% test, seed fixo (42).
- **Otimizador**: Adam, `lr = 1e-3`.
- **Batch**: 8192 (todo o dataset cabe na GPU, então batches grandes são livres).
- **Loss**: `CrossEntropyLoss` com class weights inverse-frequency.

### Class weights — por que são essenciais

Sem class weights, B é ~3.5% do dataset. Cross-entropy minimiza loss média; o modelo aprende a
**sempre predizer A** (qualquer A — basta o argmax estar certo) e atinge ~96.5% de acurácia
trivialmente. Recall de B fica próximo de 0 — quebrando completamente o routing.

A correção é inverse-frequency:

```python
class_counts = bincount(box[train_idx])             # ex.: [1.95M, 0.95M, 0.21M]
weights = class_counts.sum() / (class_counts * 3)   # média = 1 por construção
```

Valores típicos:

```
A-Legit:  0.51
A-Fraud:  1.05
B:        4.71   ← 9× maior que A-Legit, força o modelo a tratar B com seriedade
```

A divisão por `num_classes = 3` mantém a magnitude média do gradiente próxima de 1 — útil para o
learning rate scale ficar coerente.

### Early stopping

`PATIENCE=30`, `MIN_DELTA=1e-5`. O modelo treina até `EPOCHS=500` (default alto) mas para cedo
quando o val_loss não melhora por 30 epochs seguidas. Tipicamente converge em ep 30-70.

O estado salvo é o do **menor misroute_3M** (não menor val_loss). Empiricamente os dois divergem
— a epoch com menor val_loss costuma ter ~3-4× mais misroutes que a epoch de melhor recall em B.
Como o objetivo do router é justamente zerar misroute, salvamos por ele. `val_loss` continua
dirigindo a patience (sinal mais estável que o número de misroutes, que oscila ±30 por causa de
não-determinismo do argmax em samples borderline).

### Métricas durante o treino

CM completa sobre os 3M (train + val + test) por epoch — chunked em 100k pra contornar um bug
do ROCm onde forward em batch de 3M devolve logits corrompidos:

```
ep  62/500  train_loss=0.0013  val_loss=0.0018  acc_3M=99.94%
            recall=[A-L=100.00% A-F=99.82% B=99.98%]  misroute_3M=24/105504  elapsed=19s *
```

`misroute_3M` é o número safety-critical: ref com true=B classificado pelo router como A bypassa
o slow path em prod. Tipicamente alcançamos ~20-30 misroutes (≈ 0,02% dos 105k true-B); o efeito
no score é nulo porque **na prática esses misroutes geram o mesmo verdict que o slow path
geraria** (validamos 96/96 contra brute k=5 sobre Box-B post-halo).

## Inferência runtime

Em Rust, `backend/src/router.rs` faz o forward inline:

```rust
fn infer(w: &Weights, x: &[f32; 16]) -> [f32; 3] {
    // h1 = GELU(W1 x + b1)
    // h2 = GELU(W2 h1 + b2)
    // logits = W3 h2 + b3
    // probs = softmax(logits)
    ...
}
```

Sem AVX explícito — só loops aninhados que o autovetorizador do LLVM consegue resolver dada a
shape minúscula. O custo total é ~14 × 32 + 32 × 32 + 32 × 3 = 1.536 MACs, mais 64 GELUs.

GELU é implementado com um `tanh_approx` Padé[7/6] inline em vez de `libm::tanhf`. Detalhes em
[07 — Guardrails numéricos](./07-guardrails-numericos.md).

Softmax é numericamente estável (subtrai o max antes do exp), três `expf` no caminho — o último
transcendental que sobrou no hot path. Trocá-lo por um Padé é o próximo passo no
`LATENCY_BACKLOG.md` (item 2 do Tier 0).

## Regra de routing em runtime

```rust
let probs = router::infer(&w, &v);
let count: u8 = if probs[2] > 0.5 {
    unsafe { slow_path::ivf_k5(&v, ...) }   // P(B) > 0.5 → exact-ish
} else if probs[0] > probs[1] {
    0   // P(A-Legit) > P(A-Fraud)
} else {
    5   // P(A-Fraud) > P(A-Legit)
};
```

`probs[2] > 0.5` é o threshold de roteamento. Valores mais altos roteiam menos para o slow path
(latência menor, score pior); mais baixos roteiam mais (latência maior, score melhor). 0.5 é o
sweet-spot empírico — manda ~3.5-7% das queries para slow path.

## Exportação dos pesos para Rust

`export_router.py` lê `data/router.pt` e escreve `data/router_weights.bin` como **5315 f32
little-endian, sem header**, na ordem:

```
w1 (64, 14)   =  896 floats
b1 (64,)      =   64
w2 (64, 64)   = 4096
b2 (64,)      =   64
w3 (3, 64)    =  192
b3 (3,)       =    3
                ----
                5315
```

A ordem é `out × in` (a convenção `nn.Linear.weight` do PyTorch). O loader em
`backend/src/router.rs:load_weights` espera exatamente essa ordem e essa quantidade — qualquer
mudança de shape exige atualizar os dois lados (constantes `D_IN`, `H`, `D_OUT` no Rust;
`hidden`, `depth` no Python).

O assert no exporter:

```python
if hidden != 32 or depth != 2:
    raise SystemExit(...)
```

é deliberado — protege contra a regressão clássica de re-treinar com config diferente e esquecer
de atualizar o backend.
