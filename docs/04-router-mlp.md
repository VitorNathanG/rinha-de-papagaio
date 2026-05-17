# 04 — Router MLP

O router é um MLP minúsculo (14 → 32 → 32 → 3, **1.635 parâmetros**, 6.5 KB) que prediz
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
    nn.Linear(14, 32),
    nn.GELU(approximate="tanh"),
    nn.Linear(32, 32),
    nn.GELU(approximate="tanh"),
    nn.Linear(32, 3),
)
```

- **Entrada**: 14 dims (o query vetorizado, mesmo formato dos refs).
- **Escondidas**: 32 unidades × 2 camadas. Por que 32? Suficiente para hit recall_B ≈ 99.93% sem
  inflar params além do necessário. Trade-off testado empiricamente.
- **Saída**: 3 classes via softmax.
- **Ativação**: GELU com `approximate="tanh"` (a fórmula GPT-2/OpenAI). Detalhes da escolha em
  [07 — Guardrails numéricos](./07-guardrails-numericos.md).

Tamanho final em bytes: `1635 × 4 = 6.540`. Cabe em uma única página de memória.

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

O estado salvo é o do **melhor val_loss**, não o último — o método clona `state_dict()` toda vez
que `val_loss` cai mais que `MIN_DELTA`.

### Métricas durante o treino

Por epoch, o script logga:

```
ep  39/500  train_loss=0.0014  val_loss=0.0013  acc=99.98%
            recall=[A-L=99.99% A-F=99.99% B=99.93%]  elapsed=42s *
```

`recall_B` é o número que mais importa para o score, porque um miss-route B → A é um verdict
flip provável (o argmax de A-Legit/A-Fraud não tem por que estar correto para um query que
deveria ir pro slow path).

Tipicamente alcançamos `recall_B ≈ 99.93%` — isto é, ~7 a cada 10k queries de B são roteadas
erradamente para A. O efeito no score é mitigado pelo fato de que **a maioria dessas misroutes
cai no argmax certo por acidente** (refs de B perto da fronteira de cluster legit tendem a ter
prior legit, etc.).

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

`export_router.py` lê `data/router.pt` e escreve `data/router_weights.bin` como **1635 f32
little-endian, sem header**, na ordem:

```
w1 (32, 14)   = 448 floats
b1 (32,)      =  32
w2 (32, 32)   = 1024
b2 (32,)      =  32
w3 (3, 32)    =  96
b3 (3,)       =   3
                ----
                1635
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
