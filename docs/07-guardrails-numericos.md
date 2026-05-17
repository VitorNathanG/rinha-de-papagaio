# 07 — Guardrails numéricos

Quatro pontos no código onde a numerica do treino tem que casar **exatamente** com a numerica do
runtime. Cada um foi descoberto por um bug que custou pontos no score; documentá-los aqui ajuda a
não regredir.

## 1. round4 na vetorização

### O problema

O data generator da rinha (`data-generator/main.c:774`) arredonda o vetor 14-dim de cada test
entry para **4 casas decimais** *antes* de rodar o brute-force k=5 que define `expected_approved`.
As 3M references também são round4'd no mesmo lugar (`main.c:735`). Logo, o k-NN canônico é
sobre um **grid uniforme de 4 decimais**.

Nosso backend originalmente computava o query vector em fp32 cru. Para a maioria das queries isso
não faz diferença — o top-5 vs rank-6 ficam vários units de distância apart. Mas em queries de
empate muito apertado (gap < 1e-5), o `round4` reordena rank-5 vs rank-6 e o veredito muda.

### O caso 5472

Test entry 5472 era deterministicamente errado tanto no backend quanto numa simulação brute em
Python full-3M. Ambos diziam `count=2 → legit`, mas `expected_approved=fraud`.

```
rank 5   ref# 703996   d = 0.23508085   label=legit
rank 6   ref#1707961   d = 0.23508862   label=fraud   ← gap de 7.7 ppm
```

Com `round4` no query, os dois trocam de lugar — rank-5 vira o ref fraud, count vira 3, veredito
vira `fraud`. Match com `expected_approved`.

### O fix

`backend/src/vectorize.rs:97-99`:

```rust
// Round to 4 decimal places — matches `round4` applied by the data
// generator (data-generator/main.c:774) before its kNN.
for i in 0..14 {
    out[i] = (out[i] * 10000.0).round() / 10000.0;
}
```

Sweep do `test-data.json` antes e depois: **exatamente 1 entry mudou veredito** (5472:
fraud → correto), zero outras quebraram. Score subiu de 5790 → 5910 só com essa linha.

### Lição

Quando o oráculo de ground-truth tem uma transformação numérica (arredondamento, normalização,
quantização) antes do search, **a inferência tem que aplicar a mesma transformação**. Não basta
"a feature está close enough" — para empates apertados, qualquer transformação reordena.

## 2. GELU coupling (train ↔ inference)

### O problema

PyTorch oferece duas formulações de GELU:

```python
nn.GELU()                          # default: 0.5 * x * (1 + erf(x / √2))
nn.GELU(approximate="tanh")        # OpenAI/GPT-2: tanh-based approximation
```

A diferença numérica entre as duas é da ordem de **1e-4 a 1e-5** em magnitude — irrelevante em
casos típicos. Mas o router opera num espaço onde diferenças desse tamanho **flip vereditos em
queries borderline**.

### O caso

Durante o swap para `tanh-GELU` no Rust (para evitar `libm::erff`), esquecer de atualizar o
`train_router.py` causou divergência train ↔ infer. Resultado: queries que o modelo treinado
"sabia" classificar viraram falsos negativos porque a soft-max no Rust estava ~1e-4 diferente do
que o treino esperava.

### O fix

Treino e inferência têm que casar bit-for-bit. Atualizamos os dois lados no mesmo commit
(`57a930c`):

```python
# train_router.py
gelu = lambda: nn.GELU(approximate="tanh")
```

```rust
// backend/src/router.rs
fn gelu(x: f32) -> f32 {
    // 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
    const COEFF: f32 = 0.044715;
    let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
    0.5 * x * (1.0 + tanh_approx(inner))
}
```

### Lição

Mudar a função de ativação **exige re-treinar e re-exportar**. Não dá para trocar só de um lado.
Qualquer mudança num primitive matemático no caminho da inferência (ativação, normalização,
softmax) tem que ser propagada para o pipeline de treino.

## 3. Padé[7/6] tanh inline

### O problema

Mesmo após trocar `erff` por `tanhf`, o `libm::tanhf` em pure-Rust faz fan-out para `expm1f`
internamente. No profile, esses dois símbolos somavam ~5% do CPU a 10k RPS — não dá pra deixar.

### O fix

Polinômio racional Padé[7/6] inline, com saturação para ±1 fora da região de boa aproximação:

```rust
#[inline]
fn tanh_approx(x: f32) -> f32 {
    if x.abs() >= 4.97 {
        return x.signum();
    }
    let x2 = x * x;
    let num = x * (135135.0 + x2 * (17325.0 + x2 * (378.0 + x2)));
    let den = 135135.0 + x2 * (62370.0 + x2 * (3150.0 + x2 * 28.0));
    num / den
}
```

Propriedades:

- Erro absoluto < 2e-7 para |x| ≤ 4.97.
- Saturação em |x| ≥ 4.97 (tanh(4.97) ≈ 0.99989, discontinuidade < 1e-4 na borda).
- Custo: ~7 muls + 6 adds + 1 div ≈ metade dos ciclos de `libm::tanhf`.

A justificativa do range 4.97: o MLP nunca alimenta valores grandes ao tanh — a entrada é
`sqrt(2/π) * (x + 0.044715 * x³)` onde `x` é uma soma pós-Linear pequena. Empiricamente o input
ao tanh fica em `|x| < 3` em 99.99% das queries. O caso 4.97 está bem do lado seguro.

### Lição

`libm` é correta mas não é otimizada para o nosso caso de uso. Quando uma função transcendental
domina o profile, **inline + restrição de domínio** quase sempre ganha. Padé[m/n] é uma boa
opção default; Chebyshev/minimax se a precisão é crítica.

## 4. Shape congelada do router em ambos os lados

### O problema

`train_router.py` aceita `HIDDEN` e `DEPTH` como env vars. Treinar com `HIDDEN=64` e re-exportar
sem mudar o Rust resultaria em backend lendo lixo dos arquivos `.bin`.

### O fix

`export_router.py` carrega o checkpoint, lê `hidden` e `depth` do metadata salvo, e aborta se
não for o esperado:

```python
if hidden != 32 or depth != 2:
    raise SystemExit(
        f"Rust backend assumes hidden=32 depth=2, "
        f"found hidden={hidden} depth={depth}"
    )
```

Do lado Rust, as constantes são em compile time:

```rust
const D_IN: usize = 14;
const H: usize = 32;
const D_OUT: usize = 3;
pub const N_FLOATS: usize = D_IN * H + H + H * H + H + H * D_OUT + D_OUT; // 1635
```

E o loader valida o tamanho do arquivo:

```rust
let expected = N_FLOATS * 4;
assert_eq!(bytes.len(), expected, "router_weights.bin should be {expected} bytes");
```

### Lição

Qualquer constante numérica que aparece em dois lados do pipeline (treino e inferência) deve ter
uma asserção que verifica a coerência. **Não confie em comentários** ("lembrar de mudar X também")
— transforme em assert.

## 5. Layout binário rígido

Os arquivos `box_b_*.bin` e `router_weights.bin` são raw binary sem header. Qualquer
incompatibilidade silenciosa entre exporter e loader produz lixo numérico sem erro de carregamento.

Defesas:

1. **Padding obrigatório**. `box_b_refs.bin` e `box_b_ivf_centroids.bin` são padded para 16 floats
   por linha. Sem isso, os `_mm256_loadu_ps(r_ptr.add(8))` lêem além do buffer e produzem NaNs.

2. **Sorted by cluster**. Refs e labels em `box_b_refs.bin` e `box_b_labels.bin` *têm* que estar
   na ordem definida pelo `np.argsort(assignments)`. O loader não re-ordena. `offsets` aponta
   para posições absolutas — qualquer desordenação quebra o slow path silenciosamente (errar
   labels, mas sem panic).

3. **CSR offsets em unidades de linha, não bytes**. `offsets[c+1] - offsets[c]` é o número de
   refs no cluster `c`. O loader multiplica por `D_PADDED * 4` para virar bytes onde necessário.

4. **Asserts de tamanho** no `main.rs::load_state`:

   ```rust
   assert_eq!(refs.len(), labels.len() * 16);
   assert_eq!(centroids.len(), nlist * 16);
   ```

   Não pegam todos os bugs (não detectam reordenação errada), mas pegam descompasso de count.

## Checklist ao re-treinar / re-exportar

Se você mexer em algo do pipeline offline, confirme:

- [ ] `train_router.py` e `backend/src/router.rs` usam a mesma ativação.
- [ ] `HIDDEN=32` e `DEPTH=2` (o assert no exporter pega isso, mas vale checar logs).
- [ ] `vectorize.rs::round4` ainda está lá e ainda arredonda para 4 casas.
- [ ] `export_box_b.py` ainda escreve refs padded para 16 floats por linha.
- [ ] `export_box_b.py` ainda sorta por cluster antes de escrever.
- [ ] `D_PADDED = 16` em `slow_path.rs` e `vectorize.rs`.
- [ ] Re-rodar `find_mismatches.py` contra o `test-data.json` da rinha para confirmar 0 fp + 0 fn.

Se algum desses falha, debug com `analyze_mismatches.py` (ver
[09 — Debug e profiling](./09-debug-e-profiling.md)).
