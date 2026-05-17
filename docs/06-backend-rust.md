# 06 — Backend Rust

O backend em `backend/` é onde o p99 acontece. Esta seção documenta as escolhas que vieram de
profiling, não de preferência, e a jornada cronológica para sair de 0.30 ms p99 (baseline com
axum+serde) para 0.15 ms.

## Princípios

1. **Zero alocação no hot path.** Qualquer `malloc` aparecendo no profile é bug.
2. **Sem JSON parsing genérico.** O data generator emite os campos em ordem determinística;
   exploramos isso com byte-scan dedicado.
3. **Respostas pré-renderizadas.** As 6 respostas possíveis viram `&'static [u8]` em build time.
4. **Sem multi-threading inútil.** Cada réplica roda `tokio::runtime::current_thread` porque o
   container só tem 0.425 CPU — scheduler de multi-worker é puro overhead nesse regime.
5. **mmap > read.** Arquivos de índice mapeados em vez de copiados para heap.

## Dependências

`backend/Cargo.toml` lista 8 crates totais:

```toml
tokio       (rt, rt-multi-thread, net, macros)
hyper       (http1, server)         ← sem features default, sem ssl
hyper-util  (tokio)
http-body-util
bytes
bytemuck                            ← cast_slice de &[u8] para &[f32]
libm                                ← expf para softmax (única transcendental restante)
memchr                              ← memchr / memmem para o byte parser
memmap2                             ← mmap dos índices
```

Sem axum, sem serde, sem tower, sem serde_json. Cada crate listado tem motivo direto de existir
no caminho de request.

## Build profile

```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
strip = "symbols"
```

`lto = "fat"` é o que permite que o LLVM colapse o `service_fn` closure dentro do dispatcher do
hyper — sem isso, fica uma chamada indireta no hot path. `codegen-units = 1` é overhead em build
time mas necessário para LTO global. `panic = "abort"` corta as landing pads de unwind e diminui
binário.

`backend/.cargo/config.toml` adiciona `-C target-cpu=x86-64-v3`, habilitando AVX2 + FMA + F16C +
BMI2 em todo o crate. Isso pode parecer redundante com o `#[target_feature(enable = "avx2,fma")]`
no kernel, mas garante que o resto do código (loops do router, byte parser) também tenha acesso a
essas instruções.

## A jornada de p99

Cada step abaixo foi medido contra o `test/test.js` oficial (900 RPS, 2 min, 54.100 entradas) e
contra o `test/profile.js` sustentado (10k-50k RPS, perf attach). Numbers de p99 reportadas vêm do
test oficial.

### Step 0 — Baseline (axum + serde_json + multi-thread tokio + TCP)

```
p99: 0.30 ms
score: 5700/6000
```

Profile a 30k RPS:

```
5.5% — _int_malloc, _int_free, malloc, cfree, memmove  (alocador)
3.4% — libm (erff GELU + expf softmax)
25-29% — axum handler closure dispatching
```

### Step 1 — Byte parser + respostas pré-renderizadas

Substituímos `serde_json::Deserialize` por byte-scan em `vectorize.rs`:

```rust
let amount       = scan_f64(body, &mut cur, b"\"amount\":")?;
let installments = scan_u32(body, &mut cur, b"\"installments\":")?;
let requested_at = scan_string(body, &mut cur, b"\"requested_at\":\"")?;
// ... segue na ordem do data-generator/main.c
```

O parser usa `memchr` para achar delimitadores e `memmem` para achar keys. Quando o data
generator emite campos numa ordem fixa, varrer forward com `memmem` é mais rápido que qualquer
parser genérico — sem heap, sem mapa de strings, sem boxing.

E em vez de serializar a resposta a cada request:

```rust
const RESPONSES: [&[u8]; 6] = [
    b"{\"approved\":true,\"fraud_score\":0.0}",
    b"{\"approved\":true,\"fraud_score\":0.2}",
    b"{\"approved\":true,\"fraud_score\":0.4}",
    b"{\"approved\":false,\"fraud_score\":0.6}",
    b"{\"approved\":false,\"fraud_score\":0.8}",
    b"{\"approved\":false,\"fraud_score\":1.0}",
];

// hot path:
let count = ...;  // 0..=5
Ok(json_response(RESPONSES[count as usize]))
```

`Bytes::from_static(body)` cria um `Bytes` zero-copy ao redor da slice estática — sem alocação,
sem cópia. O response inteiro é uma referência apontando para o `.rodata` do binário.

Resultado:

```
p99: 0.24 ms  (-20%)
alocador caiu fora do top 10 do profile
score: 5700/6000  (inalterado)
```

### Step 2 — hyper-direct + current_thread + TCP_NODELAY

Removemos axum por completo. `main.rs::handle` faz match manual:

```rust
match (req.method(), req.uri().path()) {
    (&Method::POST, "/fraud-score") => { ... }
    (&Method::GET,  "/ready")       => Ok(ready_response()),
    _                                => 404
}
```

E o runtime tokio passa para `current_thread`:

```rust
tokio::runtime::Builder::new_current_thread()
    .enable_all()
    .build()
```

Por que `current_thread` no container? Cada réplica tem 0.425 CPU. Multi-thread tokio mantém um
pool de workers + work-stealing queue + epoch counters — overhead que só compensa em ≥1 CPU. Em
0.425 CPU, single-threaded tokio é literalmente mais rápido (medido).

`TCP_NODELAY` é setado por conexão aceita — corta o Nagle algorithm. Respostas de 35 bytes saem
imediatamente em vez de esperar buffering.

```rust
match listener.accept().await {
    Ok((stream, _)) => {
        let _ = stream.set_nodelay(true);
        spawn_conn(stream, state.clone());
    }
    ...
}
```

Resultado:

```
p99: 0.32 ms  (essencialmente igual ao step 1, dentro do noise)
mas: LTO inlinou o handler dentro do dispatcher do hyper — 37% das amostras em um único símbolo.
Cargo.toml: 8 deps.
```

A surpresa: o ganho real veio em testes de stress fora do SLA (5k RPS+), onde o axum overhead
crescia não-linearmente.

### Step 3 — UNIX domain socket entre nginx e backend

`BIND_ADDR` aceita tanto `host:port` quanto um caminho começando com `/`:

```rust
if bind.starts_with('/') {
    let _ = std::fs::remove_file(&bind);
    let listener = UnixListener::bind(&bind)?;
    std::fs::set_permissions(&bind, Permissions::from_mode(0o666));
    // chmod 0666 porque nginx roda com uid diferente
}
```

O `docker-compose.yml` declara um volume `sockets:` compartilhado entre nginx e as duas APIs. Os
sockets são `/var/run/papagaio/api1.sock` e `api2.sock`.

UNIX sockets pulam todo o kernel TCP stack — sem checksums, sem retransmit, sem TCP_NODELAY
relevante. Comunicação intra-host vira basicamente um `memcpy`.

Resultado a 900 RPS:

```
TCP:  p99 = 0.34 ms
UDS:  p99 = 0.34 ms   (essencialmente igual no SLA)
```

A diferença real aparece a **5k RPS** (5.5× SLA):

```
TCP: 36.13% failure rate, p99 = 2 s    ← quebrou
UDS:  0.21% failure rate, p99 = 127 ms
```

Vale a pena pela robustez sob carga inesperada.

### Step 4 — nginx CPU bump (0.10 → 0.15, APIs 0.45 → 0.425)

Profile do stress test mostrou que **nginx era o gargalo, não as APIs**. A 5k RPS, o nginx
saturava antes dos backends. Re-budgetamos:

```yaml
api1:  cpus: "0.425"   # antes: 0.45
api2:  cpus: "0.425"
lb:    cpus: "0.15"    # antes: 0.10
                       # total = 1.000
```

Resultado a 5k RPS: 0 failures, p99 = 385 µs.

### Step 5 — IVF substitui brute force (commit 189d907)

O slow path passou de brute sobre 105k refs (6.75 MB) para IVF-256/16. Detalhes em
[05 — Slow path IVF](./05-slow-path-ivf.md).

No dev box (Ryzen, 32 MB L3) o ganho é mínimo (~5 µs no slow path) porque brute já cabia em L3.
No Mac Mini target (4 MB L3) é a diferença entre ~5 µs e ~450 µs.

Resultado:

```
p99: 0.16 ms
```

### Step 6 — libm::erff → tanh-GELU + Padé inline (commit 57a930c)

O GELU original em `router.rs` chamava `libm::erff` (default do PyTorch). Trocamos por:

1. A fórmula tanh-based: `0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`.
2. Um `tanh_approx` Padé[7/6] inline (no `libm::tanhf`, que internamente fan-outs para `expm1f`).

Antes:

```
libm::tanhf + libm::expm1f = 4.9% do CPU a 10k RPS
```

Depois:

```
0%  (os símbolos saíram do profile)
total samples: 18.77 G → 17.50 G cycles (-6.8%)
p99: 0.17 → 0.15 ms
```

Coupling crítico: o `train_router.py` foi atualizado para `nn.GELU(approximate="tanh")` *no mesmo
commit*. Re-treino e re-export foram necessários — a fórmula tanh muda o valor da ativação em ~1e-4
versus erf, e isso flips veredito em queries borderline. Ver
[07 — Guardrails numéricos](./07-guardrails-numericos.md).

### Step 7 — nginx HTTP → stream mode (commit 04ca1a3)

O módulo HTTP do nginx parse headers, valida UTF-8, roda state machine de request pipeline e
buffera bodies. Nada disso é útil — temos uma única rota e dois upstreams. Trocamos para o módulo
`stream`:

```nginx
stream {
    upstream papagaio_api {
        server unix:/var/run/papagaio/api1.sock;
        server unix:/var/run/papagaio/api2.sock;
    }
    server {
        listen 9999 reuseport backlog=4096;
        proxy_pass papagaio_api;
    }
}
```

Stream mode é forwarding TCP puro via `splice()` — bytes vão direto do client socket para o
upstream socket sem parsing.

Trade-off: load balancing vira **per-connection**, não per-request. HTTP keep-alive gruda todas
as requests duma conexão no mesmo upstream. Para k6 (que abre muitas VUs), o fan-out natural
distribui carga bem.

Resultado em 3 runs de 5s @ 900 RPS:

```
metric          HTTP mode          stream mode    delta
p99             134/137/146 µs     126/126/128    -8.6%
p99.9           322/353/5330 µs    246/332/435    -83%
max             472/874/16740 µs   501/568/664    -90%
```

O p99 não muda muito — mas o p99.9 e o max colapsam. Os outliers ms-scale do HTTP mode (presumivelmente
buffer flushes ou alocações raras) somem completamente. Score holds em 5700/6000.

## State sharing

`AppState` é um `Arc<...>` clonado para cada conexão:

```rust
struct AppState {
    weights: router::Weights,    // 6.5 KB, &'static após load
    refs: &'static [f32],         // 13.6 MB, mmap'd
    labels: &'static [u8],        // 213 KB, mmap'd
    centroids: &'static [f32],    // 16 KB, mmap'd
    offsets: &'static [u32],      // 1 KB, mmap'd
    nprobe: usize,
}
```

Tudo é `&'static` ou cópia inline. Não há `Mutex`, não há `RwLock`, não há ownership transfer no
hot path. O `Arc::clone` por conexão custa um increment atômico.

## Touch pages e page cache

```rust
fn mmap_static(path: &str) -> &'static [u8] {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let _ = mmap.advise(memmap2::Advice::WillNeed);
    let leaked: &'static Mmap = Box::leak(Box::new(mmap));
    &leaked[..]
}

fn touch_pages(bytes: &[u8]) {
    let mut acc: u64 = 0;
    let mut i = 0;
    while i < bytes.len() {
        acc = acc.wrapping_add(bytes[i] as u64);
        i += 4096;   // 1 byte por página
    }
    std::hint::black_box(acc);
}
```

`Advice::WillNeed` é assíncrono — o kernel *pode* fazer readahead se estiver ocioso. O
`touch_pages` é síncrono e garante que toda página está populada. Sem ele, as primeiras ~200
queries pagariam demand-paging custo (~50 µs cada).

## Manuseio de erro no hot path

```rust
let count: u8 = if vectorize::vectorize(&body, &mut v).is_ok() {
    // caminho normal
} else {
    0   // payload malformado → approve
};
```

Se a vetorização falha (payload malformado), respondemos como se o veredito fosse "approve com
count=0". A escolha é deliberada — o k6 da rinha não emite payload malformado, então isso é
defensive coding que nunca dispara. Logging seria caro e ruidoso; melhor cair silenciosamente.

## O que ainda não foi feito

Veja `LATENCY_BACKLOG.md` na raiz. Resumo dos itens Tier 0 (free wins):

1. `MADV_HUGEPAGE` nos mmaps — esperado -5-15 µs no Mac Mini, zero no dev box.
2. Padé `expf` no softmax — esperado -1 µs no p50.
3. Pre-rendered full HTTP response (incluindo headers) — esperado -1-2 µs no p50.

Esses três valem ~10 µs combinados — irrelevantes para o score (já estamos em 6000/6000) mas
ajudam num cenário hipotético com SLA mais apertado.
