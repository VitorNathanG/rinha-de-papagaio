//! papagaio-api — minimum-latency HTTP backend for Rinha de Backend 2026.
//!
//! No axum, no JSON crate. Path matching is a manual `match (method, uri)`
//! switch on every connection; responses are pre-rendered `&'static [u8]`
//! wrapped in `Bytes::from_static` (zero copy, zero alloc).
//!
//! Tokio runtime defaults to `current_thread` so each container replica
//! runs on a single OS thread (no cross-worker scheduling cost inside a
//! 0.45-CPU container). Override with TOKIO_WORKERS=N for multi-threaded
//! profiling on a beefy dev box.
//!
//! TCP_NODELAY is enabled per accepted connection so 35-byte responses go
//! out immediately instead of waiting for Nagle's algorithm to buffer them.
//!
//! BIND_ADDR may be either `host:port` (TCP) or a path starting with `/`
//! (UNIX domain socket). UNIX sockets skip the kernel TCP stack — useful
//! when nginx and the backend share a container volume.

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::header::{HeaderValue, CONTENT_TYPE};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use memmap2::Mmap;
use std::convert::Infallible;
use std::fs::File;
use std::net::SocketAddr;
use std::os::unix::fs::PermissionsExt;
use std::sync::Arc;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::{TcpListener, UnixListener};

mod router;
mod slow_path;
mod vectorize;

/// Indexed by fraud count 0..5 among 5-NN. approved iff count < 3.
const RESPONSES: [&[u8]; 6] = [
    b"{\"approved\":true,\"fraud_score\":0.0}",
    b"{\"approved\":true,\"fraud_score\":0.2}",
    b"{\"approved\":true,\"fraud_score\":0.4}",
    b"{\"approved\":false,\"fraud_score\":0.6}",
    b"{\"approved\":false,\"fraud_score\":0.8}",
    b"{\"approved\":false,\"fraud_score\":1.0}",
];

struct AppState {
    weights: router::Weights,
    refs: &'static [i16],
    labels: &'static [u8],
    centroids: &'static [i16],
    offsets: &'static [u32],
    nprobe: usize,
}

fn main() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        panic!("papagaio-api requires AVX2 + FMA (Haswell or newer)");
    }

    let state = Arc::new(load_state());
    let bind = std::env::var("BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:9999".to_string());

    let workers = std::env::var("TOKIO_WORKERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);

    let rt = if workers <= 1 {
        eprintln!("[papagaio] tokio: current_thread (1 worker)");
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio rt")
    } else {
        eprintln!("[papagaio] tokio: multi_thread ({workers} workers)");
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(workers)
            .enable_all()
            .build()
            .expect("tokio rt")
    };

    rt.block_on(serve(state, bind));
}

async fn serve(state: Arc<AppState>, bind: String) {
    if bind.starts_with('/') {
        // UNIX domain socket.
        let _ = std::fs::remove_file(&bind);
        let listener = UnixListener::bind(&bind)
            .unwrap_or_else(|e| panic!("UnixListener::bind({bind}): {e}"));
        // Make the socket world-RW so nginx (different uid in its container)
        // can connect.
        let _ = std::fs::set_permissions(&bind, std::fs::Permissions::from_mode(0o666));
        eprintln!("[papagaio] listening on UNIX {bind}");
        loop {
            match listener.accept().await {
                Ok((stream, _)) => spawn_conn(stream, state.clone()),
                Err(e) => eprintln!("[papagaio] accept err: {e}"),
            }
        }
    } else {
        let addr: SocketAddr = bind.parse().expect("invalid BIND_ADDR");
        let listener = TcpListener::bind(addr).await.expect("bind");
        eprintln!("[papagaio] listening on http://{addr}");
        loop {
            match listener.accept().await {
                Ok((stream, _)) => {
                    let _ = stream.set_nodelay(true);
                    spawn_conn(stream, state.clone());
                }
                Err(e) => eprintln!("[papagaio] accept err: {e}"),
            }
        }
    }
}

fn spawn_conn<S>(stream: S, state: Arc<AppState>)
where
    S: AsyncRead + AsyncWrite + Send + Unpin + 'static,
{
    let io = TokioIo::new(stream);
    tokio::spawn(async move {
        let service = service_fn(move |req| handle(state.clone(), req));
        let _ = http1::Builder::new()
            .keep_alive(true)
            .serve_connection(io, service)
            .await;
    });
}

async fn handle(
    state: Arc<AppState>,
    req: Request<Incoming>,
) -> Result<Response<Full<Bytes>>, Infallible> {
    // Manual dispatch — no router, no extractor machinery.
    match (req.method(), req.uri().path()) {
        (&Method::POST, "/fraud-score") => {
            let body = match req.into_body().collect().await {
                Ok(b) => b.to_bytes(),
                Err(_) => return Ok(json_response(RESPONSES[0])),
            };

            let mut v = [0f32; 16];
            let mut v_i16 = [0i16; 16];
            let count: u8 = if vectorize::vectorize(&body, &mut v, &mut v_i16).is_ok() {
                let probs = router::infer(&state.weights, &v);
                if probs[2] > 0.5 {
                    unsafe {
                        slow_path::ivf_k5(
                            &v_i16,
                            state.centroids,
                            state.offsets,
                            state.refs,
                            state.labels,
                            state.nprobe,
                        )
                    }
                } else if probs[0] > probs[1] {
                    0 // A-Legit
                } else {
                    5 // A-Fraud
                }
            } else {
                0
            };

            Ok(json_response(RESPONSES[count as usize]))
        }
        (&Method::GET, "/ready") => Ok(ready_response()),
        _ => {
            let mut resp = Response::new(Full::new(Bytes::new()));
            *resp.status_mut() = StatusCode::NOT_FOUND;
            Ok(resp)
        }
    }
}

#[inline]
fn json_response(body: &'static [u8]) -> Response<Full<Bytes>> {
    let mut resp = Response::new(Full::new(Bytes::from_static(body)));
    resp.headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    resp
}

#[inline]
fn ready_response() -> Response<Full<Bytes>> {
    Response::new(Full::new(Bytes::from_static(b"OK")))
}

/// mmap a file and leak the mapping so we can hand out a `'static` slice. The
/// mapping lives for the whole process anyway (AppState is held in an Arc for
/// the lifetime of the runtime), so the leak is a one-time wash — no growth.
fn mmap_static(path: &str) -> &'static [u8] {
    let file = File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let mmap = unsafe { Mmap::map(&file).unwrap_or_else(|e| panic!("mmap {path}: {e}")) };
    // Hint the kernel to start populating page table entries — speeds up the
    // first few queries that would otherwise eat synchronous page faults.
    let _ = mmap.advise(memmap2::Advice::WillNeed);
    let leaked: &'static Mmap = Box::leak(Box::new(mmap));
    &leaked[..]
}

/// Force-fault every page so the first slow query doesn't pay the demand-paging
/// cost. Sequential touch is faster than letting Advice::WillNeed do its async
/// thing and racing the first request.
fn touch_pages(bytes: &[u8]) {
    let mut acc: u64 = 0;
    let mut i = 0;
    while i < bytes.len() {
        acc = acc.wrapping_add(bytes[i] as u64);
        i += 4096;
    }
    std::hint::black_box(acc);
}

fn load_state() -> AppState {
    let refs_path =
        std::env::var("REFS_PATH").unwrap_or_else(|_| "data/box_b_refs.i16.bin".to_string());
    let labels_path =
        std::env::var("LABELS_PATH").unwrap_or_else(|_| "data/box_b_labels.bin".to_string());
    let centroids_path = std::env::var("CENTROIDS_PATH")
        .unwrap_or_else(|_| "data/box_b_ivf_centroids.i16.bin".to_string());
    let offsets_path = std::env::var("OFFSETS_PATH")
        .unwrap_or_else(|_| "data/box_b_ivf_offsets.bin".to_string());
    let weights_path =
        std::env::var("WEIGHTS_PATH").unwrap_or_else(|_| "data/router_weights.bin".to_string());
    let nprobe: usize = std::env::var("NPROBE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);

    let refs_bytes = mmap_static(&refs_path);
    let labels_bytes = mmap_static(&labels_path);
    let centroids_bytes = mmap_static(&centroids_path);
    let offsets_bytes = mmap_static(&offsets_path);

    touch_pages(refs_bytes);
    touch_pages(labels_bytes);
    touch_pages(centroids_bytes);
    touch_pages(offsets_bytes);

    let refs: &'static [i16] = bytemuck::cast_slice(refs_bytes);
    let labels: &'static [u8] = labels_bytes;
    let centroids: &'static [i16] = bytemuck::cast_slice(centroids_bytes);
    let offsets: &'static [u32] = bytemuck::cast_slice(offsets_bytes);

    let weights_bytes = std::fs::read(&weights_path)
        .unwrap_or_else(|e| panic!("read {weights_path}: {e}"));
    let weights = router::load_weights(&weights_bytes);

    let nlist = offsets.len() - 1;
    assert_eq!(refs.len(), labels.len() * 16, "refs/labels size mismatch");
    assert_eq!(centroids.len(), nlist * 16, "centroids/offsets nlist mismatch");
    assert!((1..=64).contains(&nprobe), "NPROBE must be in 1..=64");

    eprintln!(
        "[papagaio] loaded {} refs ({:.2} MB) + nlist={} centroids ({:.1} KB), nprobe={}",
        labels.len(),
        refs_bytes.len() as f64 / 1e6,
        nlist,
        centroids_bytes.len() as f64 / 1e3,
        nprobe,
    );

    AppState {
        weights,
        refs,
        labels,
        centroids,
        offsets,
        nprobe,
    }
}
