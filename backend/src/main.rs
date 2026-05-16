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
use std::convert::Infallible;
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
    refs: Vec<f32>,
    labels: Vec<u8>,
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
            let count: u8 = if vectorize::vectorize(&body, &mut v).is_ok() {
                let probs = router::infer(&state.weights, &v);
                if probs[2] > 0.5 {
                    unsafe { slow_path::brute_k5(&v, &state.refs, &state.labels) }
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

fn load_state() -> AppState {
    let refs_path = std::env::var("REFS_PATH").unwrap_or_else(|_| "data/box_b_refs.bin".to_string());
    let labels_path =
        std::env::var("LABELS_PATH").unwrap_or_else(|_| "data/box_b_labels.bin".to_string());
    let weights_path =
        std::env::var("WEIGHTS_PATH").unwrap_or_else(|_| "data/router_weights.bin".to_string());

    eprintln!("[papagaio] loading {refs_path}");
    let refs_bytes =
        std::fs::read(&refs_path).unwrap_or_else(|e| panic!("read {refs_path}: {e}"));
    let refs: Vec<f32> = bytemuck::cast_slice(&refs_bytes).to_vec();

    eprintln!("[papagaio] loading {labels_path}");
    let labels =
        std::fs::read(&labels_path).unwrap_or_else(|e| panic!("read {labels_path}: {e}"));

    eprintln!("[papagaio] loading {weights_path}");
    let weights_bytes =
        std::fs::read(&weights_path).unwrap_or_else(|e| panic!("read {weights_path}: {e}"));
    let weights = router::load_weights(&weights_bytes);

    assert_eq!(refs.len(), labels.len() * 16, "refs/labels size mismatch");

    eprintln!(
        "[papagaio] loaded {} Box-B refs ({:.2} MB) + {} weight floats",
        labels.len(),
        refs_bytes.len() as f64 / 1e6,
        weights_bytes.len() / 4,
    );

    AppState { weights, refs, labels }
}
