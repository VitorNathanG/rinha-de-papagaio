# Build context is the repository root (see docker-compose / Makefile).
# Source lives under backend/, runtime data files under data/. Both are
# copied into the image so the runtime container is fully self-contained:
# no host bind-mounts, no external registry beyond the image itself.

# --- Build stage: cache deps separately from sources -----------------------
FROM rust:1.93-slim AS builder

WORKDIR /build

# Manifests + .cargo config first so the dep layer caches when only src changes.
COPY backend/Cargo.toml backend/Cargo.lock ./
COPY backend/.cargo ./.cargo

# Pre-fetch deps with a dummy src tree.
RUN mkdir -p src && echo "fn main() {}" > src/main.rs \
 && cargo build --release \
 && rm -rf src target/release/deps/papagaio_api*

COPY backend/src ./src
RUN cargo build --release

# --- Runtime stage: small debian image, data + binary baked in -------------
FROM debian:trixie-slim

WORKDIR /app
COPY --from=builder /build/target/release/papagaio-api /usr/local/bin/papagaio-api

# Index + weights live at /app/data — matches the binary's default paths
# (REFS_PATH=data/box_b_refs.bin, etc., resolved relative to WORKDIR).
COPY data/box_b_refs.bin \
     data/box_b_labels.bin \
     data/box_b_ivf_centroids.bin \
     data/box_b_ivf_offsets.bin \
     data/router_weights.bin \
     ./data/

EXPOSE 9999
CMD ["/usr/local/bin/papagaio-api"]
