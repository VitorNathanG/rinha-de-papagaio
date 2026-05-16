# --- Build stage: cache deps separately from sources -----------------------
FROM rust:1.93-slim AS builder

WORKDIR /build

# Copy manifests + .cargo config first so the layer caches when only src changes
COPY Cargo.toml ./
COPY .cargo ./.cargo

# Pre-fetch deps with a dummy src tree
RUN mkdir -p src && echo "fn main() {}" > src/main.rs \
 && cargo build --release \
 && rm -rf src target/release/deps/papagaio_api*

COPY src ./src
RUN cargo build --release

# --- Runtime stage: small debian image, no extras --------------------------
FROM debian:trixie-slim

WORKDIR /app
COPY --from=builder /build/target/release/papagaio-api /usr/local/bin/papagaio-api

# Data files are mounted via docker-compose volume.
EXPOSE 9999
CMD ["/usr/local/bin/papagaio-api"]
