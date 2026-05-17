# Makefile — orquestra build/up/test/down da submissão papagaio.
#
# A imagem papagaio-api:latest é monolítica: traz o binário do backend e
# os 5 artefatos do índice (refs/labels/centroids/offsets do Box-B + pesos
# do router) embutidos. Nenhum bind-mount de dados é necessário.

SHELL := /bin/bash

# Caminhos dos artefatos exigidos pela imagem (kernel i16 runtime).
DATA_FILES := \
	data/box_b_refs.i16.bin \
	data/box_b_labels.bin \
	data/box_b_ivf_centroids.i16.bin \
	data/box_b_ivf_offsets.bin \
	data/router_weights.bin

# k6 oficial vive no repo irmão da rinha.
RINHA_REPO ?= ../rinha-de-backend-2026
K6        ?= $(shell command -v k6 2>/dev/null || echo /tmp/k6)
COMPOSE   ?= docker compose

IMAGE          := papagaio-api:latest
REGISTRY_IMAGE ?= vitornathan/rinha-de-papagaio:latest

# Submission no repo upstream da rinha — usado por `make issue`.
RINHA_UPSTREAM ?= zanfranceschi/rinha-de-backend-2026
SUBMISSION_ID  ?= papagaio

.PHONY: help check-data build up down logs status test smoke clean rebuild image-size deploy issue \
        artifacts artifacts-clean

help:
	@echo "Targets:"
	@echo "  artifacts       — regenera os 5 arquivos runtime do zero (pipeline offline)"
	@echo "  artifacts-clean — apaga data/* (intermediários e finais); pede confirmação"
	@echo "  build           — build da imagem monolítica papagaio-api:latest"
	@echo "  up              — sobe lb + 2 réplicas (detached)"
	@echo "  down            — derruba o stack"
	@echo "  logs            — segue os logs do stack"
	@echo "  status          — docker compose ps"
	@echo "  test            — k6 oficial da rinha (test.js, 2 min ramp até 900 RPS)"
	@echo "  smoke           — k6 smoke.js do repo da rinha"
	@echo "  rebuild         — down + build --no-cache + up"
	@echo "  deploy          — build + tag + push pro REGISTRY_IMAGE (Docker Hub)"
	@echo "  issue           — abre issue rinha/test no upstream (dispara preview test)"
	@echo "  clean           — down + remove imagem"
	@echo "  image-size      — tamanho final da imagem"

check-data:
	@missing=0; for f in $(DATA_FILES); do \
	  if [ ! -f "$$f" ]; then echo "[papagaio] FALTANDO: $$f"; missing=1; fi; \
	done; \
	if [ $$missing -ne 0 ]; then \
	  echo "[papagaio] rode o pipeline offline antes (ver CLAUDE.md)"; exit 1; \
	fi

build: check-data
	$(COMPOSE) build api1

rebuild:
	$(COMPOSE) down --remove-orphans
	$(COMPOSE) build --no-cache api1
	$(COMPOSE) up -d

up: build
	$(COMPOSE) up -d
	@echo "[papagaio] esperando o LB responder em :9999..."
	@for i in $$(seq 1 30); do \
	  if curl -fsS -o /dev/null http://localhost:9999/ready 2>/dev/null; then \
	    echo "[papagaio] pronto."; exit 0; \
	  fi; sleep 0.5; \
	done; echo "[papagaio] LB não respondeu em 15s"; \
	$(COMPOSE) logs --tail=50; exit 1

down:
	$(COMPOSE) down --remove-orphans

logs:
	$(COMPOSE) logs -f

status:
	$(COMPOSE) ps

image-size:
	@docker image inspect $(IMAGE) --format '{{.Size}}' | \
	  awk '{printf "%s: %.1f MB\n", "$(IMAGE)", $$1/1024/1024}'

# Teste: usa o test/test.js do próprio repo (fork do oficial com série
# temporal de latência adicional). O test-data.json ainda vem do repo irmão
# da rinha (caminho fixo no nosso test.js). A stack precisa estar de pé.
test: up
	@echo "[papagaio] rodando k6 local (test/test.js)..."
	K6_NO_USAGE_REPORT=true $(K6) run test/test.js
	@echo
	@echo "=== resultado (test/results.json) ==="
	@cat test/results.json | jq .

smoke: up
	cd $(RINHA_REPO) && K6_NO_USAGE_REPORT=true $(K6) run test/smoke.js

clean: down
	-docker image rm $(IMAGE) 2>/dev/null || true

# Deploy: empurra a imagem monolítica pro registry consumido pela branch
# submission. Requer `docker login` prévio (Docker Hub neste caso).
deploy: build
	docker tag $(IMAGE) $(REGISTRY_IMAGE)
	docker push $(REGISTRY_IMAGE)
	@echo "[papagaio] pushed $(REGISTRY_IMAGE)"
	@docker manifest inspect $(REGISTRY_IMAGE) 2>/dev/null | \
	  jq -r '.manifests[]? | select(.platform.architecture=="amd64") | "digest: " + .digest' \
	  || true

# Issue: dispara o preview test na rinha. O engine do zan varre issues
# abertas com `rinha/test <id>` no body, acha o repo do participante no
# participants/<github-user>.json e roda o k6 contra a branch submission.
# Requer `gh auth login` prévio.
issue:
	@command -v gh >/dev/null || { echo "[papagaio] gh CLI não instalado"; exit 1; }
	@gh auth status >/dev/null 2>&1 || { echo "[papagaio] rode 'gh auth login' antes"; exit 1; }
	gh issue create \
	  --repo $(RINHA_UPSTREAM) \
	  --title "rinha/test $(SUBMISSION_ID)" \
	  --body  "rinha/test $(SUBMISSION_ID)"

# ----------------------------------------------------------------------------
# Artifacts pipeline — regenera os 5 arquivos runtime de forma reproduzível
# ----------------------------------------------------------------------------
#
# A imagem oficial precisa exatamente destes 5 arquivos (veja DATA_FILES acima):
#   data/box_b_refs.i16.bin
#   data/box_b_labels.bin
#   data/box_b_ivf_centroids.i16.bin
#   data/box_b_ivf_offsets.bin
#   data/router_weights.bin
#
# Eles são derivados, via 8 passos sequenciais (prepare → label → partition →
# nearest_opp → border_halo → export_box_b → train_router → export_router), de
# `../rinha-de-backend-2026/resources/references.json.gz`. Cada passo gera um
# arquivo .npy / .pt intermediário em data/, e cada regra abaixo declara essas
# dependências por arquivo — `make` só re-roda o que ficou stale.
#
# Hiperparâmetros congelados (override só se souber o que tá fazendo — todos
# tem rationale documentado em CLAUDE.md / nos docstrings dos scripts):
#
# k-NN width usado por partition.py (não confundir com k=5 da rinha)
ARTIFACT_K ?= 25
# nearest-opp threshold do border halo
ARTIFACT_D ?= 0.23
# IVF clusters (calibrado: minimiza fn sem aumentar p99)
ARTIFACT_NLIST ?= 512
# safety cap do Lloyd (converge muito antes)
ARTIFACT_KMEANS_ITER ?= 1000
# k-means init + router train split/init
ARTIFACT_SEED ?= 42
# MLP hidden width (backend Rust ASSERTA 64 em compile time)
ARTIFACT_HIDDEN ?= 64
# MLP depth (backend Rust ASSERTA 2 em compile time)
ARTIFACT_DEPTH ?= 2

UV ?= uv
PY := $(UV) run python

# Caminho default do .json.gz (override via REFERENCES_GZ se tiver em outro lugar).
REFERENCES_GZ ?= ../rinha-de-backend-2026/resources/references.json.gz

ARTIFACTS_FINAL := \
    data/box_b_refs.i16.bin \
    data/box_b_labels.bin \
    data/box_b_ivf_centroids.i16.bin \
    data/box_b_ivf_offsets.bin \
    data/router_weights.bin

# ---- Step 1: descompacta o .json.gz em arrays numpy --------------------------
# prepare.py escreve os dois .npy juntos; declaramos os dois como targets do
# mesmo recipe (regra "grouped target") pra Make não rodar duas vezes.
data/references.npy data/labels.npy &: prepare.py
	REFERENCES_GZ=$(REFERENCES_GZ) $(PY) prepare.py

# ---- Step 2: leave-one-out 25-NN sobre os 3M refs (GPU) ---------------------
data/fraud_counts_k$(ARTIFACT_K).npy: label.py data/references.npy data/labels.npy
	K=$(ARTIFACT_K) $(PY) label.py

# ---- Step 3: partition em A-Legit / A-Fraud / B -----------------------------
# Output renomeado pra .before_halo.npy (pristine) — border_halo lê dele.
data/box_labels.before_halo.npy: partition.py data/labels.npy data/fraud_counts_k$(ARTIFACT_K).npy
	$(PY) partition.py

# ---- Step 4: distância pro vizinho de label oposto (GPU) --------------------
data/nearest_opp_dist.npy: nearest_opp.py data/references.npy data/labels.npy data/box_labels.before_halo.npy
	$(PY) nearest_opp.py

# ---- Step 5: border halo — promove refs A na fronteira pra B ----------------
# Lê before_halo + nearest_opp, escreve box_labels.npy limpo (sem mutação in-place).
data/box_labels.npy: border_halo.py data/box_labels.before_halo.npy data/nearest_opp_dist.npy
	D=$(ARTIFACT_D) $(PY) border_halo.py

# ---- Step 6: k-means IVF + dump dos refs/labels/centroides/offsets ----------
# Um recipe escreve 6 arquivos (5 do runtime + 1 mirror f32 dos refs).
# Mantemos só os runtime aqui; o mirror box_b_refs.bin é produzido junto e é
# co-target. ITER é safety cap, não meta — convergência sai antes.
data/box_b_refs.i16.bin data/box_b_labels.bin \
data/box_b_ivf_centroids.i16.bin data/box_b_ivf_offsets.bin \
data/box_b_refs.bin data/box_b_ivf_centroids.bin &: export_box_b.py data/references.npy data/labels.npy data/box_labels.npy
	NLIST=$(ARTIFACT_NLIST) ITER=$(ARTIFACT_KMEANS_ITER) SEED=$(ARTIFACT_SEED) $(PY) export_box_b.py

# ---- Step 7: treina o MLP router (3 classes) --------------------------------
# train_router.py escreve router.pt + os 3 .npy de splits; o checkpoint
# é o único alimentando export_router, os splits são pra evaluate.
# Treina sobre box_labels.before_halo.npy (pristine, sem halo) — o halo
# só engorda o Box-B do slow path, não deve forçar o router a memorizar
# casca outward (ver docstring do train_router.py).
data/router.pt: train_router.py data/references.npy data/box_labels.before_halo.npy
	HIDDEN=$(ARTIFACT_HIDDEN) DEPTH=$(ARTIFACT_DEPTH) SEED=$(ARTIFACT_SEED) $(PY) train_router.py

# ---- Step 8: serializa os pesos do MLP em raw f32 ---------------------------
data/router_weights.bin: export_router.py data/router.pt
	$(PY) export_router.py

# Phony "produz tudo" — alvo padrão pra quem só quer os 5 finais.
artifacts: $(ARTIFACTS_FINAL)
	@echo
	@echo "[papagaio] artefatos prontos:"
	@ls -lh $(ARTIFACTS_FINAL)

# Nuke total de data/. Pede confirmação porque o pipeline completo demora horas
# (label.py em particular é o passo mais caro: ~30+ min de GPU mesmo numa RDNA4).
artifacts-clean:
	@echo "[papagaio] vai apagar tudo em data/ (intermediários + finais)."
	@echo "[papagaio] o pipeline completo de regeneração leva ~1h de GPU."
	@read -p "[papagaio] confirma? (digite 'sim'): " ans; \
	  if [ "$$ans" != "sim" ]; then echo "[papagaio] abortado."; exit 1; fi
	rm -rf data/*
	@echo "[papagaio] data/ limpo."
