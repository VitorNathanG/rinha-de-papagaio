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

.PHONY: help check-data build up down logs status test smoke clean rebuild image-size deploy issue

help:
	@echo "Targets:"
	@echo "  build      — build da imagem monolítica papagaio-api:latest"
	@echo "  up         — sobe lb + 2 réplicas (detached)"
	@echo "  down       — derruba o stack"
	@echo "  logs       — segue os logs do stack"
	@echo "  status     — docker compose ps"
	@echo "  test       — k6 oficial da rinha (test.js, 2 min ramp até 900 RPS)"
	@echo "  smoke      — k6 smoke.js do repo da rinha"
	@echo "  rebuild    — down + build --no-cache + up"
	@echo "  deploy     — build + tag + push pro REGISTRY_IMAGE (Docker Hub)"
	@echo "  issue      — abre issue rinha/test no upstream (dispara preview test)"
	@echo "  clean      — down + remove imagem"
	@echo "  image-size — tamanho final da imagem"

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

# Teste oficial: usa o test.js do repo irmão da rinha, lendo test-data.json
# dali mesmo. A stack precisa estar de pé (target depende de `up`).
test: up
	@echo "[papagaio] rodando k6 oficial ($(RINHA_REPO)/test/test.js)..."
	cd $(RINHA_REPO) && K6_NO_USAGE_REPORT=true $(K6) run test/test.js
	@echo
	@echo "=== resultado (test/results.json) ==="
	@cat $(RINHA_REPO)/test/results.json | jq .

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
