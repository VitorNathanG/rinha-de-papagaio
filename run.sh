#!/usr/bin/env bash
# End-to-end launcher for the papagaio feasibility study.
#
# Uses uv (https://docs.astral.sh/uv/) for dependency management and venv:
#   - installs uv if it is missing
#   - `uv sync` creates .venv and installs the locked deps (incl. torch+rocm)
#   - runs prepare -> label -> train -> evaluate via `uv run`
#   - mirrors all output to a timestamped log file under logs/
#
# Useful env vars (forwarded to the python steps):
#   SUBSET=100000   smoke-test only the first 100k queries during labeling
#   BATCH=128       drop GPU batch size if you run out of VRAM
#   HIDDEN=128      widen the MLP
#   EPOCHS=20       train longer
#
# Examples:
#   ./run.sh                            # full study
#   SUBSET=100000 BATCH=128 ./run.sh    # 5-min smoke test

set -euo pipefail

cd "$(dirname "$0")"

# ---- 1. install uv if missing ----------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "[run.sh] installing uv (Astral)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The installer drops uv into ~/.local/bin; make sure this shell sees it.
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "[run.sh] uv: $(uv --version)"

# ---- 2. sync deps from pyproject.toml / uv.lock ----------------------------
# uv sync handles: pick Python, create .venv, install pinned deps. Cached.
# If you ever need to refresh: `uv lock --upgrade && uv sync`.
echo "[run.sh] syncing dependencies (.venv + torch+rocm + numpy)"
uv sync

# ---- 3. sanity check the GPU -----------------------------------------------
uv run python - <<'PY'
import torch
if torch.cuda.is_available():
    try:
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[run.sh] torch ready on GPU: {name} ({mem:.1f} GB)")
    except Exception as e:
        print(f"[run.sh] torch.cuda available, info unavailable: {e}")
else:
    print("[run.sh] WARNING: torch will run on CPU — label step will be very slow")
    print("[run.sh] Tip: rerun with SUBSET=100000 to keep it under a few minutes")
PY

# ---- 4. run the pipeline ---------------------------------------------------
mkdir -p logs
LOG="logs/run-$(date +%Y%m%d-%H%M%S).log"
echo "[run.sh] mirroring all output to $LOG"
echo "[run.sh] you can follow progress here, OR in another terminal:  tail -f $LOG"
echo

# PYTHONUNBUFFERED=1 makes prints flush immediately, so the log is live.
PYTHONUNBUFFERED=1 uv run python run.py 2>&1 | tee "$LOG"

echo
echo "[run.sh] done."
echo "[run.sh] results table:  data/results.json"
echo "[run.sh] full log:        $LOG"
