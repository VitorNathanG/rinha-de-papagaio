"""
pipeline_log — redireciona stdout/stderr dos scripts do pipeline offline para
log/<nome do script>/<timestamp ISO 8601 GMT-3>.log com line-buffering.

Cada invocação produz um arquivo novo (timestamp do startup do script). O
redirecionamento é feito a nível de FD (`os.dup2`) para também capturar prints
de extensões C (torch, numpy nativo) e de subprocessos herdados.

Uso (precisa ser a primeira coisa que o script faz, antes de qualquer import
que possa imprimir — torch em particular):

    import pipeline_log
    pipeline_log.setup(__file__)

    # ... resto do script ...
"""
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# GMT-3 fixo (Brasília não usa DST desde 2019).
_BRT = timezone(timedelta(hours=-3))

# Base dir é o diretório do próprio pipeline_log.py — todos os scripts vivem ao
# lado dele, então isso é o root do repo.
_BASE = Path(__file__).resolve().parent


def setup(script_path: str) -> Path:
    """
    Cria log/<script>/<timestamp>.log, redireciona stdout+stderr (e FDs 1/2) e
    devolve o Path. Imprime no terminal (stderr original) o caminho do log
    antes de redirecionar — assim o operador consegue rodar `tail -F` em outro
    terminal.
    """
    script_name = Path(script_path).stem
    log_dir = _BASE / "log" / script_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # ISO 8601 basic (sem `:` no nome do arquivo). `%z` rende `-0300`.
    ts = datetime.now(_BRT).strftime("%Y-%m-%dT%H-%M-%S%z")
    log_path = log_dir / f"{ts}.log"

    # Avisa antes do redirect — depois disso a única forma de ver é via tail.
    sys.stderr.write(f"[pipeline_log] {script_name} → {log_path}\n")
    sys.stderr.flush()

    # buffering=1 → line-buffered (modo texto). Cada print() faz flush.
    fh = open(log_path, "w", buffering=1, encoding="utf-8")
    # Redireciona a nível Python.
    sys.stdout = fh
    sys.stderr = fh
    # E a nível de FD (cobre C extensions e subprocessos herdados).
    os.dup2(fh.fileno(), 1)
    os.dup2(fh.fileno(), 2)

    # Header pra fazer o arquivo auto-explicativo quando alguém abrir depois.
    print(f"[pipeline_log] script={script_name} started_at={ts} pid={os.getpid()}")
    return log_path


if __name__ == "__main__":
    p = setup(__file__)
    print("smoke: hello from log file")
    print(f"smoke: file is {p}")
