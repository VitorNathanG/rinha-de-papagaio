"""
Export the trained Router weights to raw f32 binary for the Rust backend.

Layout (all f32, little-endian, no header):
    w1: (64, 14)   = 896 floats  (row-major: out × in)
    b1: (64,)      = 64 floats
    w2: (64, 64)   = 4096 floats
    b2: (64,)      = 64 floats
    w3: (3, 64)    = 192 floats
    b3: (3,)       = 3 floats
Total: 5315 floats = 21260 bytes.

The Rust router loader assumes hidden=64 depth=2 (i.e., 3 Linear layers).
If you retrain with different sizes you must update both sides — see the
H/D_IN/D_OUT constants in backend/src/router.rs and router_bench/src/main.rs.
"""
import pipeline_log

import numpy as np
import torch
from pathlib import Path

from train_router import Router

ROOT = Path(__file__).parent
DATA = ROOT / "data"


def main():
    ckpt = torch.load(DATA / "router.pt", weights_only=True)
    hidden = ckpt["hidden"]
    depth = ckpt["depth"]
    if hidden != 64 or depth != 2:
        raise SystemExit(
            f"Rust backend assumes hidden=64 depth=2, found hidden={hidden} depth={depth}"
        )

    model = Router(hidden=hidden, depth=depth)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    parts = []
    for layer in model.net:
        if isinstance(layer, torch.nn.Linear):
            parts.append(layer.weight.detach().cpu().numpy().astype(np.float32))
            parts.append(layer.bias.detach().cpu().numpy().astype(np.float32))

    flat = np.concatenate([p.flatten() for p in parts])
    out_path = DATA / "router_weights.bin"
    flat.tofile(out_path)

    n_params = sum(p.numel() for p in model.parameters())
    expected = 5315
    print(f"[export] wrote {out_path}")
    print(f"[export] floats: {flat.size:,}  bytes: {flat.nbytes:,}")
    print(f"[export] model params: {n_params:,}  expected: {expected:,}")
    if flat.size != expected:
        raise SystemExit(f"size mismatch: got {flat.size}, expected {expected}")


if __name__ == "__main__":
    pipeline_log.setup(__file__)
    main()
