"""
Export the trained Router weights to raw f32 binary for the Rust backend.

Layout (all f32, little-endian, no header):
    w1: (32, 14)   = 448 floats  (row-major: out × in)
    b1: (32,)      = 32 floats
    w2: (32, 32)   = 1024 floats
    b2: (32,)      = 32 floats
    w3: (3, 32)    = 96 floats
    b3: (3,)       = 3 floats
Total: 1635 floats = 6540 bytes.

The Rust router loader assumes hidden=32 depth=2 (i.e., 3 Linear layers).
If you retrain with different sizes you must update both sides.
"""
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
    if hidden != 32 or depth != 2:
        raise SystemExit(
            f"Rust backend assumes hidden=32 depth=2, found hidden={hidden} depth={depth}"
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
    expected = 1635
    print(f"[export] wrote {out_path}")
    print(f"[export] floats: {flat.size:,}  bytes: {flat.nbytes:,}")
    print(f"[export] model params: {n_params:,}  expected: {expected:,}")
    if flat.size != expected:
        raise SystemExit(f"size mismatch: got {flat.size}, expected {expected}")


if __name__ == "__main__":
    main()
