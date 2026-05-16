"""
Export the Box-B subset to Rust-friendly raw binary files.

Layout:
    data/box_b_refs.bin    f32, shape (N_B, 16). Last 2 floats per row are zero
                           padding so each row is exactly one 64-byte cache line.
    data/box_b_labels.bin  u8,  shape (N_B,). 0 = legit, 1 = fraud.
"""
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"

B = 2


def main():
    refs = np.load(DATA / "references.npy")           # (3M, 14) float32
    labels = np.load(DATA / "labels.npy")             # bool, True = fraud
    box = np.load(DATA / "box_labels.npy")            # uint8 0/1/2

    B_idx = np.where(box == B)[0]
    n_b = len(B_idx)

    padded = np.zeros((n_b, 16), dtype=np.float32)
    padded[:, :14] = refs[B_idx]
    fraud = labels[B_idx].astype(np.uint8)

    refs_path = DATA / "box_b_refs.bin"
    labels_path = DATA / "box_b_labels.bin"
    padded.tofile(refs_path)
    fraud.tofile(labels_path)

    print(f"[export] N_B = {n_b:,}  (fraud rate {fraud.mean()*100:.2f}%)")
    print(f"[export] wrote {refs_path}   {padded.nbytes/1e6:.2f} MB")
    print(f"[export] wrote {labels_path}   {fraud.nbytes/1e3:.2f} KB")


if __name__ == "__main__":
    main()
