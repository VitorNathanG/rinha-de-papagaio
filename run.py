"""
End-to-end "papagaio" feasibility study for Rinha de Backend 2026.

Question being answered:
    If, instead of running 5-NN over 3M reference vectors at request time,
    we trained a tiny MLP that imitates the 5-NN's verdict — how good
    would the offline rinha "detection score" be?

The four steps:
    1. prepare.py   decompress references.json.gz to numpy
    2. label.py     leave-one-out 5-NN on GPU -> fraud_counts.npy (the target)
    3. train.py     train a small MLP to predict that count -> model.pt
    4. evaluate.py  confusion matrix + simulated score_det on the test split

Quick smoke test (CPU or low-mem GPU):
    SUBSET=100000 BATCH=128 python run.py

Full run (assumes ROCm/CUDA GPU with >= 8 GB VRAM):
    python run.py

Useful env vars are documented at the top of each script.

Install (ROCm):
    pip install -r requirements.txt
    pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
"""
import prepare
import label
import train
import evaluate

if __name__ == "__main__":
    prepare.main()
    print()
    label.main()
    print()
    train.main()
    print()
    evaluate.main()
