#!/usr/bin/env python
"""
baselines.py  —  external dimensionality-reduction methods for comparison
=========================================================================

Each ``run_*`` function takes a feature matrix ``X`` (n, d) and a
``random_state`` and returns a 2-D embedding ``Y`` (n, 2) as a float64
numpy array, so the curriculum-γ t-SNE in ``exp.py`` can be compared against
them with identical metrics on identical data.

Methods
-------
  * PaCMAP   — pacmap (the method that inspired the 3-stage idea)
  * UMAP     — umap-learn
  * TriMap   — trimap
  * AE       — a plain fully-connected autoencoder with a 2-D bottleneck
               (PyTorch, CPU); the bottleneck activations are the embedding.

Library availability is detected at import time; ``available_methods()``
returns only the methods whose dependencies are installed, so ``exp.py``
degrades gracefully if one is missing.
"""

import numpy as np


# =============================================================================
# Manifold-learning baselines
# =============================================================================

def run_pacmap(X, random_state=42, n_jobs=-1):
    """PaCMAP with default neighbor/mid-near/further-pair ratios.

    Our X is already PCA-50, so PaCMAP's internal PCA-to-100 step is a no-op;
    every method therefore sees the same input representation as t-SNE.
    """
    import pacmap
    X = np.ascontiguousarray(X, dtype=np.float32)
    reducer = pacmap.PaCMAP(n_components=2, random_state=random_state,
                            apply_pca=True)
    Y = reducer.fit_transform(X, init="pca")
    return np.asarray(Y, dtype=np.float64)


def run_umap(X, random_state=42, n_jobs=-1):
    """UMAP with library defaults (n_neighbors=15, min_dist=0.1)."""
    import umap
    X = np.ascontiguousarray(X, dtype=np.float32)
    # n_jobs>1 forces UMAP to drop the random_state for reproducibility; keep
    # it single-threaded so runs are deterministic per seed.
    reducer = umap.UMAP(n_components=2, random_state=random_state)
    Y = reducer.fit_transform(X)
    return np.asarray(Y, dtype=np.float64)


def run_trimap(X, random_state=42, n_jobs=-1):
    """TriMap with library defaults."""
    import trimap
    X = np.ascontiguousarray(X, dtype=np.float32)
    # TRIMAP's constructor accepts no random_state; it seeds numpy internally.
    np.random.seed(random_state)
    Y = trimap.TRIMAP(verbose=False).fit_transform(X)
    return np.asarray(Y, dtype=np.float64)


# =============================================================================
# Autoencoder baseline (PyTorch, CPU)
# =============================================================================

def run_autoencoder(X, random_state=42, epochs=150, batch_size=256,
                     lr=1e-3, hidden=(256, 128), n_jobs=-1):
    """Fully-connected autoencoder; the 2-D bottleneck is the embedding.

    Architecture: d -> 256 -> 128 -> 2 (bottleneck) -> 128 -> 256 -> d, ReLU
    activations, MSE reconstruction loss, Adam. Inputs are standardized first.
    This is a deliberately standard, un-tuned AE so it represents the
    "vanilla deep DR" baseline rather than a state-of-the-art one.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(random_state)
    np.random.seed(random_state)
    torch.set_num_threads(max(1, (os_cpu() if n_jobs == -1 else n_jobs)))

    X = np.asarray(X, dtype=np.float32)
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-8
    Xn = (X - mu) / sd
    d = Xn.shape[1]
    h1, h2 = hidden

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(d, h1), nn.ReLU(),
                nn.Linear(h1, h2), nn.ReLU(),
                nn.Linear(h2, 2),
            )
            self.dec = nn.Sequential(
                nn.Linear(2, h2), nn.ReLU(),
                nn.Linear(h2, h1), nn.ReLU(),
                nn.Linear(h1, d),
            )

        def forward(self, x):
            z = self.enc(x)
            return self.dec(z), z

    model = AE()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    data = torch.from_numpy(Xn)
    n = data.shape[0]
    g = torch.Generator().manual_seed(random_state)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = data[idx]
            recon, _ = model(xb)
            loss = loss_fn(recon, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        _, z = model(data)
    return z.numpy().astype(np.float64)


def os_cpu():
    import os
    return os.cpu_count() or 1


# =============================================================================
# Registry
# =============================================================================

_ALL = {
    "pacmap":      run_pacmap,
    "umap":        run_umap,
    "trimap":      run_trimap,
    "autoencoder": run_autoencoder,
}

_IMPORT_NAME = {
    "pacmap": "pacmap", "umap": "umap", "trimap": "trimap",
    "autoencoder": "torch",
}


def available_methods():
    """Return {name: fn} for every baseline whose dependency is importable."""
    import importlib
    out = {}
    for name, fn in _ALL.items():
        try:
            importlib.import_module(_IMPORT_NAME[name])
            out[name] = fn
        except Exception:
            pass
    return out
