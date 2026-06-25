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
  * VAE      — a variational autoencoder with a 2-D latent (PyTorch, CPU);
               the posterior mean of the latent is the embedding.

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
# Variational autoencoder baseline (PyTorch, CPU)
# =============================================================================

def run_vae(X, random_state=42, epochs=150, batch_size=256, lr=1e-3,
            hidden=(256, 128), beta=1.0, n_jobs=-1):
    """Variational autoencoder; the 2-D posterior mean is the embedding.

    Encoder  : d -> 256 -> 128 -> (mu[2], logvar[2])
    Latent   : reparameterized sample z = mu + eps * exp(0.5 * logvar)
    Decoder  : 2 -> 128 -> 256 -> d
    Objective: ELBO = MSE reconstruction (summed over features, averaged over
               the batch) + ``beta`` * KL(q(z|x) || N(0, I)).  Inputs are
               standardized first.

    Unlike a plain autoencoder, the unit-Gaussian prior + KL term regularize the
    latent space, so the 2-D code is a smooth, structured embedding rather than
    just whatever compresses best for reconstruction.  The posterior mean ``mu``
    is returned as the embedding (standard practice for VAE visualization).
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(random_state)
    np.random.seed(random_state)
    torch.set_num_threads(max(1, (os_cpu() if n_jobs == -1 else n_jobs)))

    X = np.asarray(X, dtype=np.float32)
    mu_, sd_ = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-8
    Xn = (X - mu_) / sd_
    d = Xn.shape[1]
    h1, h2 = hidden

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(d, h1), nn.ReLU(),
                nn.Linear(h1, h2), nn.ReLU(),
            )
            self.fc_mu = nn.Linear(h2, 2)
            self.fc_logvar = nn.Linear(h2, 2)
            self.dec = nn.Sequential(
                nn.Linear(2, h2), nn.ReLU(),
                nn.Linear(h2, h1), nn.ReLU(),
                nn.Linear(h1, d),
            )

        def encode(self, x):
            h = self.enc(x)
            return self.fc_mu(h), self.fc_logvar(h)

        def forward(self, x):
            mu, logvar = self.encode(x)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            return self.dec(z), mu, logvar

    model = VAE()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    data = torch.from_numpy(Xn)
    n = data.shape[0]
    g = torch.Generator().manual_seed(random_state)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch_size):
            xb = data[perm[i:i + batch_size]]
            recon, mu, logvar = model(xb)
            nb = xb.shape[0]
            rec = F.mse_loss(recon, xb, reduction="sum") / nb
            kl = -0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp()) / nb
            loss = rec + beta * kl
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(data)
    return mu.numpy().astype(np.float64)


def os_cpu():
    import os
    return os.cpu_count() or 1


# =============================================================================
# Registry
# =============================================================================

_ALL = {
    "pacmap": run_pacmap,
    "umap":   run_umap,
    "trimap": run_trimap,
    "vae":    run_vae,
}

_IMPORT_NAME = {
    "pacmap": "pacmap", "umap": "umap", "trimap": "trimap",
    "vae": "torch",
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
