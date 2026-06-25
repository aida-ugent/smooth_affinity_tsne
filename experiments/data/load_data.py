import os

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def load_mouse_data(pickle_path, data_dir=None, return_highdim=False):
    """
    Load preprocessed mouse cortex data from a pickle file.

    Parameters
    ----------
    pickle_path : str
        Path to tasic2018.pickle.
    data_dir : str, optional
        Directory containing importantGenesTasic2018.npy.
    return_highdim : bool, optional
        If True, also return the mean-centred log-CPM matrix (n × n_genes)
        BEFORE the SVD step.  This is the correct high-dimensional space for
        neighbourhood evaluation (NH@k, trustworthiness, Spearman).  The PCA-50
        output should only be used as input to t-SNE optimisation.

    Returns
    -------
    X_pca : ndarray, shape (n, 50)
    X_high : ndarray, shape (n, n_genes)  – only when return_highdim=True
    y, labels, colors
    """
    import pickle

    with open(pickle_path, "rb") as f:
        tasic2018 = pickle.load(f)

    # Gene selection mask
    genes_path = None
    if data_dir is not None:
        genes_path = os.path.join(data_dir, "importantGenesTasic2018.npy")

    if genes_path is not None and os.path.exists(genes_path):
        importantGenes = np.load(genes_path)
    else:
        import sys
        # rnaseqTools.py lives alongside this file (in data/); ensure it is
        # importable regardless of how load_data was invoked.
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        if _this_dir not in sys.path:
            sys.path.insert(0, _this_dir)
        import rnaseqTools
        markerGenes = [
            "Snap25", "Gad1", "Slc17a7", "Pvalb", "Sst", "Vip", "Aqp4",
            "Mog", "Itgam", "Pdgfra", "Flt1", "Bgn", "Rorb", "Foxp2",
        ]
        importantGenes = rnaseqTools.geneSelection(
            tasic2018["counts"], n=3000, threshold=32,
            markers=markerGenes, genes=tasic2018["genes"],
        )
        if genes_path is not None:
            np.save(genes_path, importantGenes)

    counts = tasic2018["counts"]
    librarySizes = np.asarray(counts.sum(axis=1)).ravel()
    counts_subset = counts[:, importantGenes]
    if hasattr(counts_subset, "toarray"):
        counts_subset = counts_subset.toarray()
    X = np.log2(counts_subset / librarySizes[:, None] * 1e6 + 1)

    X = np.array(X)

    # Remove cells that have zero library size (→ Inf/NaN in log-CPM),
    # then remove any remaining NaN/Inf rows that would segfault openTSNE.
    finite_mask = np.isfinite(X).all(axis=1)
    n_bad = (~finite_mask).sum()
    if n_bad > 0:
        import warnings
        warnings.warn(
            f"load_mouse_data: dropping {n_bad} cells with NaN/Inf log-CPM values.",
            RuntimeWarning, stacklevel=2,
        )
        X      = X[finite_mask]
        y_raw  = tasic2018["clusters"][finite_mask]
    else:
        y_raw = tasic2018["clusters"]

    X = X - X.mean(axis=0)
    X_high = X.astype(np.float32)      # mean-centred log-CPM, kept for metrics

    U, s, V = np.linalg.svd(X, full_matrices=False)
    U[:, np.sum(V, axis=1) < 0] *= -1
    X = np.dot(U, np.diag(s))
    X = X[:, np.argsort(s)[::-1]][:, :50]

    y      = y_raw
    labels = tasic2018["clusterNames"][y]
    colors = tasic2018["clusterColors"][y]

    if return_highdim:
        return X, X_high, y, labels, colors
    return X, y, labels, colors


def load_mnist_data(n_pca=50, random_state=42):
    """
    Load MNIST (70 000 × 784), normalise to [0, 1], reduce to `n_pca` dims.

    fetch_openml returns the full dataset (60k train + 10k test = 70k); the
    whole 70k is embedded, matching standard t-SNE practice.

    Returns
    -------
    X_pca : ndarray, shape (70000, n_pca)
    y     : ndarray of str, shape (70000,)
    X_raw : ndarray, shape (70000, 784)  – raw pixel values / 255
    """
    ds = fetch_openml("mnist_784", version=1, as_frame=False)
    X_raw = ds.data.astype(np.float64) / 255.0
    y = ds.target.astype(str)

    pca = PCA(n_components=n_pca, random_state=random_state)
    X_pca = pca.fit_transform(X_raw)

    return X_pca, y, X_raw


def load_coil20_data(n_pca=50, random_state=42):
    """
    Load COIL-20 (1440 × 1024) from OpenML, normalise to [0, 1], reduce to
    `n_pca` dims.

    COIL-20 is 20 objects photographed at 72 rotations each (1440 images,
    32×32 = 1024 grayscale pixels).  Each object forms a closed 1-D loop in
    pixel space, which makes it a classic structure-preservation benchmark
    for DR methods.

    Returns
    -------
    X_pca : ndarray, shape (1440, n_pca)
    y     : ndarray of int, shape (1440,)  – object id 1..20
    X_raw : ndarray, shape (1440, 1024)    – pixel values / 255
    """
    ds = fetch_openml("COIL-20", version=1, as_frame=False)
    X_raw = ds.data.astype(np.float64) / 255.0
    y = ds.target.astype(int)

    n_comp = min(n_pca, X_raw.shape[1], X_raw.shape[0])
    pca = PCA(n_components=n_comp, random_state=random_state)
    X_pca = pca.fit_transform(X_raw)

    return X_pca, y, X_raw


def load_fashion_mnist_data(n_pca=50, random_state=42):
    """
    Load Fashion-MNIST (70 000 × 784) from OpenML, normalise to [0, 1], reduce
    to `n_pca` dims.  Same shape as MNIST but with overlapping/ambiguous classes
    (e.g. shirt vs. coat vs. pullover), so it probes the fuzzy-cluster regime.

    Returns
    -------
    X_pca : ndarray, shape (70000, n_pca)
    y     : ndarray of str, shape (70000,)  – class id "0".."9"
    X_raw : ndarray, shape (70000, 784)     – pixel values / 255
    """
    ds = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
    X_raw = ds.data.astype(np.float64) / 255.0
    y = ds.target.astype(str)

    pca = PCA(n_components=n_pca, random_state=random_state)
    X_pca = pca.fit_transform(X_raw)

    return X_pca, y, X_raw


def load_swiss_roll_data(n_samples=8000, noise=0.05, n_bins=10, random_state=42):
    """
    Generate the Swiss-roll manifold: a 2-D sheet rolled into 3-D.

    Unlike the clustered datasets this has *continuous* manifold structure with
    a known intrinsic coordinate `t` (position along the roll), so it is the
    canonical test of whether an embedding preserves *global* geometry (t-SNE
    notoriously tears the roll).  `t` is binned into `n_bins` ordered bands for
    a sequential colouring — a correct unrolling shows a clean colour gradient.

    Returns
    -------
    X : ndarray, shape (n_samples, 3)        – ambient coordinates
    y : ndarray of int, shape (n_samples,)   – ordered position band 0..n_bins-1
    t : ndarray, shape (n_samples,)          – continuous manifold position
    """
    from sklearn.datasets import make_swiss_roll
    X, t = make_swiss_roll(n_samples=n_samples, noise=noise,
                           random_state=random_state)
    X = X.astype(np.float64)
    # Equal-frequency bands along the manifold position.
    edges = np.quantile(t, np.linspace(0, 1, n_bins + 1)[1:-1])
    y = np.digitize(t, edges).astype(int)
    return X, y, t


def load_pbmc3k_data(h5ad_path, n_pca=50):
    """
    Load the preprocessed PBMC 3k scRNA-seq dataset (human peripheral-blood
    immune cells) from an AnnData .h5ad file.  A second single-cell dataset
    (different organism/tissue than the mouse cortex) for within-modality
    robustness checks.

    Uses the precomputed `obsm["X_pca"]` (already the scanpy-standard
    log-normalised + scaled PCA) and the `obs["louvain"]` cell-type labels.

    Returns
    -------
    X_pca : ndarray, shape (n_cells, n_pca)
    y     : ndarray of str, shape (n_cells,)  – cell-type name
    """
    import anndata as ad
    a = ad.read_h5ad(h5ad_path)
    X_pca = np.asarray(a.obsm["X_pca"])[:, :n_pca].astype(np.float64)
    y = a.obs["louvain"].astype(str).to_numpy()
    return X_pca, y


def load_adult_data(max_rows=10000, random_state=42):
    """
    Load and preprocess the Adult (census-income) dataset.

    Returns
    -------
    X : ndarray, shape (max_rows, n_features)
    y : ndarray of int, shape (max_rows,)  – binary income label
    """
    adult = fetch_openml(name="adult", version=2, as_frame=True)
    X_df = adult.data.copy()
    y_raw = adult.target.astype(str).str.strip()
    y = (y_raw == ">50K").astype(int).to_numpy()

    if max_rows is not None and len(X_df) > max_rows:
        rng = np.random.default_rng(random_state)
        sel = np.sort(rng.choice(len(X_df), size=max_rows, replace=False))
        X_df = X_df.iloc[sel].reset_index(drop=True)
        y = y[sel]

    numeric_cols = X_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X_df.columns if c not in numeric_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ])

    X = pre.fit_transform(X_df).astype(np.float32)
    return X, y
