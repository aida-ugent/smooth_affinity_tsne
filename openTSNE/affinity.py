import logging
import numbers
import operator
from functools import reduce

import numpy as np
import scipy.sparse as sp

from openTSNE import _tsne
from openTSNE import nearest_neighbors
from openTSNE import utils
from openTSNE.utils import is_package_installed

import warnings

log = logging.getLogger(__name__)


class Affinities:
    """Compute the affinities between samples.

    t-SNE takes as input an affinity matrix :math:`P`, and does not really care
    about anything else from the data. This means we can use t-SNE for any data
    where we are able to express interactions between samples with an affinity
    matrix.

    Attributes
    ----------
    P: array_like
        The :math:`N \\times N` affinity matrix expressing interactions between
        :math:`N` initial data samples.

    verbose: bool

    """

    def __init__(self, verbose=False):
        self.P = None
        self.P_conditional = None
        self.verbose = verbose
        self.knn_index: nearest_neighbors.KNNIndex = None

    def to_new(self, data, return_distances=False):
        """Compute the affinities of new samples to the initial samples.

        This is necessary for embedding new data points into an existing
        embedding.

        Parameters
        ----------
        data: np.ndarray
            The data points to be added to the existing embedding.

        return_distances: bool
            If needed, the function can return the indices of the nearest
            neighbors and their corresponding distances.

        Returns
        -------
        P: array_like
            An :math:`N \\times M` affinity matrix expressing interactions
            between :math:`N` new data points the initial :math:`M` data
            samples.

        indices: np.ndarray
            Returned if ``return_distances=True``. The indices of the :math:`k`
            nearest neighbors in the existing embedding for every new data
            point.

        distances: np.ndarray
            Returned if ``return_distances=True``. The distances to the
            :math:`k` nearest neighbors in the existing embedding for every new
            data point.

        """

    @property
    def n_samples(self):
        if self.knn_index is None:
            raise RuntimeError("`knn_index` is not set!")
        return self.knn_index.n_samples


class PerplexityBasedNN(Affinities):
    """Compute standard, Gaussian affinities using nearest neighbors.

    Please see the :ref:`parameter-guide` for more information.

    Parameters
    ----------
    data: np.ndarray
        The data matrix.

    perplexity: float
        Perplexity can be thought of as the continuous :math:`k` number of
        nearest neighbors, for which t-SNE will attempt to preserve distances.

    method: str
        Specifies the nearest neighbor method to use. Can be ``exact``, ``annoy``,
        ``pynndescent``, ``hnsw``, ``approx``, or ``auto`` (default). ``approx``
        uses Annoy if the input data matrix is not a sparse object and if Annoy
        supports the given metric. Otherwise, it uses Pynndescent. ``auto`` uses
        exact nearest neighbors for N<1000 and the same heuristic as ``approx``
        for N>=1000.

    metric: Union[str, Callable]
        The metric to be used to compute affinities between points in the
        original space.

    metric_params: dict
        Additional keyword arguments for the metric function.

    symmetrize: bool
        Symmetrize the affinity matrix. During standard t-SNE optimization, the
        affinities are symmetrized. However, when embedding new data points into
        existing embeddings, symmetrization is not performed.

    n_jobs: int
        The number of threads to use while running t-SNE. This follows the
        scikit-learn convention, ``-1`` meaning all processors, ``-2`` meaning
        all but one, etc.

    random_state: Union[int, RandomState]
        If the value is an int, random_state is the seed used by the random
        number generator. If the value is a RandomState instance, then it will
        be used as the random number generator. If the value is None, the random
        number generator is the RandomState instance used by `np.random`.

    verbose: bool

    k_neighbors: int or ``auto``
        The number of neighbors to use in the kNN graph. If ``auto`` (default),
        it is set to three times the perplexity.

    knn_kwargs: Optional[None, dict]
        Optional keyword arguments that will be passed to the ``knn_index``.

    knn_index: Optional[nearest_neighbors.KNNIndex]
        Optionally, a precomputed ``openTSNE.nearest_neighbors.KNNIndex`` object
        can be specified. This option will ignore any KNN-related parameters.
        When ``knn_index`` is specified, ``data`` must be set to None.

    """

    def __init__(
        self,
        data=None,
        perplexity=30,
        method="auto",
        metric="euclidean",
        metric_params=None,
        symmetrize=True,
        n_jobs=1,
        random_state=None,
        verbose=False,
        k_neighbors="auto",
        knn_kwargs=None,
        knn_index=None,
        gamma=1.0,
    ):
        # This can't work if neither data nor the knn index are specified
        if data is None and knn_index is None:
            raise ValueError(
                "At least one of the parameters `data` or `knn_index` must be specified!"
            )
        # This can't work if both data and the knn index are specified
        if data is not None and knn_index is not None:
            raise ValueError(
                "Both `data` or `knn_index` were specified! Please pass only one."
            )

        # Find the nearest neighbors
        if knn_index is None:
            n_samples = data.shape[0]

            if k_neighbors == "auto":
                _k_neighbors = min(n_samples - 1, int(3 * perplexity))
            else:
                _k_neighbors = k_neighbors

            effective_perplexity = self.check_perplexity(perplexity, _k_neighbors)
            if _k_neighbors > int(3 * effective_perplexity):
                log.warning(
                    "The k_neighbors value is over 3 times larger than the perplexity value. "
                    "This may result in an unnecessary slowdown."
                )

            self.knn_index = get_knn_index(
                data,
                method,
                k=_k_neighbors,
                metric=metric,
                metric_params=metric_params,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                knn_kwargs=knn_kwargs,
            )

        else:
            self.knn_index = knn_index
            effective_perplexity = self.check_perplexity(perplexity, self.knn_index.k)
            log.info("KNN index provided. Ignoring KNN-related parameters.")

        self.__neighbors, self.__distances = self.knn_index.build()
        self.knn_indices = self.__neighbors
        self.knn_distances = self.__distances

        with utils.Timer("Calculating affinity matrix...", verbose):
            self.P, self.P_conditional = joint_probabilities_nn(
                self.__neighbors,
                self.__distances,
                [effective_perplexity],
                symmetrize=symmetrize,
                n_jobs=n_jobs,
                gamma=gamma,
            )

        self.perplexity = perplexity
        self.effective_perplexity_ = effective_perplexity
        self.symmetrize = symmetrize
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.knn_kwargs = knn_kwargs
        self.gamma = gamma

    def set_perplexity(self, new_perplexity):
        """Change the perplexity of the affinity matrix.

        Note that we only allow setting the perplexity to a value not larger
        than the number of neighbors used for the original perplexity. This
        restriction exists because setting a higher perplexity value requires
        recomputing all the nearest neighbors, which can take a long time.
        To avoid potential confusion as to why execution time is slow, this
        is not allowed. If you would like to increase the perplexity above
        that value, simply create a new instance.

        Parameters
        ----------
        new_perplexity: float
            The new perplexity.

        """
        # If the value hasn't changed, there's nothing to do
        if new_perplexity == self.perplexity:
            return
        # Verify that the perplexity isn't negative
        effective_perplexity = self.check_perplexity(new_perplexity, np.inf)
        # Verify that the perplexity isn't too large for the kNN graph
        if effective_perplexity > self.__neighbors.shape[1]:
            raise RuntimeError(
                "The desired perplexity `%.2f` is larger than the kNN graph "
                "allows. This would need to recompute the nearest neighbors, "
                "which is not efficient. Please create a new `%s` instance "
                "with the increased perplexity."
                % (effective_perplexity, self.__class__.__name__)
            )
        # Warn if the perplexity is larger than the heuristic
        if 3 * effective_perplexity > self.__neighbors.shape[1]:
            log.warning(
                "The new perplexity is quite close to the computed number of "
                "nearest neighbors. The results may be unexpected. Consider "
                "creating a new `%s` instance with the increased perplexity."
                % self.__class__.__name__
            )

        # Recompute the affinity matrix
        self.perplexity = new_perplexity
        self.effective_perplexity_ = effective_perplexity
        k_neighbors = int(3 * new_perplexity)

        with utils.Timer(
            "Perplexity changed. Recomputing affinity matrix...", self.verbose
        ):
            self.P, self.P_conditional = joint_probabilities_nn(
                self.__neighbors[:, :k_neighbors],
                self.__distances[:, :k_neighbors],
                [self.effective_perplexity_],
                symmetrize=self.symmetrize,
                n_jobs=self.n_jobs,
                gamma=self.gamma,
            )

    def to_new(
        self, data, perplexity=None, return_distances=False, k_neighbors="auto"
    ):
        """Compute the affinities of new samples to the initial samples.

        This is necessary for embedding new data points into an existing
        embedding.

        Please see the :ref:`parameter-guide` for more information.

        Parameters
        ----------
        data: np.ndarray
            The data points to be added to the existing embedding.

        perplexity: float
            Perplexity can be thought of as the continuous :math:`k` number of
            nearest neighbors, for which t-SNE will attempt to preserve
            distances.

        return_distances: bool
            If needed, the function can return the indices of the nearest
            neighbors and their corresponding distances.

        k_neighbors: int or ``auto``
            The number of neighbors to query kNN graph for. If ``auto``
            (default), it is set to three times the perplexity.

        Returns
        -------
        P: array_like
            An :math:`N \\times M` affinity matrix expressing interactions
            between :math:`N` new data points the initial :math:`M` data
            samples.

        indices: np.ndarray
            Returned if ``return_distances=True``. The indices of the :math:`k`
            nearest neighbors in the existing embedding for every new data
            point.

        distances: np.ndarray
            Returned if ``return_distances=True``. The distances to the
            :math:`k` nearest neighbors in the existing embedding for every new
            data point.

        """

        perplexity = perplexity if perplexity is not None else self.perplexity

        if k_neighbors == "auto":
            _k_neighbors = min(self.n_samples, int(3 * perplexity))
        else:
            _k_neighbors = k_neighbors

        effective_perplexity = self.check_perplexity(perplexity, _k_neighbors)

        neighbors, distances = self.knn_index.query(data, _k_neighbors)

        with utils.Timer("Calculating affinity matrix...", self.verbose):
            P, _ = joint_probabilities_nn(
                neighbors,
                distances,
                [effective_perplexity],
                symmetrize=False,
                normalization="point-wise",
                n_reference_samples=self.n_samples,
                n_jobs=self.n_jobs,
                gamma=self.gamma,
            )

        if return_distances:
            return P, neighbors, distances

        return P

    @staticmethod
    def check_perplexity(perplexity, k_neighbors):
        if perplexity <= 0:
            raise ValueError("Perplexity must be >=0. %.2f given" % perplexity)

        if perplexity > k_neighbors:
            old_perplexity, perplexity = perplexity, k_neighbors / 3
            log.warning(
                "Perplexity value %d is too high. Using perplexity %.2f instead"
                % (old_perplexity, perplexity)
            )

        return perplexity


def get_knn_index(
        data,
        method,
        k,
        metric,
        metric_params=None,
        n_jobs=1,
        random_state=None,
        verbose=False,
        knn_kwargs=None,
):
    # If we're dealing with a precomputed distance matrix, our job is very easy,
    # so we can skip all the remaining checks
    if metric == "precomputed":
        return nearest_neighbors.PrecomputedDistanceMatrix(data, k=k)

    preferred_approx_method = nearest_neighbors.Annoy
    if is_package_installed("pynndescent") and (sp.issparse(data) or metric not in [
        "cosine",
        "euclidean",
        "manhattan",
        "hamming",
        "dot",
        "l1",
        "l2",
        "taxicab",
    ]):
        preferred_approx_method = nearest_neighbors.NNDescent

    if data.shape[0] < 1000:
        preferred_method = nearest_neighbors.Sklearn
    else:
        preferred_method = preferred_approx_method

    methods = {
        "exact": nearest_neighbors.Sklearn,
        "auto": preferred_method,
        "approx": preferred_approx_method,
        "annoy": nearest_neighbors.Annoy,
        "pynndescent": nearest_neighbors.NNDescent,
        "hnsw": nearest_neighbors.HNSW
    }
    if isinstance(method, nearest_neighbors.KNNIndex):
        knn_index = method

    elif method not in methods:
        raise ValueError(
            "Unrecognized nearest neighbor algorithm `%s`. Please choose one "
            "of the supported methods or provide a valid `KNNIndex` instance." % method
        )
    else:
        knn_index = methods[method](
            data=data,
            k=k,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            knn_kwargs=knn_kwargs,
        )

    return knn_index


def joint_probabilities_nn(
    neighbors,
    distances,
    perplexities,
    symmetrize=True,
    normalization="pair-wise",
    n_reference_samples=None,
    n_jobs=1,
    gamma=1.0,
):
    """Compute the conditional probability matrix P_{j|i}.

    This method computes an approximation to P using the nearest neighbors.

    Parameters
    ----------
    neighbors: np.ndarray
        A `n_samples * k_neighbors` matrix containing the indices to each
        points' nearest neighbors in descending order.
    distances: np.ndarray
        A `n_samples * k_neighbors` matrix containing the distances to the
        neighbors at indices defined in the neighbors parameter.
    perplexities: double
        The desired perplexity of the probability distribution.
    symmetrize: bool
        Whether to symmetrize the probability matrix or not. Symmetrizing is
        used for typical t-SNE, but does not make sense when embedding new data
        into an existing embedding.
    normalization: str
        The normalization scheme to use for the affinities. Standard t-SNE
        considers interactions between all the data points, therefore the entire
        affinity matrix is regarded as a probability distribution, and must sum
        to 1. When embedding new points, we only consider interactions to
        existing points, and treat each point separately. In this case, we
        row-normalize the affinity matrix, meaning each point gets its own
        probability distribution.
    n_reference_samples: int
        The number of samples in the existing (reference) embedding. Needed to
        properly construct the sparse P matrix.
    n_jobs: int
        Number of threads.
    gamma: float
        Power transform exponent applied to the conditional probabilities before
        symmetrization. Each row is raised to the power `gamma` and renormalized
        to sum to 1. ``gamma=1`` (default) leaves the probabilities unchanged.
        ``gamma > 1`` sharpens the distribution (emphasizes the nearest
        neighbors), while ``0 < gamma < 1`` smooths it.

    Returns
    -------
    csr_matrix
        A `n_samples * n_reference_samples` matrix containing the probabilities
        that a new sample would appear as a neighbor of a reference point.

    """
    assert normalization in (
        "pair-wise",
        "point-wise",
    ), f"Unrecognized normalization scheme `{normalization}`."

    n_samples, k_neighbors = distances.shape

    if n_reference_samples is None:
        n_reference_samples = n_samples

    # Compute asymmetric pairwise input similarities
    conditional_P = _tsne.compute_gaussian_perplexity(
        np.array(distances, dtype=float),
        np.array(perplexities, dtype=float),
        num_threads=n_jobs,
    )
    conditional_P = np.asarray(conditional_P)

    # Row-wise power transform: p_{j|i} -> p_{j|i}^gamma / sum_m p_{m|i}^gamma
    # Operates on the dense (n_samples, k_neighbors) array before sparse construction.
    # Zero entries remain zero (0**gamma == 0 for gamma > 0), preserving sparsity.
    # gamma=1 is the identity; skip to avoid unnecessary computation.
    if gamma != 1.0:
        conditional_P **= gamma
        conditional_P /= conditional_P.sum(axis=1, keepdims=True)

    P_conditional = sp.csr_matrix(
        (
            conditional_P.ravel(),
            neighbors.ravel(),
            range(0, n_samples * k_neighbors + 1, k_neighbors),
        ),
        shape=(n_samples, n_reference_samples),
    )

    # Symmetrize the probability matrix
    if symmetrize:
        P = (P_conditional + P_conditional.T) / 2
    else:
        # Decouple P from P_conditional so downstream in-place ops don't mutate it
        P = P_conditional.copy()

    if normalization == "pair-wise":
        P /= np.sum(P)
    elif normalization == "point-wise":
        P = sp.diags(np.asarray(1 / P.sum(axis=1)).ravel()) @ P

    return P, P_conditional


def estimate_intrinsic_dim(distances):
    """Maximum-likelihood (Levina–Bickel) estimate of the intrinsic dimensionality.

    Uses the per-point kNN distances already computed for the affinity graph, so
    no extra neighbor search is required. For each point with sorted neighbor
    distances :math:`r_1 \\le \\dots \\le r_k`, the local estimate is

    .. math::
        \\hat m_i = \\Big( \\tfrac{1}{k-1} \\sum_{j=1}^{k-1} \\log \\tfrac{r_k}{r_j} \\Big)^{-1},

    and the global estimate averages the inverses across points (the more robust
    Levina–Bickel/MacKay aggregation). Returned value is clipped to ``>= 1``.

    Parameters
    ----------
    distances: np.ndarray
        A ``(n_samples, k)`` array of neighbor distances, sorted ascending along
        each row, self excluded.

    Returns
    -------
    float
        The estimated intrinsic dimensionality :math:`M'`.

    """
    d = np.asarray(distances, dtype=np.float64)
    if d.shape[1] < 2:
        raise ValueError("Need at least 2 neighbors to estimate intrinsic dimensionality.")

    eps = np.finfo(np.float64).tiny
    d = np.maximum(d, eps)
    r_k = d[:, -1][:, np.newaxis]
    log_ratios = np.log(r_k / d[:, :-1]).sum(axis=1)   # sum_{j=1}^{k-1} log(r_k / r_j)
    log_ratios = np.maximum(log_ratios, eps)
    m_i = (d.shape[1] - 1) / log_ratios

    m_hat = 1.0 / np.mean(1.0 / m_i)
    return float(max(m_hat, 1.0))


def student_conditional_probabilities(distances, perplexity, dof, n_iter=100, tol=1e-5):
    """Row-wise Student-t conditional probabilities calibrated to a perplexity.

    For each point ``i``, computes weights with a Student-t kernel of ``dof``
    degrees of freedom over its neighbors,

    .. math::
        w_{ij} \\propto \\big(1 + \\pi_i \\delta_{ij}^2 / \\nu \\big)^{-\\nu/2},

    where the precision :math:`\\pi_i` is found by binary search so that the
    Shannon entropy of the normalized row equals ``log(perplexity)`` — exactly
    the calibration t-SNE uses for Gaussian affinities, but with a heavy-tailed
    kernel. As ``dof`` :math:`\\to \\infty` the kernel tends to a Gaussian and
    this reduces to the standard perplexity-based affinity.

    Parameters
    ----------
    distances: np.ndarray
        A ``(n_samples, k)`` array of neighbor distances.
    perplexity: float
        Target perplexity for every row.
    dof: float
        Degrees of freedom :math:`\\nu` of the Student-t kernel (the intrinsic
        dimensionality :math:`M'` in tt-SNE). Must be ``> 0``.
    n_iter: int
        Number of binary-search iterations for the per-point precision.
    tol: float
        Stop early once every row is within ``tol`` of the target entropy.

    Returns
    -------
    np.ndarray
        A ``(n_samples, k)`` array of conditional probabilities; each row sums
        to 1.

    """
    sq = np.asarray(distances, dtype=np.float64) ** 2
    n_samples = sq.shape[0]
    target_entropy = np.log(perplexity)

    beta = np.ones(n_samples)            # precision pi_i
    beta_min = np.full(n_samples, -np.inf)
    beta_max = np.full(n_samples, np.inf)
    half_dof = dof / 2.0

    P = np.full_like(sq, 1.0 / sq.shape[1])
    for _ in range(n_iter):
        W = (1.0 + (beta[:, np.newaxis] * sq) / dof) ** (-half_dof)
        sum_W = W.sum(axis=1)
        sum_W[sum_W <= 0] = np.finfo(np.float64).tiny
        P = W / sum_W[:, np.newaxis]

        log_P = np.where(P > 0, np.log(P), 0.0)
        entropy = -(P * log_P).sum(axis=1)
        diff = entropy - target_entropy

        if np.max(np.abs(diff)) < tol:
            break

        # entropy decreases as the precision (beta) grows
        too_uniform = diff > 0           # entropy too high -> need larger beta
        too_peaked = ~too_uniform

        beta_min = np.where(too_uniform, beta, beta_min)
        beta = np.where(
            too_uniform & np.isinf(beta_max), beta * 2.0,
            np.where(too_uniform, (beta + beta_max) / 2.0, beta),
        )
        beta_max = np.where(too_peaked, beta, beta_max)
        beta = np.where(
            too_peaked & np.isinf(beta_min), beta / 2.0,
            np.where(too_peaked, (beta + beta_min) / 2.0, beta),
        )

    return P


def student_joint_probabilities_nn(
    neighbors,
    distances,
    perplexity,
    dof,
    symmetrize=True,
    normalization="pair-wise",
    n_reference_samples=None,
    gamma=1.0,
):
    """Build a (symmetrized) Student-t joint probability matrix from kNN graphs.

    Mirrors :func:`joint_probabilities_nn`, but the conditional similarities use
    a heavy-tailed Student-t kernel (tt-SNE) instead of a Gaussian. See
    :func:`student_conditional_probabilities` for the kernel and the perplexity
    calibration, and :class:`StudentTNN` for the corresponding affinity class.

    """
    assert normalization in (
        "pair-wise",
        "point-wise",
    ), f"Unrecognized normalization scheme `{normalization}`."

    n_samples, k_neighbors = distances.shape
    if n_reference_samples is None:
        n_reference_samples = n_samples

    conditional_P = student_conditional_probabilities(distances, perplexity, dof)

    # Same row-wise power transform as the Gaussian affinities (gamma=1 is identity).
    if gamma != 1.0:
        conditional_P **= gamma
        conditional_P /= conditional_P.sum(axis=1, keepdims=True)

    P_conditional = sp.csr_matrix(
        (
            conditional_P.ravel(),
            neighbors.ravel(),
            range(0, n_samples * k_neighbors + 1, k_neighbors),
        ),
        shape=(n_samples, n_reference_samples),
    )

    if symmetrize:
        P = (P_conditional + P_conditional.T) / 2
    else:
        P = P_conditional.copy()

    if normalization == "pair-wise":
        P /= np.sum(P)
    elif normalization == "point-wise":
        P = sp.diags(np.asarray(1 / P.sum(axis=1)).ravel()) @ P

    return P, P_conditional


class StudentTNN(Affinities):
    """Compute heavy-tailed Student-t affinities (twice Student, tt-SNE).

    Standard t-SNE uses a Gaussian kernel in the high-dimensional space and a
    Student-t kernel only in the embedding. The *twice Student* variant (tt-SNE)
    of de Bodt et al. (ESANN 2018) uses a Student-t kernel in **both** spaces:
    the high-dimensional conditional similarities become

    .. math::
        \\sigma_{ij} = \\frac{(1 + \\pi_i \\delta_{ij}^2 / M')^{-M'/2}}
                            {\\sum_{k \\neq i} (1 + \\pi_i \\delta_{ik}^2 / M')^{-M'/2}},

    where the degrees of freedom equal the intrinsic dimensionality :math:`M'`
    of the data and the precisions :math:`\\pi_i` are calibrated to a
    user-specified perplexity, exactly as in standard t-SNE. The heavier tails
    of the Student kernel give mid- to far-range neighbors more weight than a
    Gaussian; as :math:`M' \\to \\infty` the kernel tends to a Gaussian and this
    class reduces to :class:`PerplexityBasedNN`.

    The embedding-space (low-dimensional) Student-t kernel is unchanged and is
    handled by the t-SNE optimizer, so this affinity is a drop-in replacement
    for any other affinity object.

    Reference: C. de Bodt, D. Mulders, M. Verleysen, J. A. Lee, "Perplexity-free
    t-SNE and twice Student tt-SNE", ESANN 2018, pp. 123-128.

    Parameters
    ----------
    data: np.ndarray
        The data matrix.

    perplexity: float
        Perplexity can be thought of as the continuous :math:`k` number of
        nearest neighbors, for which t-SNE will attempt to preserve distances.

    dof: float or ``auto``
        Degrees of freedom of the high-dimensional Student-t kernel, i.e. the
        intrinsic dimensionality :math:`M'`. If ``auto`` (default), it is
        estimated from the kNN distances with the Levina–Bickel maximum
        likelihood estimator (:func:`estimate_intrinsic_dim`). Larger values
        give lighter tails (closer to Gaussian/standard t-SNE).

    method: str
        Specifies the nearest neighbor method to use. Can be ``exact``, ``annoy``,
        ``pynndescent``, ``hnsw``, ``approx``, or ``auto`` (default).

    metric: Union[str, Callable]
        The metric to be used to compute affinities between points in the
        original space.

    metric_params: dict
        Additional keyword arguments for the metric function.

    symmetrize: bool
        Symmetrize the affinity matrix.

    n_jobs: int
        The number of threads to use while running t-SNE. This follows the
        scikit-learn convention, ``-1`` meaning all processors, ``-2`` meaning
        all but one, etc.

    random_state: Union[int, RandomState]
        Seed or random state for the nearest-neighbor search.

    verbose: bool

    k_neighbors: int or ``auto``
        The number of neighbors to use in the kNN graph. If ``auto`` (default),
        it is set to three times the perplexity.

    knn_kwargs: Optional[None, dict]
        Optional keyword arguments that will be passed to the ``knn_index``.

    knn_index: Optional[nearest_neighbors.KNNIndex]
        Optionally, a precomputed ``openTSNE.nearest_neighbors.KNNIndex`` object
        can be specified. When ``knn_index`` is specified, ``data`` must be None.

    gamma: float
        Optional row-wise power transform applied to the conditional
        probabilities, matching :class:`PerplexityBasedNN`. ``gamma=1`` (default)
        leaves the Student-t affinities unchanged.

    """

    def __init__(
        self,
        data=None,
        perplexity=30,
        dof="auto",
        method="auto",
        metric="euclidean",
        metric_params=None,
        symmetrize=True,
        n_jobs=1,
        random_state=None,
        verbose=False,
        k_neighbors="auto",
        knn_kwargs=None,
        knn_index=None,
        gamma=1.0,
    ):
        if data is None and knn_index is None:
            raise ValueError(
                "At least one of the parameters `data` or `knn_index` must be specified!"
            )
        if data is not None and knn_index is not None:
            raise ValueError(
                "Both `data` or `knn_index` were specified! Please pass only one."
            )

        if knn_index is None:
            n_samples = data.shape[0]

            if k_neighbors == "auto":
                _k_neighbors = min(n_samples - 1, int(3 * perplexity))
            else:
                _k_neighbors = k_neighbors

            effective_perplexity = PerplexityBasedNN.check_perplexity(
                perplexity, _k_neighbors
            )
            if _k_neighbors > int(3 * effective_perplexity):
                log.warning(
                    "The k_neighbors value is over 3 times larger than the perplexity value. "
                    "This may result in an unnecessary slowdown."
                )

            self.knn_index = get_knn_index(
                data,
                method,
                k=_k_neighbors,
                metric=metric,
                metric_params=metric_params,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                knn_kwargs=knn_kwargs,
            )

        else:
            self.knn_index = knn_index
            effective_perplexity = PerplexityBasedNN.check_perplexity(
                perplexity, self.knn_index.k
            )
            log.info("KNN index provided. Ignoring KNN-related parameters.")

        self._neighbors, self._distances = self.knn_index.build()
        self.knn_indices = self._neighbors
        self.knn_distances = self._distances

        if dof == "auto":
            dof = estimate_intrinsic_dim(self._distances)
            log.info("Estimated intrinsic dimensionality (dof) = %.3f" % dof)
        dof = float(dof)
        if dof <= 0:
            raise ValueError("`dof` must be > 0, got %.3f" % dof)

        with utils.Timer("Calculating affinity matrix...", verbose):
            self.P, self.P_conditional = student_joint_probabilities_nn(
                self._neighbors,
                self._distances,
                effective_perplexity,
                dof=dof,
                symmetrize=symmetrize,
                gamma=gamma,
            )

        self.perplexity = perplexity
        self.effective_perplexity_ = effective_perplexity
        self.dof = dof
        self.symmetrize = symmetrize
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.knn_kwargs = knn_kwargs
        self.gamma = gamma

    def to_new(
        self, data, perplexity=None, return_distances=False, k_neighbors="auto"
    ):
        """Compute the Student-t affinities of new samples to the initial samples."""
        perplexity = perplexity if perplexity is not None else self.perplexity

        if k_neighbors == "auto":
            _k_neighbors = min(self.n_samples, int(3 * perplexity))
        else:
            _k_neighbors = k_neighbors

        effective_perplexity = PerplexityBasedNN.check_perplexity(
            perplexity, _k_neighbors
        )

        neighbors, distances = self.knn_index.query(data, _k_neighbors)

        with utils.Timer("Calculating affinity matrix...", self.verbose):
            P, _ = student_joint_probabilities_nn(
                neighbors,
                distances,
                effective_perplexity,
                dof=self.dof,
                symmetrize=False,
                normalization="point-wise",
                n_reference_samples=self.n_samples,
                gamma=self.gamma,
            )

        if return_distances:
            return P, neighbors, distances

        return P


class FixedSigmaNN(Affinities):
    """Compute affinities using nearest neighbors and a fixed bandwidth
    for the Gaussians in the ambient space.

    Using a fixed Gaussian bandwidth can enable us to find smaller clusters of
    data points than we might be able to using the automatically determined
    bandwidths using perplexity. Note however that this requires mostly trial
    and error.

    Parameters
    ----------
    data: np.ndarray
        The data matrix.

    sigma: float
        The bandwidth to use for the Gaussian kernels in the ambient space.

    k: int
        The number of nearest neighbors to consider for each kernel.

    method: str
        Specifies the nearest neighbor method to use. Can be ``exact``, ``annoy``,
        ``pynndescent``, ``hnsw``, ``approx``, or ``auto`` (default). ``approx``
        uses Annoy if the input data matrix is not a sparse object and if Annoy
        supports the given metric. Otherwise, it uses Pynndescent. ``auto`` uses
        exact nearest neighbors for N<1000 and the same heuristic as ``approx``
        for N>=1000.

    metric: Union[str, Callable]
        The metric to be used to compute affinities between points in the
        original space.

    metric_params: dict
        Additional keyword arguments for the metric function.

    symmetrize: bool
        Symmetrize the affinity matrix. During standard t-SNE optimization, the
        affinities are symmetrized. However, when embedding new data points into
        existing embeddings, symmetrization is not performed.

    n_jobs: int
        The number of threads to use while running t-SNE. This follows the
        scikit-learn convention, ``-1`` meaning all processors, ``-2`` meaning
        all but one, etc.

    random_state: Union[int, RandomState]
        If the value is an int, random_state is the seed used by the random
        number generator. If the value is a RandomState instance, then it will
        be used as the random number generator. If the value is None, the random
        number generator is the RandomState instance used by `np.random`.

    verbose: bool

    knn_kwargs: Optional[None, dict]
        Optional keyword arguments that will be passed to the ``knn_index``.

    knn_index: Optional[nearest_neighbors.KNNIndex]
        Optionally, a precomptued ``openTSNE.nearest_neighbors.KNNIndex`` object
        can be specified. This option will ignore any KNN-related parameters.
        When ``knn_index`` is specified, ``data`` must be set to None.

    """

    def __init__(
        self,
        data=None,
        sigma=None,
        k=30,
        method="auto",
        metric="euclidean",
        metric_params=None,
        symmetrize=True,
        n_jobs=1,
        random_state=None,
        verbose=False,
        knn_kwargs=None,
        knn_index=None,
        gamma=1.0,
    ):
        # Sigma must be specified, but has default set to none, so the parameter
        # order makes more sense
        if sigma is None:
            raise ValueError("`sigma` must be specified!")

        # This can't work if neither data nor the knn index are specified
        if data is None and knn_index is None:
            raise ValueError(
                "At least one of the parameters `data` or `knn_index` must be specified!"
            )
        # This can't work if both data and the knn index are specified
        if data is not None and knn_index is not None:
            raise ValueError(
                "Both `data` or `knn_index` were specified! Please pass only one."
            )

        # Find the nearest neighbors
        if knn_index is None:
            if k >= data.shape[0]:
                raise ValueError(
                    "`k` (%d) cannot be larger than N-1 (%d)." % (k, data.shape[0])
                )

            self.knn_index = get_knn_index(
                data,
                method,
                k=k,
                metric=metric,
                metric_params=metric_params,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                knn_kwargs=knn_kwargs,
            )

        else:
            self.knn_index = knn_index
            log.info("KNN index provided. Ignoring KNN-related parameters.")

        neighbors, distances = self.knn_index.build()

        with utils.Timer("Calculating affinity matrix...", verbose):
            # Compute asymmetric pairwise input similarities
            conditional_P = np.exp(-(distances ** 2) / (2 * sigma ** 2))
            conditional_P /= np.sum(conditional_P, axis=1)[:, np.newaxis]

            if gamma != 1.0:
                conditional_P **= gamma
                conditional_P /= conditional_P.sum(axis=1, keepdims=True)

            n_samples = self.knn_index.n_samples
            P_conditional = sp.csr_matrix(
                (
                    conditional_P.ravel(),
                    neighbors.ravel(),
                    range(0, n_samples * k + 1, k),
                ),
                shape=(n_samples, n_samples),
            )

            # Symmetrize the probability matrix
            if symmetrize:
                P = (P_conditional + P_conditional.T) / 2
            else:
                P = P_conditional.copy()

            # Convert weights to probabilities
            P /= np.sum(P)

        self.sigma = sigma
        self.gamma = gamma
        self.P = P
        self.P_conditional = P_conditional
        self.n_jobs = n_jobs
        self.verbose = verbose

    def to_new(self, data, k=None, sigma=None, return_distances=False):
        """Compute the affinities of new samples to the initial samples.

        This is necessary for embedding new data points into an existing
        embedding.

        Parameters
        ----------
        data: np.ndarray
            The data points to be added to the existing embedding.

        k: int
            The number of nearest neighbors to consider for each kernel.

        sigma: float
            The bandwidth to use for the Gaussian kernels in the ambient space.

        return_distances: bool
            If needed, the function can return the indices of the nearest
            neighbors and their corresponding distances.

        Returns
        -------
        P: array_like
            An :math:`N \\times M` affinity matrix expressing interactions
            between :math:`N` new data points the initial :math:`M` data
            samples.

        indices: np.ndarray
            Returned if ``return_distances=True``. The indices of the :math:`k`
            nearest neighbors in the existing embedding for every new data
            point.

        distances: np.ndarray
            Returned if ``return_distances=True``. The distances to the
            :math:`k` nearest neighbors in the existing embedding for every new
            data point.

        """
        n_samples = data.shape[0]
        n_reference_samples = self.n_samples

        if k is None:
            k = self.knn_index.k
        elif k >= n_reference_samples:
            raise ValueError(
                "`k` (%d) cannot be larger than the number of reference "
                "samples (%d)." % (k, self.n_samples)
            )

        if sigma is None:
            sigma = self.sigma

        # Find nearest neighbors and the distances to the new points
        neighbors, distances = self.knn_index.query(data, k)

        with utils.Timer("Calculating affinity matrix...", self.verbose):
            # Compute asymmetric pairwise input similarities
            conditional_P = np.exp(-(distances ** 2) / (2 * sigma ** 2))

            # Convert weights to probabilities
            conditional_P /= np.sum(conditional_P, axis=1)[:, np.newaxis]

            if self.gamma != 1.0:
                conditional_P **= self.gamma
                conditional_P /= conditional_P.sum(axis=1, keepdims=True)

            P = sp.csr_matrix(
                (
                    conditional_P.ravel(),
                    neighbors.ravel(),
                    range(0, n_samples * k + 1, k),
                ),
                shape=(n_samples, n_reference_samples),
            )

        if return_distances:
            return P, neighbors, distances

        return P


class MultiscaleMixture(Affinities):
    """Calculate affinities using a Gaussian mixture kernel.

    Instead of using a single perplexity to compute the affinities between data
    points, we can use a multiscale Gaussian kernel instead. This allows us to
    incorporate long range interactions.

    Please see the :ref:`parameter-guide` for more information.

    Parameters
    ----------
    data: np.ndarray
        The data matrix.

    perplexities: List[float]
        A list of perplexity values, which will be used in the multiscale
        Gaussian kernel. Perplexity can be thought of as the continuous
        :math:`k` number of nearest neighbors, for which t-SNE will attempt to
        preserve distances.

    method: str
        Specifies the nearest neighbor method to use. Can be ``exact``, ``annoy``,
        ``pynndescent``, ``hnsw``, ``approx``, or ``auto`` (default). ``approx``
        uses Annoy if the input data matrix is not a sparse object and if Annoy
        supports the given metric. Otherwise, it uses Pynndescent. ``auto`` uses
        exact nearest neighbors for N<1000 and the same heuristic as ``approx``
        for N>=1000.

    metric: Union[str, Callable]
        The metric to be used to compute affinities between points in the
        original space.

    metric_params: dict
        Additional keyword arguments for the metric function.

    symmetrize: bool
        Symmetrize the affinity matrix. During standard t-SNE optimization, the
        affinities are symmetrized. However, when embedding new data points into
        existing embeddings, symmetrization is not performed.

    n_jobs: int
        The number of threads to use while running t-SNE. This follows the
        scikit-learn convention, ``-1`` meaning all processors, ``-2`` meaning
        all but one, etc.

    random_state: Union[int, RandomState]
        If the value is an int, random_state is the seed used by the random
        number generator. If the value is a RandomState instance, then it will
        be used as the random number generator. If the value is None, the random
        number generator is the RandomState instance used by `np.random`.

    verbose: bool

    knn_kwargs: Optional[None, dict]
        Optional keyword arguments that will be passed to the ``knn_index``.

    knn_index: Optional[nearest_neighbors.KNNIndex]
        Optionally, a precomptued ``openTSNE.nearest_neighbors.KNNIndex`` object
        can be specified. This option will ignore any KNN-related parameters.
        When ``knn_index`` is specified, ``data`` must be set to None.

    """

    def __init__(
        self,
        data=None,
        perplexities=None,
        method="auto",
        metric="euclidean",
        metric_params=None,
        symmetrize=True,
        n_jobs=1,
        random_state=None,
        verbose=False,
        knn_kwargs=None,
        knn_index=None,
        gamma=1.0,
    ):
        # Perplexities must be specified, but has default set to none, so the
        # parameter order makes more sense
        if perplexities is None:
            raise ValueError("`perplexities` must be specified!")

        # This can't work if neither data nor the knn index are specified
        if data is None and knn_index is None:
            raise ValueError(
                "At least one of the parameters `data` or `knn_index` must be specified!"
            )
        # This can't work if both data and the knn index are specified
        if data is not None and knn_index is not None:
            raise ValueError(
                "Both `data` or `knn_index` were specified! Please pass only one."
            )

        # Find the nearest neighbors
        if knn_index is None:
            # We will compute the nearest neighbors to the max value of perplexity,
            # smaller values can just use indexing to truncate unneeded neighbors
            n_samples = data.shape[0]
            effective_perplexities = self.check_perplexities(perplexities, n_samples)
            max_perplexity = np.max(effective_perplexities)
            k_neighbors = min(n_samples - 1, int(3 * max_perplexity))

            self.knn_index = get_knn_index(
                data,
                method,
                k=k_neighbors,
                metric=metric,
                metric_params=metric_params,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                knn_kwargs=knn_kwargs,
            )

        else:
            self.knn_index = knn_index
            n_samples = self.knn_index.n_samples
            effective_perplexities = self.check_perplexities(perplexities, n_samples)
            log.info("KNN index provided. Ignoring KNN-related parameters.")

        self.__neighbors, self.__distances = self.knn_index.build()

        with utils.Timer("Calculating affinity matrix...", verbose):
            self.P, self.P_conditional = self._calculate_P(
                self.__neighbors,
                self.__distances,
                effective_perplexities,
                symmetrize=symmetrize,
                n_jobs=n_jobs,
                gamma=gamma,
            )

        self.perplexities = perplexities
        self.effective_perplexities_ = effective_perplexities
        self.symmetrize = symmetrize
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.gamma = gamma

    @staticmethod
    def _calculate_P(
        neighbors,
        distances,
        perplexities,
        symmetrize=True,
        normalization="pair-wise",
        n_reference_samples=None,
        n_jobs=1,
        gamma=1.0,
    ):
        return joint_probabilities_nn(
            neighbors,
            distances,
            perplexities,
            symmetrize=symmetrize,
            normalization=normalization,
            n_reference_samples=n_reference_samples,
            n_jobs=n_jobs,
            gamma=gamma,
        )
        # Returns (P, P_conditional)

    def set_perplexities(self, new_perplexities):
        """Change the perplexities of the affinity matrix.

        Note that we only allow lowering the perplexities or restoring them to
        their original maximum value. This restriction exists because setting a
        higher perplexity value requires recomputing all the nearest neighbors,
        which can take a long time. To avoid potential confusion as to why
        execution time is slow, this is not allowed. If you would like to
        increase the perplexity above the initial value, simply create a new
        instance.

        Parameters
        ----------
        new_perplexities: List[float]
            The new list of perplexities.

        """
        if np.array_equal(self.perplexities, new_perplexities):
            return

        effective_perplexities = self.check_perplexities(new_perplexities, self.n_samples)
        max_perplexity = np.max(effective_perplexities)
        k_neighbors = min(self.n_samples - 1, int(3 * max_perplexity))

        if k_neighbors > self.__neighbors.shape[1]:
            raise RuntimeError(
                "The largest perplexity `%.2f` is larger than the initial one "
                "used. This would need to recompute the nearest neighbors, "
                "which is not efficient. Please create a new `%s` instance "
                "with the increased perplexity."
                % (max_perplexity, self.__class__.__name__)
            )

        self.perplexities = new_perplexities
        self.effective_perplexities_ = effective_perplexities
        with utils.Timer(
            "Perplexity changed. Recomputing affinity matrix...", self.verbose
        ):
            self.P, self.P_conditional = self._calculate_P(
                self.__neighbors[:, :k_neighbors],
                self.__distances[:, :k_neighbors],
                self.effective_perplexities_,
                symmetrize=self.symmetrize,
                n_jobs=self.n_jobs,
                gamma=self.gamma,
            )

    def to_new(self, data, perplexities=None, return_distances=False):
        """Compute the affinities of new samples to the initial samples.

        This is necessary for embedding new data points into an existing
        embedding.

        Please see the :ref:`parameter-guide` for more information.

        Parameters
        ----------
        data: np.ndarray
            The data points to be added to the existing embedding.

        perplexities: List[float]
            A list of perplexity values, which will be used in the multiscale
            Gaussian kernel. Perplexity can be thought of as the continuous
            :math:`k` number of nearest neighbors, for which t-SNE will attempt
            to preserve distances.

        return_distances: bool
            If needed, the function can return the indices of the nearest
            neighbors and their corresponding distances.

        Returns
        -------
        P: array_like
            An :math:`N \\times M` affinity matrix expressing interactions
            between :math:`N` new data points the initial :math:`M` data
            samples.

        indices: np.ndarray
            Returned if ``return_distances=True``. The indices of the :math:`k`
            nearest neighbors in the existing embedding for every new data
            point.

        distances: np.ndarray
            Returned if ``return_distances=True``. The distances to the
            :math:`k` nearest neighbors in the existing embedding for every new
            data point.

        """
        perplexities = perplexities if perplexities is not None else self.perplexities
        effective_perplexities = self.check_perplexities(perplexities, self.n_samples)

        max_perplexity = np.max(effective_perplexities)
        k_neighbors = min(self.n_samples - 1, int(3 * max_perplexity))

        neighbors, distances = self.knn_index.query(data, k_neighbors)

        with utils.Timer("Calculating affinity matrix...", self.verbose):
            P, _ = self._calculate_P(
                neighbors,
                distances,
                effective_perplexities,
                symmetrize=False,
                normalization="point-wise",
                n_reference_samples=self.n_samples,
                n_jobs=self.n_jobs,
                gamma=self.gamma,
            )

        if return_distances:
            return P, neighbors, distances

        return P

    def check_perplexities(self, perplexities, n_samples):
        """Check and correct/truncate perplexities.

        If a perplexity is too large, it is corrected to the largest allowed
        value. It is then inserted into the list of perplexities only if that
        value doesn't already exist in the list.

        """
        if isinstance(perplexities, numbers.Number):
            perplexities = [perplexities]

        usable_perplexities = []
        for perplexity in sorted(perplexities):
            if perplexity <= 0:
                raise ValueError("Perplexity must be >=0. %.2f given" % perplexity)

            if 3 * perplexity > n_samples - 1:
                new_perplexity = (n_samples - 1) / 3

                if new_perplexity in usable_perplexities:
                    log.warning(
                        "Perplexity value %d is too high. Dropping "
                        "because the max perplexity is already in the "
                        "list." % perplexity
                    )
                else:
                    usable_perplexities.append(new_perplexity)
                    log.warning(
                        "Perplexity value %d is too high. Using "
                        "perplexity %.2f instead" % (perplexity, new_perplexity)
                    )
            else:
                usable_perplexities.append(perplexity)

        return usable_perplexities


class Multiscale(MultiscaleMixture):
    """Calculate affinities using averaged Gaussian perplexities.

    In contrast to :class:`MultiscaleMixture`, which uses a Gaussian mixture
    kernel, here, we first compute single scale Gaussian kernels, convert them
    to probability distributions, then average them out between scales.

    Please see the :ref:`parameter-guide` for more information.

    Parameters
    ----------
    data: np.ndarray
        The data matrix.

    perplexities: List[float]
        A list of perplexity values, which will be used in the multiscale
        Gaussian kernel. Perplexity can be thought of as the continuous
        :math:`k` number of nearest neighbors, for which t-SNE will attempt to
        preserve distances.

    method: str
        Specifies the nearest neighbor method to use. Can be ``exact``, ``annoy``,
        ``pynndescent``, ``hnsw``, ``approx``, or ``auto`` (default). ``approx``
        uses Annoy if the input data matrix is not a sparse object and if Annoy
        supports the given metric. Otherwise, it uses Pynndescent. ``auto`` uses
        exact nearest neighbors for N<1000 and the same heuristic as ``approx``
        for N>=1000.

    metric: Union[str, Callable]
        The metric to be used to compute affinities between points in the
        original space.

    metric_params: dict
        Additional keyword arguments for the metric function.

    symmetrize: bool
        Symmetrize the affinity matrix. During standard t-SNE optimization, the
        affinities are symmetrized. However, when embedding new data points into
        existing embeddings, symmetrization is not performed.

    n_jobs: int
        The number of threads to use while running t-SNE. This follows the
        scikit-learn convention, ``-1`` meaning all processors, ``-2`` meaning
        all but one, etc.

    random_state: Union[int, RandomState]
        If the value is an int, random_state is the seed used by the random
        number generator. If the value is a RandomState instance, then it will
        be used as the random number generator. If the value is None, the random
        number generator is the RandomState instance used by `np.random`.

    verbose: bool

    knn_index: Optional[nearest_neighbors.KNNIndex]
        Optionally, a precomptued ``openTSNE.nearest_neighbors.KNNIndex`` object
        can be specified. This option will ignore any KNN-related parameters.
        When ``knn_index`` is specified, ``data`` must be set to None.

    """

    @staticmethod
    def _calculate_P(
        neighbors,
        distances,
        perplexities,
        symmetrize=True,
        normalization="pair-wise",
        n_reference_samples=None,
        n_jobs=1,
        gamma=1.0,
    ):
        # Compute normalized probabilities for each perplexity
        # Returns (P, P_conditional)
        partial_Ps = []
        partial_P_conds = []
        for perplexity in perplexities:
            P_i, P_cond_i = joint_probabilities_nn(
                neighbors,
                distances,
                [perplexity],
                symmetrize=symmetrize,
                normalization=normalization,
                n_reference_samples=n_reference_samples,
                n_jobs=n_jobs,
                gamma=gamma,
            )
            partial_Ps.append(P_i)
            partial_P_conds.append(P_cond_i)

        # Sum them together, then normalize
        P = reduce(operator.add, partial_Ps, 0)

        # Take care to properly normalize the affinity matrix
        if normalization == "pair-wise":
            P /= np.sum(P)
        elif normalization == "point-wise":
            P = sp.diags(np.asarray(1 / P.sum(axis=1)).ravel()) @ P

        # Aggregate conditional Ps: sum then row-renormalize
        P_conditional = reduce(operator.add, partial_P_conds, 0)
        row_sums = np.asarray(P_conditional.sum(axis=1)).ravel()
        P_conditional = sp.diags(1.0 / row_sums) @ P_conditional

        return P, P_conditional


class Uniform(Affinities):
    """Compute affinities using nearest neighbors and uniform kernel in
    the ambient space.

    Parameters
    ----------
    data: np.ndarray
        The data matrix.

    k_neighbors: int

    method: str
        Specifies the nearest neighbor method to use. Can be ``exact``, ``annoy``,
        ``pynndescent``, ``hnsw``, ``approx``, or ``auto`` (default). ``approx``
        uses Annoy if the input data matrix is not a sparse object and if Annoy
        supports the given metric. Otherwise, it uses Pynndescent. ``auto`` uses
        exact nearest neighbors for N<1000 and the same heuristic as ``approx``
        for N>=1000.

    metric: Union[str, Callable]
        The metric to be used to compute affinities between points in the
        original space.

    metric_params: dict
        Additional keyword arguments for the metric function.

    symmetrize: Union[str, bool]
        Symmetrize the affinity matrix. During standard t-SNE optimization, the
        affinities are symmetrized. However, when embedding new data points into
        existing embeddings, symmetrization is not performed.
        The uniform affinity supports ``max`` and ``mean`` symmetrization, as
        well as no symmetrization via ``none``.
        The ``max`` symmetrization yields a binary affinity matrix with all
        non-zero elements (corresponding to edges of the kNN graph) being the
        same. The ``mean`` symmetrization performs symmetrization via
        (A + A.T) / 2, resulting in the affinity matrix with two possible
        non-zero values. Applying no symmetrization results in a non-symmetric
        affinity matrix. We default to ``mean`` symmetrization, but the default
        will change to ``max`` in future versions.

    n_jobs: int
        The number of threads to use while running t-SNE. This follows the
        scikit-learn convention, ``-1`` meaning all processors, ``-2`` meaning
        all but one, etc.

    random_state: Union[int, RandomState]
        If the value is an int, random_state is the seed used by the random
        number generator. If the value is a RandomState instance, then it will
        be used as the random number generator. If the value is None, the random
        number generator is the RandomState instance used by `np.random`.

    verbose: bool

    knn_kwargs: Optional[None, dict]
        Optional keyword arguments that will be passed to the ``knn_index``.

    knn_index: Optional[nearest_neighbors.KNNIndex]
        Optionally, a precomptued ``openTSNE.nearest_neighbors.KNNIndex`` object
        can be specified. This option will ignore any KNN-related parameters.
        When ``knn_index`` is specified, ``data`` must be set to None.

    """

    def __init__(
        self,
        data=None,
        k_neighbors=30,
        method="auto",
        metric="euclidean",
        metric_params=None,
        symmetrize=True,
        n_jobs=1,
        random_state=None,
        verbose=False,
        knn_kwargs=None,
        knn_index=None,
    ):
        # This can't work if neither data nor the knn index are specified
        if data is None and knn_index is None:
            raise ValueError(
                "At least one of the parameters `data` or `knn_index` must be specified!"
            )
        # This can't work if both data and the knn index are specified
        if data is not None and knn_index is not None:
            raise ValueError(
                "Both `data` or `knn_index` were specified! Please pass only one."
            )

        if knn_index is None:
            if k_neighbors >= data.shape[0]:
                raise ValueError(
                    "`k_neighbors` (%d) cannot be larger than N-1 (%d)." %
                    (k_neighbors, data.shape[0])
                )

            self.knn_index = get_knn_index(
                data,
                method,
                k=k_neighbors,
                metric=metric,
                metric_params=metric_params,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                knn_kwargs=knn_kwargs,
            )

        else:
            self.knn_index = knn_index
            log.info("KNN index provided. Ignoring KNN-related parameters.")

        neighbors, distances = self.knn_index.build()

        k_neighbors = self.knn_index.k
        n_samples = self.knn_index.n_samples
        P = sp.csr_matrix(
            (
                np.ones_like(distances).ravel(),
                neighbors.ravel(),
                range(0, n_samples * k_neighbors + 1, k_neighbors),
            ),
            shape=(n_samples, n_samples),
        )

        # Symmetrize the probability matrix
        if symmetrize == "max" or symmetrize is True:
            P = (P + P.T > 0).astype(float)
        elif symmetrize == "mean":
            P = (P + P.T) / 2
        elif symmetrize == "none" or symmetrize is False:
            pass
        else:
            raise ValueError(
                f"Symmetrization method `{symmetrize}` is not recognized."
            )

        # Convert weights to probabilities
        P /= np.sum(P)

        self.P = P
        self.verbose = verbose
        self.n_jobs = n_jobs

    def to_new(self, data, k_neighbors=None, return_distances=False):
        """Compute the affinities of new samples to the initial samples.

        This is necessary for embedding new data points into an existing
        embedding.

        Parameters
        ----------
        data: np.ndarray
            The data points to be added to the existing embedding.

        k_neighbors: int
            The number of nearest neighbors to consider.

        return_distances: bool
            If needed, the function can return the indices of the nearest
            neighbors and their corresponding distances.

        Returns
        -------
        P: array_like
            An :math:`N \\times M` affinity matrix expressing interactions
            between :math:`N` new data points the initial :math:`M` data
            samples.

        indices: np.ndarray
            Returned if ``return_distances=True``. The indices of the :math:`k`
            nearest neighbors in the existing embedding for every new data
            point.

        distances: np.ndarray
            Returned if ``return_distances=True``. The distances to the
            :math:`k` nearest neighbors in the existing embedding for every new
            data point.

        """
        n_samples = data.shape[0]
        n_reference_samples = self.n_samples

        if k_neighbors is None:
            k_neighbors = self.knn_index.k
        elif k_neighbors >= n_reference_samples:
            raise ValueError(
                "`k` (%d) cannot be larger than the number of reference "
                "samples (%d)." % (k_neighbors, self.n_samples)
            )

        # Find nearest neighbors and the distances to the new points
        neighbors, distances = self.knn_index.query(data, k_neighbors)

        values = np.ones_like(distances)
        values /= np.sum(values, axis=1)[:, np.newaxis]

        P = sp.csr_matrix(
            (
                values.ravel(),
                neighbors.ravel(),
                range(0, n_samples * k_neighbors + 1, k_neighbors),
            ),
            shape=(n_samples, n_reference_samples),
        )

        if return_distances:
            return P, neighbors, distances

        return P


class PrecomputedAffinities(Affinities):
    """Use a precomputed affinity matrix.

    Parameters
    ----------
    affinities: scipy.sparse.csr_matrix, np.ndarray
        An N x N matrix containing the affinities.
    normalize: bool
        Normalize the affinity matrix to sum to 1. Default is True.

    """

    def __init__(self, affinities, normalize=True):
        if not isinstance(affinities, sp.csr_matrix):
            affinities = sp.csr_matrix(affinities)
        if normalize:
            affinities /= np.sum(affinities)
        self.P = affinities

    def to_new(self, data, return_distances=False):
        raise RuntimeError("Precomputed affinity matrices cannot be queried.")
