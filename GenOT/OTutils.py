import numpy as np
import ot
import anndata as ad
import torch
import warnings
warnings.filterwarnings("ignore")
from scipy.sparse import issparse
from scipy.spatial import distance_matrix


def compute_spatial_barycenter(adata1, adata2, weight1=0.5, alpha=0.5,
                               num_barycenters=5000,
                               max_iter=10, tol=1e-1):
    """
    Compute fused Gromov-Wasserstein spatial barycenter with optimized initialization and structure matrix construction.

    Parameters:
        adata1, adata2: AnnData objects containing spatial coordinates (obsm['spatial'])
        weight1: float (0-1), weight for dataset 1 in barycenter computation
        alpha: float (0-1), balance parameter between structure (Cs) and features (Ys)
        num_barycenters: int, number of points in target barycenter
        max_iter: int, maximum iterations for optimization
        tol: float, convergence threshold

    Returns:
        tuple: (spatial_bary, C_bary, transport_plans) - Spatial coordinates of the computed barycenter,
               its optimized structure matrix, and a list of optimal transport plans.
    """

    # 1. Data extraction and validation
    Ys = [
        adata1.obsm['spatial'].astype(np.float64),
        adata2.obsm['spatial'].astype(np.float64)
    ]

    Cs = [
        distance_matrix(adata1.obsm['spatial'], adata1.obsm['spatial']),
        distance_matrix(adata2.obsm['spatial'], adata2.obsm['spatial'])
    ]

    # 2. Initialization
    np.random.seed(0)
    init_Y = np.random.randn(num_barycenters, Ys[0].shape[1])
    init_C = distance_matrix(init_Y, init_Y)

    # 3. Compute barycenter using Fused Gromov-Wasserstein
    Y_bary, C_bary, log = ot.gromov.fgw_barycenters(
        N=num_barycenters,
        Ys=Ys,
        Cs=Cs,
        ps=None,
        p=None,
        lambdas=[weight1, 1 - weight1],
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        init_C=init_C,
        init_X=init_Y,
        stop_criterion='barycenter',
        verbose=True,
        log=True
    )

    spatial_bary = Y_bary
    transport_plans = log['T']  # Extract the transport plans

    return spatial_bary, transport_plans


def compute_emb_barycenter(adata1, adata2, weight1=0.5, alpha=0.5,
                           num_barycenters=5000,
                           max_iter=10, tol=1e-1):
    """
    Compute fused Gromov-Wasserstein barycenter embeddings for two datasets.

    Args:
        adata1, adata2: AnnData objects containing biological data
        weight1: Weight (0-1) for the first dataset in barycenter computation
        alpha: Trade-off parameter (0-1) between structure and feature alignment
               (0 = pure Gromov-Wasserstein, 1 = pure Wasserstein)
        num_barycenters: Number of points in the barycenter
        max_iter: Maximum number of iterations for optimization
        tol: Tolerance threshold for convergence

    Returns:
        Y_bary: Numpy array of fused barycenter embeddings
    """

    # Extract embeddings from both datasets and convert to float64 for numerical stability
    Ys = [
        adata1.obsm['emb'].astype(np.float64),
        adata2.obsm['emb'].astype(np.float64)
    ]

    # Compute pairwise distance matrices for each dataset's embeddings
    Cs = [
        distance_matrix(adata1.obsm['emb'].astype(np.float64), adata1.obsm['emb'].astype(np.float64)),
        distance_matrix(adata2.obsm['emb'].astype(np.float64), adata2.obsm['emb'].astype(np.float64))
    ]

    # Initialize random barycenter embeddings and distance matrix for optimization
    np.random.seed(0)  # Seed for reproducibility
    init_Y = np.random.randn(num_barycenters, Ys[0].shape[1])  # Random initial embeddings
    init_C = distance_matrix(init_Y, init_Y)  # Initial distance matrix

    # Compute fused Gromov-Wasserstein barycenter using POT library
    Y_bary, C_bary, log = ot.gromov.fgw_barycenters(
        N=num_barycenters,
        Ys=Ys,
        Cs=Cs,
        ps=None,  # Use uniform distribution for input datasets
        p=None,  # Use uniform distribution for barycenter
        lambdas=[weight1, 1 - weight1],  # Dataset weights
        alpha=alpha,  # Structure-feature tradeoff
        max_iter=max_iter,
        tol=tol,
        init_C=init_C,  # Initial distance matrix
        init_X=init_Y,  # Initial embeddings
        stop_criterion='barycenter',  # Stopping criterion type
        verbose=True,  # Print progress
        log=True  # Return optimization log
    )
    transport_plans = log['T']
    return Y_bary, transport_plans


def compute_transport_plan_lp(X, barycenter):
    """
    Compute optimal transport plan using Linear Programming (Earth Mover's Distance).

    Parameters:
    X (np.ndarray): Source dataset coordinates (n_points_X, dim)
    barycenter (np.ndarray): Target barycenter coordinates (n_points_barycenter, dim)

    Returns:
    np.ndarray: Optimal transport plan matrix (n_points_X, n_points_barycenter)
    """
    n_points_X = X.shape[0]
    n_points_barycenter = barycenter.shape[0]

    # Uniform distributions for source and target
    a = np.ones(n_points_X) / n_points_X  # Source distribution
    b = np.ones(n_points_barycenter) / n_points_barycenter  # Target distribution

    # Compute pairwise Euclidean distance cost matrix
    cost_matrix = ot.dist(X, barycenter)

    # Solve optimal transport problem using EMD
    transport_plan = ot.emd(a, b, cost_matrix)

    return transport_plan


def compute_cost_matrix(P1b_s, P1b_z, P2b_s, P2b_z):
    """
    Compute the cost matrix generated from the transport plan according to the formula .
    Returns a 2D cost matrix where each element represents the cost between spatial coordinates and embedded coordinates.
    """
    # Calculate pairwise Euclidean distances between two 2D transport plans
    cost_matrix_spatial = ot.dist(P1b_s, P1b_z)
    cost_matrix_embedding = ot.dist(P2b_s, P2b_z)

    # Combine spatial cost and embedding cost to return the final cost matrix
    cost_matrix = cost_matrix_spatial + cost_matrix_embedding

    return cost_matrix


def compute_optimal_matching(Xb_s, Xb_e, cost_matrix):
    """
    Find optimal embedding indices matching spatial coordinates.

    Parameters:
    Xb_s (np.ndarray): Spatial barycenter coordinates (n_points, spatial_dim)
    Xb_e (np.ndarray): Embedding barycenter coordinates (n_points, embed_dim)
    cost_matrix (np.ndarray): Precomputed cost matrix

    Returns:
    np.ndarray: Optimal embedding indices for each spatial point
    """
    n_src = Xb_s.shape[0]
    n_tgt = Xb_e.shape[0]

    # Uniform distributions
    a = np.ones(n_src) / n_src  # Source distribution
    b = np.ones(n_tgt) / n_tgt  # Target distribution

    # Compute optimal transport plan
    transport_plan = ot.emd(a, b, cost_matrix)

    # Find max-weight connections for embedding matching
    optimal_embed_indices = np.argmax(transport_plan, axis=1)

    return optimal_embed_indices


def update_embedding_barycenter(Xb_s, Xb_e, s_transport_plans, e_transport_plans ):
    """
    Update embedding barycenter through iterative optimal transport alignment.

    Parameters:
    Xb_s (np.ndarray): Current spatial barycenter
    Xb_e (np.ndarray): Current embedding barycenter
    adata1 (AnnData): Dataset 1 with spatial/embedding data
    adata2 (AnnData): Dataset 2 with spatial/embedding data

    Returns:
    tuple: Updated embedding and spatial barycenters
    """


    P1b_s = s_transport_plans[0]  # spatial bary -> Dataset1
    P2b_s = s_transport_plans[1]  # spatial bary -> Dataset2

    P1b_z = e_transport_plans[0]  # embed bary -> Emb1
    P2b_z = e_transport_plans[1]  # embed bary -> Emb2

    # Calculate combined cost matrix
    cost_matrix = compute_cost_matrix(P1b_s, P1b_z, P2b_s, P2b_z)
    # Find optimal matches
    embed_matches = compute_optimal_matching(Xb_s, Xb_e, cost_matrix)

    # Update barycenters using matches
    updated_embed_bary = Xb_e[embed_matches, :]

    return updated_embed_bary


def create_mapped_adata(
        source_adata: ad.AnnData,
        target_adata: ad.AnnData,
        spatial_key: str = 'spatial',
        threshold_denominator: float = 2.0
) -> ad.AnnData:
    """
    Create a minimal mapped AnnData containing only X, spatial coords, obs, and var

    Parameters:
        source_adata (AnnData): Source dataset with expression data
        target_adata (AnnData): Target dataset with spatial coordinates
        spatial_key (str): Key name for spatial coordinates (default: 'spatial')
        threshold_denominator (float): Denominator used to compute threshold for zeroing values.
            Threshold per gene is `max_expression / denominator` (default: 2.0).
            Must be greater than 0.

    Returns:
        AnnData: New adata with:
            - X: Mapped expression matrix
            - obsm[spatial_key]: Target spatial coordinates
            - obs: Target observation metadata
            - var: Source variable metadata

    Example:
        mapped_adata = create_mapped_adata(adata1, adata3)
        mapped_adata = create_mapped_adata(adata1, adata3, threshold_denominator=3.0)
    """
    if threshold_denominator <= 0:
        raise ValueError("threshold_denominator must be greater than 0")

    # Compute transport plan
    transport_plan = compute_transport_plan_lp(
        source_adata.obsm[spatial_key],
        target_adata.obsm[spatial_key]
    )

    # Normalize transport matrix
    transport_plan = transport_plan / transport_plan.sum(axis=1, keepdims=True)

    # Matrix projection handling sparse/dense
    if issparse(source_adata.X):
        mapped_X = transport_plan.T @ source_adata.X.toarray()
    else:
        mapped_X = transport_plan.T @ source_adata.X

    thresholds = mapped_X.max(axis=0) / threshold_denominator

    for i, threshold in enumerate(thresholds):
        mapped_X[:, i][mapped_X[:, i] < threshold] = 0

    mapped_X[mapped_X < 0] = 0

    return ad.AnnData(
        X=mapped_X,
        obs=source_adata.obs.copy(),
        var=source_adata.var.copy(),
        obsm={spatial_key: target_adata.obsm[spatial_key].copy()},
        uns={},
        obsp={},
        layers={}
    )
