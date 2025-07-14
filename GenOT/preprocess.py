import os
import ot
import torch
import random
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch.nn.functional as F
from torch.backends import cudnn
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.sparse import csc_matrix, csr_matrix

def feature_reconstruct_loss(embd: torch.Tensor,
                             x: torch.Tensor,
                             recon_model: torch.nn.Module,
                             mse_weight: float = 1.0,
                             ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the feature reconstruction loss using Mean Squared Error (MSE).

    Parameters
    ----------
    embd : torch.Tensor
        Embedding of a cell or spatial spot.
    x : torch.Tensor
        Target input data (e.g., gene expression, protein expression, or other features)
        that the `recon_model` attempts to reconstruct from `embd`.
    recon_model : torch.nn.Module
        A reconstruction model (typically a decoder) that takes the embedding `embd`
        as input and outputs a reconstructed version of `x`.
    mse_weight : float, default 1.0
        Weight for the MSE loss term. This allows scaling the contribution of the
        reconstruction loss in a combined loss function.

    Returns
    -------
    recon_x : torch.Tensor
        The reconstructed output from the decoder, which is an approximation of `x`.
    loss : torch.Tensor
        The computed reconstruction loss value (weighted MSE loss).
    """
    # Ensure `x` is a tensor and moved to the same device as `embd`
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32).to(embd.device)
    else:
        x = x.to(embd.device)

    # Forward pass through the reconstruction model to get the reconstructed features
    recon_x = recon_model(embd)

    # Calculate Mean Squared Error loss between the reconstructed features and the target features
    mse_loss_value = F.mse_loss(recon_x, x)

    # Combine losses (currently only MSE, but allows for expansion)
    loss = mse_weight * mse_loss_value

    return recon_x, loss


def permutation(feature: np.ndarray) -> np.ndarray:
    """
    Randomly permutes the rows (samples) of a given feature matrix.

    This is often used to create a negative sample or a shuffled version of data
    for tasks like contrastive learning, where the relationship between original
    features and permuted features is broken.

    Parameters
    ----------
    feature : np.ndarray
        The input feature matrix, where each row represents a sample/cell/spot
        and columns represent features.

    Returns
    -------
    feature_permutated : np.ndarray
        A new feature matrix with its rows randomly permuted.
    """
    # Generate an array of indices from 0 to the number of rows in the feature matrix
    ids = np.arange(feature.shape[0])
    # Randomly permute these indices
    ids = np.random.permutation(ids)
    # Use the permuted indices to reorder the rows of the feature matrix
    feature_permutated = feature[ids]

    return feature_permutated


def construct_interaction(adata: 'anndata.AnnData', n_neighbors: int = 3):
    """
    Constructs a spot-to-spot interaction graph based on spatial proximity
    using Euclidean distance and a fixed number of nearest neighbors (KNN).
    The adjacency matrix is stored in `adata.obsm['adj']` and `adata.obsm['graph_neigh']`.

    This function uses `ot.dist` for distance calculation, which might be
    computationally intensive for very large datasets.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing spatial coordinates in `adata.obsm['spatial']`.
    n_neighbors : int, default 3
        The number of nearest neighbors to consider for each spot when constructing
        the interaction graph.
    """
    # Get spatial coordinates of the spots
    position = adata.obsm['spatial']
    n_spot = position.shape[0]

    # Calculate the Euclidean distance matrix between all spots
    distance_matrix = ot.dist(position, position, metric='euclidean')
    adata.obsm['distance_matrix'] = distance_matrix

    # Initialize an empty interaction matrix
    interaction = np.zeros([n_spot, n_spot])
    # For each spot, find its n_neighbors closest neighbors and set interaction to 1
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        # Get the indices that would sort the distance vector (closest first)
        distance = vec.argsort()
        # Iterate from the 1st neighbor (index 1 as index 0 is self) up to n_neighbors
        for t in range(1, n_neighbors + 1):
            y = distance[t]  # Get the index of the t-th nearest neighbor
            interaction[i, y] = 1  # Set interaction to 1 for this neighbor

    adata.obsm['graph_neigh'] = interaction

    # Transform the adjacency matrix to be symmetrical
    adj = interaction
    adj = adj + adj.T  # Add its transpose to make it symmetrical
    adj = np.where(adj > 1, 1, adj)  # Ensure values are binary (0 or 1)

    adata.obsm['adj'] = adj


def construct_interaction_KNN(adata: 'anndata.AnnData', n_neighbors: int = 3):
    """
    Constructs a spot-to-spot interaction graph using scikit-learn's NearestNeighbors.
    This method is generally more efficient than `construct_interaction` for large datasets.
    The adjacency matrix is stored in `adata.obsm['adj']` and `adata.obsm['graph_neigh']`.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing spatial coordinates in `adata.obsm['spatial']`.
    n_neighbors : int, default 3
        The number of nearest neighbors to consider for each spot when constructing
        the interaction graph.
    """
    # Get spatial coordinates of the spots
    position = adata.obsm['spatial']
    n_spot = position.shape[0]

    # Use NearestNeighbors to efficiently find the k-nearest neighbors
    # n_neighbors + 1 because the first neighbor will be the point itself
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(position)
    # Get distances and indices of the k-nearest neighbors
    _, indices = nbrs.kneighbors(position)

    # Prepare indices for constructing the sparse interaction matrix
    # x contains the row indices (source nodes)
    x = indices[:, 0].repeat(n_neighbors)
    # y contains the column indices (target nodes, excluding self)
    y = indices[:, 1:].flatten()

    # Initialize an empty interaction matrix
    interaction = np.zeros([n_spot, n_spot])
    # Set interaction to 1 for k-nearest neighbors
    interaction[x, y] = 1

    adata.obsm['graph_neigh'] = interaction

    # Transform the adjacency matrix to be symmetrical
    adj = interaction
    adj = adj + adj.T  # Add its transpose to make it symmetrical
    adj = np.where(adj > 1, 1, adj)  # Ensure values are binary (0 or 1)

    adata.obsm['adj'] = adj
    print('Graph constructed!')


def preprocess(adata: 'anndata.AnnData'):
    """
    Performs standard preprocessing steps on single-cell or spatial transcriptomics data
    stored in an AnnData object.

    The steps include:
    1. Normalization to total count per cell/spot.
    2. Log-transformation (log1p).
    3. Scaling features to unit variance (with zero_center=False and max_value=10).

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing the raw count data in `adata.X`.
    """
    # Normalize total counts per cell/spot to 1e4
    sc.pp.normalize_total(adata, target_sum=1e4)
    # Log-transform the data (log(1+x))
    sc.pp.log1p(adata)
    # Scale features to a max value of 10, without centering (maintains sparsity for sparse matrices)
    sc.pp.scale(adata, zero_center=False, max_value=10)


def get_feature(adata: 'anndata.AnnData', deconvolution: bool = False,
                n_components: int = 200, use_pca: bool = True):
    """
    Extracts features from an AnnData object, optionally performs PCA,
    and generates a permuted version of the features.
    The original and permuted features are stored in `adata.obsm['feat']`
    and `adata.obsm['feat_a']`, respectively.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing the feature data in `adata.X`.
    deconvolution : bool, default False
        A flag that, if True, implies `adata` might already be processed
        for deconvolution and its `X` might represent proportions or similar.
        If False, a copy is made before processing.
    n_components : int, default 200
        The number of principal components to keep if `use_pca` is True.
    use_pca : bool, default True
        If True, Principal Component Analysis (PCA) will be applied to reduce
        the dimensionality of the features.

    Returns
    -------
    adata : anndata.AnnData
        The AnnData object updated with original and permuted features.
    """
    if deconvolution:
        # If deconvolution is true, assume adata is already in the desired format
        adata_vars = adata
    else:
        # Otherwise, create a copy to avoid modifying the original adata.X if not desired
        adata_vars = adata.copy()

    # Convert sparse matrix to dense array if necessary
    if isinstance(adata_vars.X, (csc_matrix, csr_matrix)):
        raw_feat = adata_vars.X.toarray()
    else:
        raw_feat = adata_vars.X

    feat = raw_feat.copy()

    # Apply PCA if requested
    if use_pca:
        pca = PCA(n_components=n_components)
        feat = pca.fit_transform(feat)

    # Generate a permuted version of the features
    feat_a = permutation(feat)

    # Store the original and permuted features in adata.obsm
    adata.obsm['feat'] = feat
    adata.obsm['feat_a'] = feat_a

    return adata


def get_feature2(adata: 'anndata.AnnData', pca_model: PCA):
    """
    Transforms features of an AnnData object using an already fitted PCA model
    and generates a permuted version of the transformed features.
    The transformed and permuted features are stored in `adata.obsm['feat']`
    and `adata.obsm['feat_a']`, respectively.

    This function is useful when applying a PCA model fitted on a training set
    to new data (e.g., a test set or different batches).

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing the feature data in `adata.X`.
    pca_model : sklearn.decomposition.PCA
        An already fitted PCA model.

    Returns
    -------
    adata : anndata.AnnData
        The AnnData object updated with transformed and permuted features.
    """
    adata_vars = adata.copy()

    # Convert sparse matrix to dense array if necessary
    if isinstance(adata_vars.X, (csc_matrix, csr_matrix)):
        raw_feat = adata_vars.X.toarray()
    else:
        raw_feat = adata_vars.X

    feat = raw_feat.copy()
    pca = pca_model  # Use the provided PCA model
    feat = pca.transform(feat)  # Transform the features using the fitted PCA model

    # Generate a permuted version of the transformed features
    feat_a = permutation(feat)

    # Store the transformed and permuted features in adata.obsm
    adata.obsm['feat'] = feat
    adata.obsm['feat_a'] = feat_a

    return adata


def add_contrastive_label(adata: 'anndata.AnnData'):
    """
    Adds contrastive learning labels to the AnnData object.
    It creates a label matrix `adata.obsm['label_CSL']` where the first column
    is all ones (representing positive samples) and the second column is all zeros
    (representing negative samples). This is typically used in a contrastive setting
    where original samples are positive and permuted/augmented samples are negative.

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object to which the contrastive labels will be added.
    """
    n_spot = adata.n_obs  # Number of observations (cells/spots)
    one_matrix = np.ones([n_spot, 1])  # Column vector of ones
    zero_matrix = np.zeros([n_spot, 1])  # Column vector of zeros
    # Concatenate to form a 2-column label matrix
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm['label_CSL'] = label_CSL


def normalize_adj(adj: np.ndarray) -> np.ndarray:
    """
    Symmetrically normalizes an adjacency matrix using the degree matrix.
    The normalization formula is $D^{-0.5} A D^{-0.5}$, where $A$ is the adjacency
    matrix and $D$ is the degree matrix. This is a common preprocessing step for GCNs.

    Parameters
    ----------
    adj : np.ndarray
        The input adjacency matrix (can be dense or sparse).

    Returns
    -------
    adj : np.ndarray
        The symmetrically normalized adjacency matrix.
    """
    # Convert to sparse COO matrix for efficient computation
    adj = sp.coo_matrix(adj)
    # Calculate row sums (degrees)
    rowsum = np.array(adj.sum(1))
    # Calculate inverse square root of degrees
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    # Handle cases where degree is zero (to avoid division by zero)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # Create a sparse diagonal matrix from the inverse square roots
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # Perform the symmetric normalization: D^(-0.5) * A * D^(-0.5)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()


def preprocess_adj(adj: np.ndarray) -> np.ndarray:
    """
    Preprocesses an adjacency matrix for use in a Graph Convolutional Network (GCN) model.
    It adds an identity matrix to the adjacency matrix (self-loops) and then
    symmetrically normalizes it.

    Parameters
    ----------
    adj : np.ndarray
        The input adjacency matrix.

    Returns
    -------
    adj_normalized : np.ndarray
        The preprocessed and normalized adjacency matrix, including self-loops.
    """
    # Add self-loops to the adjacency matrix (A_hat = A + I)
    adj_normalized = normalize_adj(adj) + np.eye(adj.shape[0])
    return adj_normalized


def sparse_mx_to_torch_sparse_tensor(sparse_mx: sp.spmatrix) -> torch.sparse.FloatTensor:
    """
    Converts a SciPy sparse matrix to a PyTorch sparse tensor.

    Parameters
    ----------
    sparse_mx : scipy.sparse.spmatrix
        The input SciPy sparse matrix (e.g., coo_matrix, csr_matrix).

    Returns
    -------
    torch.sparse.FloatTensor
        The converted PyTorch sparse tensor.
    """
    # Ensure the sparse matrix is in COO format for easy conversion
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    # Extract row and column indices
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    # Extract values
    values = torch.from_numpy(sparse_mx.data)
    # Get the shape of the sparse matrix
    shape = torch.Size(sparse_mx.shape)
    # Create a PyTorch sparse tensor
    return torch.sparse.FloatTensor(indices, values, shape)


def preprocess_adj_sparse(adj: np.ndarray) -> torch.sparse.FloatTensor:
    """
    Preprocesses an adjacency matrix for GCNs and converts it to a PyTorch sparse tensor.
    This function handles the addition of self-loops and symmetric normalization,
    and ensures the output is in a sparse PyTorch tensor format suitable for GCN layers.

    Parameters
    ----------
    adj : np.ndarray
        The input adjacency matrix (dense or sparse).

    Returns
    -------
    torch.sparse.FloatTensor
        The preprocessed and normalized adjacency matrix as a PyTorch sparse tensor.
    """
    # Convert to sparse COO matrix
    adj = sp.coo_matrix(adj)
    # Add self-loops (A_hat = A + I)
    adj_ = adj + sp.eye(adj.shape[0])
    # Calculate row sums (degrees)
    rowsum = np.array(adj_.sum(1))
    # Calculate inverse square root of degrees, handling zero degrees
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    # Perform symmetric normalization: D^(-0.5) * A_hat * D^(-0.5)
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # Convert the normalized sparse matrix to a PyTorch sparse tensor
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def fix_seed(seed: int):
    """
    Sets the random seed for reproducibility across multiple libraries.

    This function ensures that results obtained from operations involving randomness
    (e.g., neural network initialization, data shuffling) are consistent
    when the same seed is used.

    Parameters
    ----------
    seed : int
        The integer seed to set for all random number generators.
    """
    # Set seed for Python's hash function (important for some hash-based operations)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Set seed for Python's built-in random module
    random.seed(seed)
    # Set seed for NumPy's random number generator
    np.random.seed(seed)
    # Set seed for PyTorch on CPU
    torch.manual_seed(seed)
    # Set seed for PyTorch on GPU (if available)
    torch.cuda.manual_seed(seed)
    # Set seed for all PyTorch GPUs (if multiple are available)
    torch.cuda.manual_seed_all(seed)
    # Ensure that cuDNN operations are deterministic (may impact performance slightly)
    cudnn.deterministic = True
    # Disable cuDNN benchmarking (can also impact performance but ensures determinism)
    cudnn.benchmark = False

    # Setting CUBLAS_WORKSPACE_CONFIG for deterministic CUDA operations (for newer PyTorch versions)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
