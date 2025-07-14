import numpy as np
import pandas as pd
from sklearn import metrics
import scanpy as sc
import ot
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, SpectralClustering, Birch
from sklearn.mixture import GaussianMixture
import os
import seaborn as sns
import anndata as ad
from paste2 import PASTE2, projection
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import torch
from somde import SomNode
import warnings
from scipy.optimize import linear_sum_assignment
warnings.filterwarnings("ignore")
import plotly.graph_objects as go
from scipy.sparse import issparse
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance_matrix
from scipy.spatial import Delaunay
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def clustering(adata, n_clusters=7, radius=50, search=True, method='mclust', start=0.1, end=3.0, increment=0.01,
               refinement=False):
    """\
    Spatial clustering based the learned representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    radius : int, optional
        The number of neighbors considered during refinement. The default is 50.
    key : string, optional
        The key of the learned representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'. 
    start : float
        The start value for searching. The default is 0.1.
    end : float 
        The end value for searching. The default is 3.0.
    increment : float
        The step size to increase. The default is 0.01.   
    refinement : bool, optional
        Refine the predicted labels or not. The default is False.

    Returns
    -------
    None.

    """

    pca = PCA(n_components=16, random_state=42)
    embedding = pca.fit_transform(adata.obsm['emb'].copy())
    adata.obsm['emb_pca'] = embedding

    if method == 'mclust':
        adata = mclust_R(adata, used_obsm='emb_pca', num_cluster=n_clusters)
        adata.obs['domain'] = adata.obs['mclust'].astype(str)
    elif method == 'leiden':
        sc.pp.neighbors(adata, use_rep='emb_pca', random_state=666)
        if search:
            res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
            sc.tl.leiden(adata, random_state=0, resolution=res)
        else:
            sc.tl.leiden(adata, random_state=0, resolution=0.3)
        adata.obs['domain'] = adata.obs['leiden']
    elif method == 'louvain':
        sc.pp.neighbors(adata, use_rep='emb_pca', random_state=666)
        if search:
            res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
            sc.tl.louvain(adata, random_state=0, resolution=res)
        else:
            sc.tl.louvain(adata, random_state=0, resolution=0.3)
        adata.obs['domain'] = adata.obs['louvain']
    elif method == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        adata.obs['domain']  = kmeans.fit_predict(adata.obsm['emb_pca']).astype(str)
    elif method == 'minibatch_kmeans':
        minibatch_kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0)
        adata.obs['domain']  = minibatch_kmeans.fit_predict(adata.obsm['emb_pca']).astype(str)
    elif method == 'agglomerative':
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
        adata.obs['domain']  = agglomerative.fit_predict(adata.obsm['emb_pca']).astype(str)
    elif method == 'spectral':
        spectral = SpectralClustering(n_clusters=n_clusters, random_state=0, affinity='nearest_neighbors')
        adata.obs['domain']  = spectral.fit_predict(adata.obsm['emb_pca']).astype(str)
    elif method == 'birch':
        birch = Birch(n_clusters=n_clusters)
        adata.obs['domain']  = birch.fit_predict(adata.obsm['emb_pca']).astype(str)
    elif method == 'gaussian_mixture':
        gmm = GaussianMixture(n_components=n_clusters, random_state=0)
        adata.obs['domain']  = gmm.fit_predict(adata.obsm['emb_pca']).astype(str)

    if refinement:
        new_type = refine_label(adata, radius, key='domain')
        adata.obs['domain'] = new_type


def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    # adata.obs['label_refined'] = np.array(new_type)

    return new_type


def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    '''\
    Searching corresponding resolution according to given cluster number
    
    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.    
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float 
        The end value for searching.
    increment : float
        The step size to increase.
        
    Returns
    -------
    res : float
        Resolution.
        
    '''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label == 1, "Resolution is not found. Please try bigger range or smaller step!."

    return res


# def SpatialDE_svg(adata,
#                subsample_n=None,
#                qval_threshold=0.0000001,
#                save=False,
#                base_name=None):
#     """
#     Detect spatially variable genes (SVGs) using SpatialDE
#
#     Parameters:
#     adata -- AnnData object containing spatial transcriptomics data
#     subsample_n -- Number of cells to subsample (None for no subsampling)
#     qval_threshold -- Significance threshold for q-values (default: 0.05)
#     save -- Whether to save results to file (default: False)
#     base_name -- Base name for output files (required if save=True)
#
#     Returns:
#     List of significant gene names
#     """
#     # Create copy to avoid modifying original object
#     adata = adata.copy()
#
#     # Subsampling
#     if subsample_n is not None:
#         adata = sc.pp.subsample(adata, n_obs=subsample_n, random_state=0, copy=True)
#
#     try:
#         # Extract and filter expression matrix
#         expr_matrix = adata.to_df()
#         expression_data = expr_matrix.loc[:, expr_matrix.sum(0) >= 3]  # Filter low-expressed genes
#         expression_data = expression_data.astype('float32')  # Reduce memory usage
#
#         # Extract spatial coordinates
#         spatial_coords = pd.DataFrame(adata.obsm['spatial'],
#                                       columns=['x', 'y'],
#                                       index=adata.obs_names)
#
#         # Stabilize variance using Anscombe transformation
#         norm_expr = NaiveDE.stabilize(expression_data).T
#
#         # Add total counts for normalization
#         spatial_coords['total_counts'] = expression_data.sum(1).values
#
#         # Regress out technical effects
#         resid_expr = NaiveDE.regress_out(spatial_coords,
#                                          norm_expr,
#                                          'np.log(total_counts)').T
#
#         # Run SpatialDE analysis
#         results = SpatialDE.run(
#             spatial_coords[['x', 'y']].values.astype('float32'),
#             resid_expr
#         )
#
#         # Filter significant results
#         significant_genes = results[results['qval'] < qval_threshold]
#         gene_names = significant_genes['g'].tolist()
#
#         # Save results if requested
#         if save:
#             if not base_name:
#                 raise ValueError("base_name must be specified when save=True")
#             output_path = f"{base_name}_hvg_gene_names(SpatialDE).txt"
#             with open(output_path, "w") as f:
#                 f.write("\n".join(gene_names))
#             print(f"Results saved to: {output_path}")
#
#         return gene_names
#
#     except KeyError as e:
#         print(f"Missing required data: {str(e)}")
#         return []
#     except Exception as e:
#         print(f"Analysis failed: {str(e)}")
#         return []
#





# def SpaGCN_svg(adata, annotation_name='Ground Truth',min_in_group_fraction=0.5, min_in_out_group_ratio=1.2, min_fold_change=1.1):
#     """
#     Identify Spatially Variable Genes (SVGs) using SpaGCN algorithm
#
#     Parameters:
#     adata (AnnData): Annotated data matrix with spatial coordinates and domain annotations
#     min_in_group_fraction (float): Minimum fraction of cells expressing the gene in target domain
#     min_in_out_group_ratio (float): Minimum expression ratio between target and neighboring domains
#     min_fold_change (float): Minimum fold change for differential expression
#
#     Returns:
#     list: Significant spatially variable genes meeting all criteria
#     """
#
#     # Create a working copy to preserve original data
#     raw = adata.copy()
#     raw.var_names_make_unique()
#
#     # Prepare domain annotations as categorical variable
#     raw.obs["pred"] = adata.obs[annotation_name].astype('category')
#
#     # Extract spatial coordinates from adata
#     spatial_coords = adata.obsm['spatial']
#     raw.obs["x_array"] = spatial_coords[:, 0]  # X-coordinates
#     raw.obs["y_array"] = spatial_coords[:, 1]  # Y-coordinates
#
#     # Convert sparse matrix to dense and apply log normalization
#     raw.X = raw.X.A if issparse(raw.X) else raw.X
#     sc.pp.log1p(raw)  # Log-transform normalized counts
#
#     # Calculate adjacency matrix for spatial graph construction
#     adj_2d = spg.calculate_adj_matrix(x=raw.obs["x_array"],
#                                       y=raw.obs["y_array"],
#                                       histology=False)
#
#     # Determine adaptive distance thresholds for neighborhood detection
#     non_zero_adj = adj_2d[adj_2d != 0]
#     start, end = np.quantile(non_zero_adj, q=0.001), np.quantile(non_zero_adj, q=0.1)
#
#     svg_gene_set = set()  # Using set to avoid duplicate genes
#
#     # Process each biological domain independently
#     for domain in raw.obs['pred'].cat.categories:
#         # Optimize spatial neighborhood radius for current domain
#         optimal_radius = spg.search_radius(
#             target_cluster=domain,
#             cell_id=raw.obs.index.tolist(),
#             x=raw.obs["x_array"].values,
#             y=raw.obs["y_array"].values,
#             pred=raw.obs["pred"].tolist(),
#             start=start,
#             end=end,
#             num_min=10,  # Minimum neighbors required
#             num_max=14,  # Maximum neighbors allowed
#             max_run=100  # Maximum optimization iterations
#         )
#
#         # Identify neighboring domains using optimized radius
#         neighboring_domains = spg.find_neighbor_clusters(
#             target_cluster=domain,
#             cell_id=raw.obs.index.tolist(),
#             x=raw.obs["x_array"].tolist(),
#             y=raw.obs["y_array"].tolist(),
#             pred=raw.obs["pred"].tolist(),
#             radius=optimal_radius,
#             ratio=1 / 2  # Minimum fraction of cells in neighbor regions
#         )[:3]  # Select top 3 neighboring domains
#
#         # Perform differential expression analysis
#         de_results = spg.rank_genes_groups(
#             input_adata=raw,
#             target_cluster=domain,
#             nbr_list=neighboring_domains,
#             label_col="pred",
#             adj_nbr=True,  # Adjust for neighbor domain expression
#             log=True  # Use log-transformed values
#         )
#
#         # Apply multi-criteria filtering for significant SVGs
#         significant_genes = de_results[
#             (de_results["pvals_adj"] < 0.05) &  # Adjusted p-value
#             (de_results["in_out_group_ratio"] > min_in_out_group_ratio) &  # Specificity ratio
#             (de_results["in_group_fraction"] > min_in_group_fraction) &  # Expression prevalence
#             (de_results["fold_change"] > min_fold_change)  # Magnitude of change
#             ].sort_values("in_group_fraction", ascending=False)
#
#         # Aggregate significant genes across domains
#         svg_gene_set.update(significant_genes["genes"].tolist())
#
#     return sorted(svg_gene_set)  # Return sorted unique gene list

def PASTE2_align_spatial_data(
    adata1: sc.AnnData,
    adata2: sc.AnnData,
    overlap_ratio: float = 0.7,
    visualize: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align two spatial transcriptomics slices using PASTE2.

    Parameters:
        adata1 (AnnData): First spatial transcriptomics slice.
        adata2 (AnnData): Second spatial transcriptomics slice.
        overlap_ratio (float): Estimated overlap ratio between slices (0-1).
        visualize (bool): Whether to visualize the aligned slices. Defaults to False.

    Returns:
        tuple: Aligned coordinates (array1, array2) as NumPy arrays.

    Example:
        coord1, coord2 = PASTE2_align_spatial_data(slice1, slice2, overlap_ratio=0.6, visualize=True)
    """
    # Calculate pairwise alignment matrix
    alignment_matrix = PASTE2.partial_pairwise_align(adata1, adata2, s=overlap_ratio)

    # Perform coordinate projection
    aligned_slices = projection.partial_stack_slices_pairwise(
        slices=[adata1.copy(), adata2.copy()],
        pis=[alignment_matrix, ]
    )

    # Visualize aligned slices if requested
    if visualize:
        # Create color mapping for visualization
        layer_to_color_map = {
            f'Layer_{i+1}': sns.color_palette()[i] for i in range(6)
        }
        layer_to_color_map['WM'] = sns.color_palette()[6]

        def plot_slices_overlap(slices, layer_to_color_map=layer_to_color_map):
            """Helper function to visualize aligned slices."""
            plt.figure(figsize=(10, 10))
            for i, adata in enumerate(slices):
                # Map ground truth labels to colors
                colors = adata.obs['Ground Truth'].astype(str).apply(
                    lambda x: layer_to_color_map.get(x, 'gray')
                )
                plt.scatter(
                    adata.obsm['spatial'][:, 0],
                    adata.obsm['spatial'][:, 1],
                    linewidth=0,
                    s=100,
                    marker=".",
                    color=colors
                )
            plt.gca().invert_yaxis()
            plt.axis('off')
            plt.show()

        # Plot the aligned slices
        plot_slices_overlap(aligned_slices)

    # Extract and return aligned coordinates
    return (
        aligned_slices[0].obsm['spatial'].copy(),
        aligned_slices[1].obsm['spatial'].copy()
    )





def calculate_triangle_area(p1, p2, p3):
    """Calculates the area of a triangle given its three 2D vertices."""
    return 0.5 * abs(p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1]))

def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two 2D points."""
    return np.linalg.norm(p1 - p2)

def alpha_shape_area(points, alpha):
    """
    Calculate the area of the Alpha Shape for 2D points.
    This implementation constructs the Delaunay triangulation and sums the areas
    of triangles whose circumradius is less than or equal to alpha.

    Parameters:
    - points: ndarray
        2D spatial coordinates, shape (n_samples, 2)
    - alpha: float
        Alpha parameter for Alpha Shape calculation. Triangles with a circumradius
        larger than alpha will be excluded. Use float('inf') for convex hull-like area.

    Returns:
    - float
        The calculated area of the Alpha Shape.
    """
    if len(points) < 3:
        return 0.0

    try:
        tri = Delaunay(points)
    except Exception: # Handle cases where triangulation might fail (e.g., all points collinear or too few points)
        return 0.0

    total_alpha_area = 0.0

    for simplex_indices in tri.simplices:
        p1 = points[simplex_indices[0]]
        p2 = points[simplex_indices[1]]
        p3 = points[simplex_indices[2]]

        # Calculate edge lengths
        a_len = calculate_distance(p2, p3)
        b_len = calculate_distance(p1, p3)
        c_len = calculate_distance(p1, p2)

        # Calculate triangle area
        triangle_area = calculate_triangle_area(p1, p2, p3)

        if triangle_area < 1e-9: # Avoid division by zero for degenerate triangles
            continue

        # Calculate circumradius
        # R = (a * b * c) / (4 * Area)
        circumradius = (a_len * b_len * c_len) / (4 * triangle_area)

        # If the circumradius is less than or equal to alpha, include this triangle's area
        if circumradius <= alpha:
            total_alpha_area += triangle_area

    return total_alpha_area


def adjust_area(spatial1, spatial2, area_ratio_threshold=0.8, alpha=float('inf')):
    """
    Adjust the area ratio between two spatial datasets to meet a specified threshold.
    Now uses Alpha Shape area calculation.

    Parameters:
    - spatial1: ndarray
        Spatial coordinates of dataset 1, shape (n_samples1, 2)
    - spatial2: ndarray
        Spatial coordinates of dataset 2, shape (n_samples2, 2)
    - area_ratio_threshold: float
        Minimum acceptable ratio between smaller/larger areas, default 0.8
    - alpha: float
        Alpha parameter for Alpha Shape area calculation. Use float('inf') for convex hull area.

    Returns:
    - spatial2_adjusted: ndarray
        Scaled coordinates of spatial2
    - scale_factor: float
        Applied scaling factor for area adjustment
    """
    area1 = alpha_shape_area(spatial1, alpha)
    area2 = alpha_shape_area(spatial2, alpha)

    if area1 == 0 or area2 == 0:
        return spatial2, 1.0

    current_ratio = min(area1, area2) / max(area1, area2)

    if current_ratio >= area_ratio_threshold:
        return spatial2, 1.0

    scale_factor = np.sqrt(area1 * area_ratio_threshold / area2)
    spatial2_adjusted = spatial2 * scale_factor

    return spatial2_adjusted, scale_factor


# --- ICP-like Translation Optimization ---
def find_nearest_neighbors(source_points, target_points):
    """
    For each point in source_points, find its nearest neighbor in target_points.
    Returns the indices of nearest neighbors in target_points.
    """
    # Efficiently compute all pairwise squared distances
    # (source_points[:, np.newaxis, :] - target_points) creates shape (N_src, 1, 2) - (1, N_tgt, 2)
    # which broadcasts to (N_src, N_tgt, 2)
    distances_squared = np.sum((source_points[:, np.newaxis, :] - target_points)**2, axis=2)
    nearest_neighbor_indices = np.argmin(distances_squared, axis=1)
    return nearest_neighbor_indices

def icp_translate(source_points, target_points, max_iterations=100, tolerance=1e-9):
    """
    Performs ICP-like translation optimization.
    It iteratively finds nearest neighbors and applies a translation based on centroid alignment
    of matched points. This version only focuses on translation.

    Parameters:
    - source_points: ndarray, Reference coordinates (e.g., spatial1), shape (n_samples1, 2)
    - target_points: ndarray, Target coordinates to align (e.g., spatial2), shape (n_samples2, 2)
    - max_iterations: int, Maximum number of ICP iterations
    - tolerance: float, Convergence threshold based on change in translation vector magnitude

    Returns:
    - target_points_aligned: ndarray, Translated coordinates of target_points
    - final_translation_vector: ndarray, The total accumulated translation
    """
    current_target_points = np.copy(target_points)
    total_translation = np.array([0.0, 0.0]) # Accumulate translations

    for i in range(max_iterations):
        # Step 1: Find nearest neighbors
        # For each point in current_target_points, find its closest point in source_points
        nearest_neighbor_indices = find_nearest_neighbors(current_target_points, source_points)
        matched_source_points = source_points[nearest_neighbor_indices]

        # Step 2: Calculate centroids of matched points
        centroid_current_target = np.mean(current_target_points, axis=0)
        centroid_matched_source = np.mean(matched_source_points, axis=0)

        # Step 3: Calculate translation to align centroids
        translation_step = centroid_matched_source - centroid_current_target

        # Step 4: Apply translation
        current_target_points += translation_step
        total_translation += translation_step

        # Check for convergence
        if np.linalg.norm(translation_step) < tolerance:
            # print(f"ICP converged after {i+1} iterations.")
            break
    # else:
        # print(f"ICP reached max iterations ({max_iterations}) without converging to tolerance {tolerance}.")

    return current_target_points, total_translation


def align_spatial_coords(spatial1, spatial2, area_ratio_threshold=0.9, alpha=float('inf')):
    """
    Full alignment pipeline: area scaling followed by ICP-like translation optimization.
    Now supports Alpha Shape area calculation and more robust translation.

    Parameters:
    - spatial1: ndarray
        Reference spatial coordinates, shape (n_samples1, 2)
    - spatial2: ndarray
        Target spatial coordinates to align, shape (n_samples2, 2)
    - area_ratio_threshold: float
        Area ratio threshold for initial scaling, default 0.8
    - alpha: float
        Alpha parameter for Alpha Shape area calculation. Use float('inf') for convex hull area.

    Returns:
    - spatial2_aligned: ndarray
        Final aligned coordinates of spatial2
    """

    # Stage 1: Area scaling (remains unchanged)
    spatial2_scaled, scale_factor = adjust_area(
        spatial1, spatial2, area_ratio_threshold, alpha
    )

    # Stage 2: ICP-like Translation optimization
    # spatial1 is the 'reference', spatial2_scaled is the 'moving' point set
    spatial2_aligned, total_translation_params = icp_translate(
        spatial1, spatial2_scaled
    )

    return spatial2_aligned




# def compute_spatial_barycenter(adata1, adata2, weight1=0.5, num_barycenters=5000,
#                                numItermax=20, save_output=False, filename_prefix=None):
#     """
#     Compute spatial barycenter between two datasets with optional result saving.
#
#     Parameters:
#     adata1 (AnnData): First spatial dataset (must contain 'spatial' in obsm)
#     adata2 (AnnData): Second spatial dataset (must contain 'spatial' in obsm)
#     weight1 (float): Weight for first dataset (0-1), default 0.5
#     num_barycenters (int): Number of output barycenter points, default 5000
#     numItermax (int): Maximum optimization iterations, default 20
#     save_output (bool): Save result to .npy file if True, default False
#     filename_prefix (str): Custom prefix for output filename. If None,
#                         uses "spatial_barycenter" when saving
#
#     Returns:
#     np.ndarray: Combined barycenter coordinates (num_barycenters, 2)
#     """
#     weight2 = 1 - weight1
#
#     # Convert to JAX arrays for accelerated computing
#     X1 = jnp.array(adata1.obsm['spatial'])
#     X2 = jnp.array(adata2.obsm['spatial'])
#
#     # Uniform distribution initialization
#     a = jnp.ones(X1.shape[0]) / X1.shape[0]
#     b = jnp.ones(X2.shape[0]) / X2.shape[0]
#
#     # Configure barycenter problem with entropic regularization
#     bar_prob = barycenter_problem.FreeBarycenterProblem(
#         y=jnp.stack([X1, X2]),
#         b=jnp.stack([a, b]),
#         weights=jnp.array([weight1, weight2]),
#         epsilon=10  # Balances accuracy and computational speed
#     )
#
#     # Set up solver with convergence thresholds
#     barycenter_solver = FreeWassersteinBarycenter(
#         min_iterations=10,
#         max_iterations=numItermax,
#         threshold=0.0001
#     )
#
#     # Compute final barycenter
#     Xb_combined = barycenter_solver(
#         bar_prob=bar_prob,
#         bar_size=num_barycenters
#     ).x
#     result = np.array(Xb_combined)
#
#     # Save results if requested
#     if save_output:
#         prefix = filename_prefix or "spatial_barycenter"
#         filename = f"{prefix}({weight1}).npy"
#         np.save(filename, result)
#         print(f"Saved spatial barycenter to {filename}")
#
#     return result


# def compute_emb_barycenter(emb0, emb1, weight1=0.5, num_barycenters=5000,
#                            numItermax=2, save_output=False, filename_prefix=None):
#     """
#     Compute embedding space barycenter with optional result saving.
#
#     Parameters:
#     emb0 (np.ndarray): First dataset embeddings (n_samples, n_features)
#     emb1 (np.ndarray): Second dataset embeddings (n_samples, n_features)
#     weight1 (float): Weight for first embedding (0-1), default 0.5
#     num_barycenters (int): Number of output points, default 5000
#     numItermax (int): Maximum optimization iterations, default 2
#     save_output (bool): Save result to .npy file if True, default False
#     filename_prefix (str): Custom prefix for output filename. If None,
#                         uses "emb_barycenter" when saving
#
#     Returns:
#     np.ndarray: Combined barycenter embeddings (num_barycenters, n_features)
#     """
#     weight2 = 1 - weight1
#
#     # Convert embeddings to JAX arrays
#     X1 = emb0
#     X2 = emb1
#
#     # Initialize uniform distributions
#     a = jnp.ones(X1.shape[0]) / X1.shape[0]
#     b = jnp.ones(X2.shape[0]) / X2.shape[0]
#
#     # Configure embedding space problem
#     bar_prob = barycenter_problem.FreeBarycenterProblem(
#         y=[X1, X2],
#         b=[a, b],
#         weights=jnp.array([weight1, weight2]),
#         epsilon=0.00001  # Tighter regularization for high-D spaces
#     )
#
#     # Faster solver setup for embedding space
#     barycenter_solver = FreeWassersteinBarycenter(
#         min_iterations=1,
#         max_iterations=numItermax,
#         threshold=0.0001
#     )
#
#     # Compute final barycenter
#     Xb_combined = barycenter_solver(
#         bar_prob=bar_prob,
#         bar_size=num_barycenters
#     ).x
#     result = np.array(Xb_combined)
#
#     # Save results if requested
#     if save_output:
#         prefix = filename_prefix or "emb_barycenter"
#         filename = f"{prefix}({weight1}).npy"
#         np.save(filename, result)
#         print(f"Saved embedding barycenter to {filename}")
#
#     return result




def align_cluster_colors(adata, cluster_key: str, groupby: str = 'annotation'):
    """
    Align cluster colors based on marker gene overlap between original annotations and new clusters.

    Parameters:
        adata (AnnData): Annotated data matrix
        cluster_key (str): Key in adata.obs containing cluster labels to align
        groupby (str): Key for original annotation groups (default: 'annotation')

    Returns:
        AnnData: Updated AnnData object with aligned colors in:
            - adata.uns[f'{cluster_key}_colors']
            - adata.uns[f'{cluster_key}_mapping']

    Raises:
        ValueError: If cluster key is missing or cluster counts mismatch

    Process:
        1. Validate input structure
        2. Compute marker genes for both annotations
        3. Create gene overlap matrix
        4. Optimal cluster matching using Hungarian algorithm
        5. Color mapping based on biological correspondence
        6. Data structure updates
    """
    # Input validation
    if cluster_key not in adata.obs:
        raise ValueError(f"Cluster key '{cluster_key}' not found in adata.obs")
    if groupby not in adata.obs:
        raise ValueError(f"Groupby key '{groupby}' not found in adata.obs")

    # Ensure categorical stability
    adata.obs[cluster_key] = adata.obs[cluster_key].astype('category')
    adata.obs[groupby] = adata.obs[groupby].astype('category')

    # Compute differential expression
    grouping =groupby
    sc.tl.rank_genes_groups(
            adata,
            groupby=grouping,
            method='wilcoxon',
            use_raw=False  # Use processed data if available
        )
    groupby_markers=adata.uns['rank_genes_groups']
    grouping = cluster_key
    sc.tl.rank_genes_groups(
        adata,
        groupby=grouping,
        method='wilcoxon',
        use_raw=False  # Use processed data if available
    )
    cluster_key_markers = adata.uns['rank_genes_groups']



    # Marker gene extraction
    def _get_top_markers(rank_dict, n=20):
        return {
            clust: set(rank_dict['names'][clust][:n])
            for clust in rank_dict['names'].dtype.names
        }

    ref_markers = _get_top_markers(groupby_markers)
    clust_markers = _get_top_markers(cluster_key_markers)

    # Overlap matrix construction
    overlap_matrix = pd.DataFrame(
        index=ref_markers.keys(),
        columns=pd.Index(clust_markers.keys(), name=cluster_key),
        dtype=float
    )

    for ref_clust, ref_genes in ref_markers.items():
        for clust, clust_genes in clust_markers.items():
            overlap_matrix.loc[ref_clust, clust] = len(ref_genes & clust_genes)

    # Optimal bijective matching
    cost_matrix = -overlap_matrix.to_numpy()  # Convert to minimization problem
    row_idx, col_idx = linear_sum_assignment(cost_matrix)

    # Create mapping dictionaries
    cluster_to_ref = {
        overlap_matrix.columns[col]: overlap_matrix.index[row]
        for row, col in zip(row_idx, col_idx)
    }
    ref_to_cluster = {v: k for k, v in cluster_to_ref.items()}

    # Color alignment
    ref_colors = dict(zip(
        adata.obs[groupby].cat.categories,
        adata.uns[f'{groupby}_colors']
    ))

    # Validate cluster counts
    n_clust = len(adata.obs[cluster_key].cat.categories)
    n_ref = len(adata.obs[groupby].cat.categories)
    if n_clust != n_ref:
        raise ValueError(
            f"Cluster count mismatch: {n_clust} clusters vs {n_ref} annotations. "
            "Bijective mapping requires equal numbers."
        )

    # Create ordered color list
    aligned_colors = [
        ref_colors[cluster_to_ref[clust]]
        for clust in adata.obs[cluster_key].cat.categories
    ]

    # Update AnnData structure
    adata.uns[f'{cluster_key}_colors'] = aligned_colors
    adata.uns[f'{cluster_key}_mapping'] = cluster_to_ref

    return adata


def normalize_sparse(adata):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    scaler = MaxAbsScaler()
    adata.X = scaler.fit_transform(adata.X)
    return adata



def normalize_dense(adata):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    scaler = MinMaxScaler()
    adata.X = scaler.fit_transform(adata.X)
    return adata


def find_hvg_somde(
        adata,
        layer=None,
        save_output=True,
        filename="significant_genes.txt",
        n_node=20,
        n_retrain=100
):
    """
    Identify highly variable genes using SOMDE method

    Parameters:
    adata : AnnData
        AnnData object containing spatial transcriptomics data
    layer : str, optional (default: None)
        Layer in adata.layers to use, uses adata.X if None
    save_output : bool, optional (default: True)
        Whether to save the results to a text file
    filename : str, optional (default: "significant_genes.txt")
        Output filename for saving results
    n_node : int, optional (default: 20)
        Number of SOM nodes to use
    n_retrain : int, optional (default: 100)
        Number of training iterations

    Returns:
    list
        List of highly variable gene names
    """

    # Ensure unique var_names
    adata.var_names_make_unique(join="++")

    # Extract expression matrix
    if layer is not None:
        data = adata.layers[layer].T
    else:
        data = adata.X.T

    # Create expression DataFrame
    df = pd.DataFrame(
        data.toarray() if hasattr(data, 'toarray') else data,
        index=adata.var_names,
        columns=adata.obs_names
    )

    # Extract spatial coordinates
    try:
        X = adata.obsm['spatial'][:, :2].astype(np.float32)
    except KeyError:
        raise ValueError("Spatial coordinates not found in adata.obsm['spatial']")

    # Initialize and train SOM
    som = SomNode(X, n_node)
    som.reTrain(n_retrain)

    # Process data
    ndf, ninfo = som.mtx(df)
    # nres = som.norm()

    # Run analysis
    result, SVnum = som.run()

    # Extract significant genes
    significant_genes = result.g.head(SVnum).tolist()

    # Save results if requested
    if save_output:
        with open(filename, "w") as f:
            f.write("\n".join(significant_genes))
        print(f"Saved top {SVnum} variable genes to {filename}")

    return significant_genes




def get_unique_marker_genes(adata, annotation_column_name: str, n_top_genes: int = 10):
    """
    Performs gene filtering, runs Scanpy's rank_genes_groups to find marker genes,
    and extracts a unique list of top marker genes for each group.

    Args:
        adata: AnnData object containing your single-cell data.
        annotation_column_name: The name of the column in adata.obs that contains
                                 the cell type or cluster annotations (e.g., 'cell_type', 'leiden').
        n_top_genes: The number of top genes to extract for each annotation group.
                     Defaults to 10.

    Returns:
        A sorted list of unique marker genes.
    """

    # Ensure the annotation column exists
    if annotation_column_name not in adata.obs.columns:
        raise ValueError(f"Error: The annotation column '{annotation_column_name}' was not found in adata.obs.")

    # Filter genes
    sc.pp.filter_genes(adata, min_cells=int(0.2 * adata.n_obs))
    sc.pp.filter_genes(adata, max_cells=int(0.9 * adata.n_obs))

    print(f"Using annotation column '{annotation_column_name}' for marker gene analysis...")
    print(f"Unique values in annotation column: {adata.obs[annotation_column_name].unique().tolist()}")

    # Run rank_genes_groups
    sc.tl.rank_genes_groups(adata,
                            groupby=annotation_column_name,
                            method='wilcoxon',
                            pts=True,
                            key_added='rank_genes_annotation')

    # Extract marker genes
    result = adata.uns['rank_genes_annotation']
    marker_genes_list = []

    # Handle both numpy.recarray and pandas.DataFrame cases for result['names']
    if isinstance(result['names'], np.recarray):
        df_names = pd.DataFrame(result['names'])
    else:
        df_names = pd.DataFrame(result['names'])

    for group_name in df_names.columns:
        top_genes_for_group = list(df_names[group_name][:n_top_genes])
        marker_genes_list.extend(top_genes_for_group)

    # Get unique and sorted marker genes
    unique_marker_genes = sorted(list(set(marker_genes_list)))

    print(f"Extracted {len(unique_marker_genes)} unique marker genes from the top {n_top_genes} genes per annotation group.")
    if not unique_marker_genes:
        print("Warning: No marker genes were extracted. Please check 'n_top_genes' setting and the results of 'rank_genes_groups'.")
    else:
        print(f"First 10 extracted marker genes: {unique_marker_genes[:10]}...")

    return unique_marker_genes





