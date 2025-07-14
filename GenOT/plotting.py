import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors
import scanpy as sc
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.transform import resize
from tqdm import tqdm


def visualize_alignment(spatial1, spatial2_aligned):
    """
    Visualize the alignment results with original and transformed coordinates.

    Parameters:
    - spatial1: ndarray
        Reference coordinates (typically fixed during alignment)
    - spatial2_aligned: ndarray
        Aligned coordinates after transformation
    """

    plt.figure(figsize=(8, 8))
    plt.scatter(
        spatial1[:, 0], spatial1[:, 1],
        c='blue', label='Reference Dataset', alpha=0.6
    )
    plt.scatter(
        spatial2_aligned[:, 0], spatial2_aligned[:, 1],
        c='green', label='Aligned Dataset', alpha=0.6
    )

    plt.title('Spatial Alignment Visualization')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_gene_flow(adatas, gene_name, expr_threshold=0.5):
    """
    Visualize spatial expression patterns and inter-sample connections for a gene across multiple samples.

    Parameters:
        adatas (list): List of AnnData objects containing spatial data
        gene_name (str): Target gene to visualize
        expr_threshold (float): Normalized expression threshold for highlighting (0-1)

    Returns:
        plotly.graph_objects.Figure: Interactive 3D visualization figure
    """

    # Initialize visualization components
    fig = go.Figure()
    z_spacing = 200  # Vertical spacing between sample layers
    all_high_expr_coords = []  # Stores coordinates across samples for connection mapping

    # Process each sample dataset
    for sample_idx, adata in enumerate(adatas):
        # === Data Preparation ===
        # Extract and validate spatial coordinates
        spatial_coords = adata.obsm['spatial']
        if spatial_coords.shape[1] != 2:
            spatial_coords = spatial_coords.T  # Ensure (n_samples, 2) shape

        # Extract gene expression data with sparse matrix handling
        gene_data = adata[:, gene_name].X
        gene_expr = gene_data.toarray().flatten() if hasattr(gene_data, "toarray") else gene_data.flatten()

        # === Background Points ===
        # Create base Z-coordinate for current sample layer
        z_base = sample_idx * z_spacing
        z_all = np.full(spatial_coords.shape[0], z_base)

        # Add semi-transparent background points
        fig.add_trace(go.Scatter3d(
            x=spatial_coords[:, 0],
            y=spatial_coords[:, 1],
            z=z_all,
            mode='markers',
            marker=dict(
                size=2,
                color='rgba(128,128,128,0.8)',  # Semi-transparent grey
                opacity=0.3
            ),
            name=f'Sample {sample_idx + 1}',
            showlegend=False
        ))

        # === High-expression Points ===
        # Normalize expression values for color scaling
        normalized_expr = (gene_expr - gene_expr.min()) / (gene_expr.max() - gene_expr.min() + 1e-8)
        high_expr_mask = normalized_expr > expr_threshold

        # Skip samples without high-expression points
        if not high_expr_mask.any():
            continue

        # Store coordinates for inter-sample connections
        high_expr_coords = spatial_coords[high_expr_mask]
        z_highlight = z_base + 1  # Slight elevation for visual emphasis

        all_high_expr_coords.append(
            np.column_stack([
                high_expr_coords[:, 0],
                high_expr_coords[:, 1],
                np.full(len(high_expr_coords), z_highlight)
            ])
        )

        # Add colored expression points
        fig.add_trace(go.Scatter3d(
            x=high_expr_coords[:, 0],
            y=high_expr_coords[:, 1],
            z=np.full(len(high_expr_coords), z_highlight),
            mode='markers',
            marker=dict(
                size=5,
                color=normalized_expr[high_expr_mask],
                colorscale='Rainbow',
                opacity=0.8,
                colorbar=dict(title='Normalized Expression')
            ),
            name=''
        ))

    # === Inter-sample Connections ===
    for i in range(len(all_high_expr_coords) - 1):
        current_layer = all_high_expr_coords[i]
        next_layer = all_high_expr_coords[i + 1]

        # Find nearest neighbors between consecutive layers
        nn = NearestNeighbors(n_neighbors=1).fit(next_layer)
        distances, indices = nn.kneighbors(current_layer)

        # Create connections with dynamic distance threshold
        max_distance = 400  # Adjusted based on spatial resolution
        for src_idx, (distance, dest_idx) in enumerate(zip(distances, indices)):
            if distance < max_distance:
                src = current_layer[src_idx]
                dest = next_layer[dest_idx[0]]

                # Add curved connection lines
                fig.add_trace(go.Scatter3d(
                    x=[src[0], (src[0] + dest[0]) / 2, dest[0]],
                    y=[src[1], (src[1] + dest[1]) / 2, dest[1]],
                    z=[src[2], (src[2] + dest[2]) / 2 - z_spacing / 4, dest[2]],
                    mode='lines',
                    line=dict(
                        width=1.5,
                        color="rgba(207, 67, 62, 1)",  # Consistent orange-red
                    ),
                    hoverinfo='skip',
                    showlegend=False
                ))

    # === Visualization Layout ===
    fig.update_layout(
        scene=dict(
            # Axis configuration
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),

            # Camera perspective
            aspectratio=dict(x=1, y=1, z=2),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=-0.1),
                eye=dict(x=1.8, y=-1.5, z=0.4)  # Optimal 3D viewing angle
            ),

            # Background styling
            bgcolor='rgb(255,255,255)'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        paper_bgcolor='white',

    )

    return fig


def calculate_gene_embedding_metrics(adata: sc.AnnData, mapped_adata: sc.AnnData,
                                     n_pca_components: int = 16, random_seed: int = 42) -> pd.DataFrame:
    """
    Calculates similarity metrics (PCC, Cosine Similarity, RMSE, JS Divergence)
    between gene embeddings from two AnnData objects after PCA dimensionality reduction.

    Args:
        adata: The first AnnData object (e.g., reference data).
        mapped_adata: The second AnnData object (e.g., query/mapped data).
        n_pca_components: Number of PCA components to use for gene embeddings. Defaults to 16.
        random_seed: Random seed for PCA reproducibility. Defaults to 42.

    Returns:
        pd.DataFrame: A DataFrame containing 'Gene', 'PCC', 'Cosine Similarity', 'RMSE',
                      and 'JS Divergence' for each common gene.
    """
    print("--- Starting gene embedding metric calculation ---")

    # Transpose AnnData objects (genes as rows, cells as columns)
    # Work on copies to avoid modifying original AnnData objects
    adata_g1 = adata.T.copy()
    adata_g2 = mapped_adata.T.copy()

    print(f"Original adata shape: {adata.shape}")
    print(f"Original mapped_adata shape: {mapped_adata.shape}")
    print(f"Transposed adata_g1 shape (genes x cells): {adata_g1.shape}")
    print(f"Transposed adata_g2 shape (genes x cells): {adata_g2.shape}")

    # Ensure common genes are used for comparison
    common_genes = list(set(adata_g1.obs_names) & set(adata_g2.obs_names))
    if not common_genes:
        raise ValueError("No common genes found between the two AnnData objects after transposition.")
    print(f"Found {len(common_genes)} common genes for analysis.")

    adata_g1 = adata_g1[common_genes, :].copy()
    adata_g2 = adata_g2[common_genes, :].copy()

    print(f"adata_g1 shape after common gene filtering: {adata_g1.shape}")
    print(f"adata_g2 shape after common gene filtering: {adata_g2.shape}")

    # Perform PCA dimensionality reduction
    print(f"\nPerforming PCA ({n_pca_components} dimensions) for both datasets...")
    sc.tl.pca(adata_g1, n_comps=n_pca_components, svd_solver='arpack',
              use_highly_variable=False, random_state=random_seed)
    sc.tl.pca(adata_g2, n_comps=n_pca_components, svd_solver='arpack',
              use_highly_variable=False, random_state=random_seed)

    embedding1 = adata_g1.obsm['X_pca'].copy()
    embedding2 = adata_g2.obsm['X_pca'].copy()

    print(f"Gene embedding dimensions for Dataset 1: {embedding1.shape}")
    print(f"Gene embedding dimensions for Dataset 2: {embedding2.shape}")

    # --- Calculate similarity metrics ---
    print("\nCalculating similarity metrics: PCC, Cosine Similarity, RMSE, JS Divergence...")

    # 1. Cosine Similarity
    # Calculate dot products and norms
    dot_products = np.sum(embedding1 * embedding2, axis=1)
    norms1 = np.linalg.norm(embedding1, axis=1)
    norms2 = np.linalg.norm(embedding2, axis=1)

    # Avoid division by zero: if either norm is zero, cosine similarity is 0
    with np.errstate(divide='ignore', invalid='ignore'):
        cosine_similarities = np.where((norms1 * norms2) != 0, dot_products / (norms1 * norms2), 0.0)

    # 2. Pearson Correlation Coefficient (PCC)
    pcc_values = np.array([
        np.corrcoef(embedding1[i], embedding2[i])[0, 1]
        for i in range(embedding1.shape[0])
    ])
    # Handle NaNs: np.corrcoef returns NaN if either array has zero variance. Replace with 0.0.
    pcc_values = np.nan_to_num(pcc_values, nan=0.0)

    # 3. RMSE (Root Mean Squared Error)
    rmse_values = np.sqrt(np.mean((embedding1 - embedding2) ** 2, axis=1))

    # 4. Jensen-Shannon Divergence (JS Divergence)
    def _softmax(x):
        """Numerically stable softmax implementation."""
        e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return e_x / e_x.sum() if e_x.sum() != 0 else np.ones_like(x) / len(x)  # Handle sum being zero

    js_values = np.zeros(embedding1.shape[0])
    for i in range(embedding1.shape[0]):
        # Convert embedding vectors to probability distributions using softmax
        p = _softmax(embedding1[i])
        q = _softmax(embedding2[i])
        js_values[i] = jensenshannon(p, q)

    # Create results DataFrame
    results_df = pd.DataFrame({
        'Gene': common_genes,  # Use common_genes as the index/column
        'PCC': pcc_values,
        'Cosine Similarity': cosine_similarities,
        'RMSE': rmse_values,
        'JS Divergence': js_values
    })

    print("\n--- Metric calculation complete ---")
    return results_df


def plot_gene_embedding_metrics_distributions(results_df: pd.DataFrame):
    """
    Generates and displays distribution plots for gene embedding similarity metrics:
    Pearson Correlation Coefficient (PCC), Cosine Similarity, Root Mean Squared Error (RMSE),
    and Jensen-Shannon Divergence (JS Divergence).

    Args:
        results_df: A pandas DataFrame containing 'PCC', 'Cosine Similarity', 'RMSE',
                    and 'JS Divergence' columns, typically generated by
                    `calculate_gene_embedding_metrics`.
    """
    print("\n--- Generating distribution plots for similarity metrics ---")

    # Calculate average metrics for plot annotations
    avg_pcc = results_df['PCC'].mean()
    avg_cosine = results_df['Cosine Similarity'].mean()
    avg_rmse = results_df['RMSE'].mean()
    avg_js = results_df['JS Divergence'].mean()

    print(f"Average PCC: {avg_pcc:.4f}")
    print(f"Average Cosine Similarity: {avg_cosine:.4f}")
    print(f"Average RMSE: {avg_rmse:.4f}")
    print(f"Average JS Divergence: {avg_js:.4f}")

    plt.figure(figsize=(16, 12))  # Adjusted figure size for a 2x2 grid with good readability
    sns.set_style("whitegrid")
    sns.set(font_scale=1.2)  # Consistent font scale

    # --- Plot 1: PCC Distribution ---
    plt.subplot(2, 2, 1)
    sns.histplot(
        results_df['PCC'],
        kde=True,  # Kernel Density Estimate for smooth curve
        bins=40,
        color='#4C72B0',  # A pleasant blue color
        edgecolor='none',
        alpha=0.8
    )
    plt.axvline(avg_pcc, color='#C44E52', linestyle='--', linewidth=2)  # Red dashed line for mean
    # Annotate mean value on the plot
    plt.text(avg_pcc + 0.01, plt.ylim()[1] * 0.9,
             f'Mean: {avg_pcc:.4f}', color='#C44E52', fontsize=10)
    plt.title('Distribution of Pearson Correlation', fontsize=14)
    plt.xlabel('PCC')
    plt.ylabel('Gene Count')

    # --- Plot 2: Cosine Similarity Distribution ---
    plt.subplot(2, 2, 2)
    sns.histplot(
        results_df['Cosine Similarity'],
        kde=True,
        bins=40,
        color='#55A868',  # A pleasant green color
        edgecolor='none',
        alpha=0.8
    )
    plt.axvline(avg_cosine, color='#C44E52', linestyle='--', linewidth=2)
    plt.text(avg_cosine + 0.01, plt.ylim()[1] * 0.9,
             f'Mean: {avg_cosine:.4f}', color='#C44E52', fontsize=10)
    plt.title('Distribution of Cosine Similarity', fontsize=14)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Gene Count')

    # --- Plot 3: RMSE Distribution ---
    plt.subplot(2, 2, 3)
    sns.histplot(
        results_df['RMSE'],
        kde=True,
        bins=40,
        color='#DD8452',  # A pleasant orange color
        edgecolor='none',
        alpha=0.8
    )
    plt.axvline(avg_rmse, color='#C44E52', linestyle='--', linewidth=2)
    plt.text(avg_rmse + 0.01, plt.ylim()[1] * 0.9,
             f'Mean: {avg_rmse:.4f}', color='#C44E52', fontsize=10)
    plt.title('Distribution of RMSE', fontsize=14)
    plt.xlabel('RMSE')
    plt.ylabel('Gene Count')

    # --- Plot 4: JS Divergence Distribution ---
    plt.subplot(2, 2, 4)
    sns.histplot(
        results_df['JS Divergence'],
        kde=True,
        bins=40,
        color='#8172B3',  # A pleasant purple color
        edgecolor='none',
        alpha=0.8
    )
    plt.axvline(avg_js, color='#C44E52', linestyle='--', linewidth=2)
    plt.text(avg_js + 0.01, plt.ylim()[1] * 0.9,
             f'Mean: {avg_js:.4f}', color='#C44E52', fontsize=10)
    plt.title('Distribution of JS Divergence', fontsize=14)
    plt.xlabel('JS Divergence')
    plt.ylabel('Gene Count')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle('Gene Embedding Similarity Metric Distributions', fontsize=20, y=0.98)
    plt.show()

    print("\n--- Visualization complete ---")


def get_gene_expression(adata, gene_name, adata_name_for_error=""):
    """Retrieve expression data for a specific gene (ensure dense array)"""
    if gene_name not in adata.var_names:
        raise ValueError(f"Gene '{gene_name}' not found in AnnData object '{adata_name_for_error}'.")
    gene_idx = adata.var_names.get_loc(gene_name)
    expression = adata.X[:, gene_idx]
    if hasattr(expression, "toarray"):
        expression = expression.toarray()
    return expression.flatten()


def generate_spatial_plot_as_array(
        adata,
        gene_name,
        vmin=None,
        vmax=None,
        cmap='viridis',
        spot_size_factor=1.5,
        figure_dpi=100,
        custom_spot_size=1
):
    """Generate an in-memory spatial plot as a numpy array"""
    fig, ax = plt.subplots(dpi=figure_dpi)
    sc.pl.spatial(
        adata,
        color=gene_name,
        img_key=None,
        ax=ax,
        show=False,
        cmap=cmap,
        size=spot_size_factor,
        vmin=vmin,
        vmax=vmax,
        legend_loc=None,
        spot_size=custom_spot_size
    )
    ax.set_axis_off()
    fig.canvas.draw()
    image_array_rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_array_rgb = image_array_rgb.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image_array_rgb


def compare_image_arrays(img_array1_rgb, img_array2_rgb, target_size=(256, 256)):
    """Compare two image arrays using SSIM and MSE metrics"""
    img1_pil_gray = Image.fromarray(img_array1_rgb).convert('L')
    img2_pil_gray = Image.fromarray(img_array2_rgb).convert('L')
    img1_gray_np = np.array(img1_pil_gray)
    img2_gray_np = np.array(img2_pil_gray)
    img1_resized = resize(img1_gray_np, target_size, anti_aliasing=True, preserve_range=True).astype(np.uint8)
    img2_resized = resize(img2_gray_np, target_size, anti_aliasing=True, preserve_range=True).astype(np.uint8)

    range1 = img1_resized.max() - img1_resized.min()
    range2 = img2_resized.max() - img2_resized.min()
    data_range = max(range1, range2)

    if data_range == 0:
        if np.array_equal(img1_resized, img2_resized):
            ssim_score = 1.0
        else:
            ssim_score = 0.0
    else:
        ssim_score = ssim(img1_resized, img2_resized, data_range=data_range if data_range > 0 else None)

    mse_score = mean_squared_error(img1_resized, img2_resized)
    return {"ssim": ssim_score, "mse": mse_score}


def compare_spatial_expression_for_gene(
        adata_obj1,
        adata_obj1_name,
        adata_obj2,
        adata_obj2_name,
        gene_to_compare,
        cmap_to_use='viridis',
        spot_size_factor_to_use=2.5,
        custom_spot_size_to_use=2,
        figure_dpi_to_use=300,
        target_comparison_size=(1000, 1000)
):
    """Compare spatial expression patterns for a single gene between two datasets"""
    try:
        # Get expression values for both datasets
        expression1 = get_gene_expression(adata_obj1, gene_to_compare, adata_obj1_name)
        expression2 = get_gene_expression(adata_obj2, gene_to_compare, adata_obj2_name)

        # Calculate global expression range
        global_vmin = min(np.min(expression1), np.min(expression2))
        global_vmax = max(np.max(expression1), np.max(expression2))

        # Handle case where expression range is zero
        if global_vmax == global_vmin:
            global_vmax += 1e-9
            if global_vmin == 0 and global_vmax == 1e-9:
                global_vmin -= 1e-9

    except ValueError as e:
        return None  # Skip genes not found in either dataset

    # Generate spatial plots as numpy arrays
    img_array1 = generate_spatial_plot_as_array(
        adata_obj1, gene_to_compare,
        cmap=cmap_to_use, spot_size_factor=spot_size_factor_to_use,
        figure_dpi=figure_dpi_to_use, custom_spot_size=custom_spot_size_to_use
    )

    img_array2 = generate_spatial_plot_as_array(
        adata_obj2, gene_to_compare,
        cmap=cmap_to_use, spot_size_factor=spot_size_factor_to_use,
        figure_dpi=figure_dpi_to_use, custom_spot_size=custom_spot_size_to_use
    )

    # Calculate similarity metrics
    similarity_scores = compare_image_arrays(img_array1, img_array2, target_size=target_comparison_size)

    return similarity_scores


def compare_spatial_expression_all_genes(
        adata1,
        adata1_name,
        adata2,
        adata2_name,
        max_genes=None,
        output_csv="ssim_scores.csv",
        plot_file="ssim_distribution.png",
        figsize=(12, 8)
):
    """
    Calculate SSIM scores for all common genes between two datasets
    and create a comprehensive visualization of the distribution.

    Returns:
    mean_ssim: Average SSIM score
    ssim_df: DataFrame with SSIM scores for all genes
    """
    # Find common genes between datasets
    common_genes = list(set(adata1.var_names) & set(adata2.var_names))

    if not common_genes:
        print("Error: No common genes found!")
        return None, None

    # Optionally limit number of genes for testing
    if max_genes:
        common_genes = common_genes[:max_genes]
        print(f"Testing mode: Only processing first {max_genes} genes")

    print(f"Starting comparison for {len(common_genes)} common genes...")

    # Initialize results dictionary
    results = {'gene': [], 'ssim': [], 'mse': []}
    skipped_genes = []

    # Setup progress bar
    pbar = tqdm(total=len(common_genes), desc="Calculating gene SSIM scores")

    # Process each gene
    for gene in common_genes:
        scores = compare_spatial_expression_for_gene(
            adata1, adata1_name,
            adata2, adata2_name,
            gene
        )

        if not scores:
            skipped_genes.append(gene)
        else:
            results['gene'].append(gene)
            results['ssim'].append(scores['ssim'])
            results['mse'].append(scores['mse'])

        pbar.update(1)

    pbar.close()

    # Report skipped genes
    if skipped_genes:
        print(f"Skipped {len(skipped_genes)} genes (not expressed or error)")

    # Create DataFrame with results
    ssim_df = pd.DataFrame(results)

    # Handle no results case
    if ssim_df.empty:
        print("Failed to calculate SSIM scores for any genes")
        return None, None

    # Calculate statistics
    mean_ssim = ssim_df['ssim'].mean()
    median_ssim = ssim_df['ssim'].median()
    std_ssim = ssim_df['ssim'].std()

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"Analysis complete! Processed {len(ssim_df)} genes")
    print(f"Mean SSIM: {mean_ssim:.4f} ± {std_ssim:.4f}")
    print(f"Median SSIM: {median_ssim:.4f}")
    print(f"{'=' * 50}\n")

    # Save results
    if output_csv:
        ssim_df.to_csv(output_csv, index=False)
        print(f"SSIM scores saved to: {output_csv}")

    # Create comprehensive visualization
    plt.figure(figsize=figsize)

    # Apply styling
    sns.set_style("whitegrid")
    sns.set_palette("pastel")

    # Create distribution plot
    ax = sns.histplot(
        ssim_df['ssim'],
        bins=30,
        kde=True,
        stat="density",
        alpha=0.7,
        color="#66a0e9"
    )

    # Add mean line
    plt.axvline(mean_ssim, color='#d62728', linestyle='-', linewidth=2.5,
                label=f'Mean = {mean_ssim:.4f}')

    # Add median line
    plt.axvline(median_ssim, color='#2ca02c', linestyle='--', linewidth=2.5,
                label=f'Median = {median_ssim:.4f}')

    # Highlight standard deviation region
    plt.axvspan(mean_ssim - std_ssim, mean_ssim + std_ssim,
                color='#d62728', alpha=0.1, label=f'Std. Dev. = {std_ssim:.4f}')

    # Set titles and labels
    plt.title(f'Spatial Expression Similarity Distribution ({adata1_name} vs {adata2_name})',
              fontsize=16, pad=20)
    plt.xlabel('Structural Similarity Index (SSIM)', fontsize=13)
    plt.ylabel('Density', fontsize=13)

    # Add legend
    plt.legend(fontsize=12, frameon=True, loc='best')

    # Adjust layout
    plt.tight_layout()

    # Save visualization
    if plot_file:
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {plot_file}")

    plt.show()

    return mean_ssim, ssim_df
