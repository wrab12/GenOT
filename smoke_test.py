"""
GenOT smoke test (identification + interpolation, synthetic data)
=================================================================

This is a regression / integration smoke test, NOT a performance benchmark.
The goal is to run both main GenOT task chains end-to-end on a small
synthetic spatial-transcriptomics dataset in about a minute. Any crash or
behavioural regression in any part of the pipeline causes a non-zero exit
code, so the script is suitable as:

  - A post-edit sanity check after changing source or upgrading dependencies
  - A minimal CI job (GitHub Actions, local pre-push hook)
  - A "did I install the environment correctly?" check for new users

Tasks covered
-------------
A. Spatial domain identification (matches Tutorial 1: DLPFC)
     synthetic visium-like slice
     -> GenOT.utils.find_hvg_somde     (exercises somde + somoclu binary library)
     -> GenOT.genot.Encoder            (preprocess / KNN graph / contrastive
                                        labels / GCN / PCA)
     -> GenOT.utils.clustering         (rpy2 + R + mclust)
     -> sklearn metrics: ARI / NMI / HOM / COM

B. Spatiotemporal interpolation (matches Tutorial 6: DLPFC interpolation)
     3 synthetic visium-like slices sharing the same gene panel
     (adata1, adata2 are references; adata3 is the target whose role is
     only to provide num_barycenters and a ground-truth comparison)
     -> GenOT.utils.normalize_sparse
     -> GenOT.genot.DualEncoder.train_encoder
     -> GenOT.OTutils.compute_spatial_barycenter    (FGW on spatial coords)
     -> GenOT.genot.Decoder + train_decoder         (emb -> gene expression)
     -> GenOT.OTutils.compute_emb_barycenter        (FGW on embeddings)
     -> GenOT.OTutils.update_embedding_barycenter
     -> Decoder forward at the interpolated positions
     -> custom metric: marker discrimination ratio
        = mean predicted marker expression at points in the correct quadrant
        / mean predicted marker expression at points in the wrong quadrants
        High values mean the interpolation chain put the right markers in
        the right places.

Synthetic data shape
--------------------
- One slice = an n_per_side x n_per_side grid; each cell is one spot,
  giving N = n_per_side ** 2 spots total
- The grid is split by its two midlines into 4 quadrants (Q0..Q3) used as
  ground-truth domains
- Of n_genes genes, the first 4 * n_markers_per_domain are markers:
  for domain d, gene indices [d*n_markers_per_domain, (d+1)*n_markers_per_domain)
  have Poisson rate = marker_rate inside Qd and = baseline_rate everywhere
  else. All non-marker genes are Poisson(baseline_rate) everywhere. The
  default signal ratio (15:1) is intentionally clean.
- The signal is "too clean to fail" on purpose: a smoke test should only
  verify the pipeline runs and goes in the right direction, not whether
  the method handles realistic noise (that is what the paper benchmarks
  are for).
- Task B uses three slices generated with different seeds. They share the
  marker layout and coordinate grid, so we deliberately skip PASTE2
  alignment (PASTE2 is an external dependency with its own tests).

Pass thresholds
---------------
- Task A: ARI >= 0.30 (typically lands at 1.0 with this synthetic signal)
- Task B: discrimination_ratio >= 1.5 (typically 2.5 ~ 3.5)
Both thresholds are deliberately well below what the synthetic data should
yield. They tolerate random init jitter but FAIL hard if any link in the
chain "learns nothing".

Environment prerequisites
-------------------------
- A working GenOT runtime: somde + somoclu (from conda-forge, NOT pip; the
  pip wheel may ship without the compiled .so on Linux) + R + r-mclust +
  rpy2 + the usual scientific Python stack.
- Run from the repo root (the directory containing the GenOT/ package).
  The script prepends the repo root to sys.path so `from GenOT import ...`
  works even without `pip install -e .`.
- If you are in a conda environment with r-base installed, the script
  auto-sets R_HOME to $CONDA_PREFIX/lib/R, so you do not need the hard-
  coded Windows R path from the tutorial notebooks.

Usage
-----
    cd <repo root>
    python smoke_test.py

Exit code: 0 if both tasks pass, 1 otherwise.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

# --- sys.path so `from GenOT import ...` works without setup.py / pyproject ---
# Tutorial notebooks assume cwd == repo root, but Jupyter or any other cwd
# would otherwise ModuleNotFoundError. Force the repo root onto sys.path[0].
_HERE = os.path.abspath(os.path.dirname(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# --- R_HOME so rpy2 can find the R inside the active conda env ---------------
# This replaces the hard-coded 'C:/Program Files/R/R-4.4.1' from the tutorials,
# which is invalid on Linux / macOS.
if 'R_HOME' not in os.environ:
    _conda = os.environ.get('CONDA_PREFIX', '')
    if _conda and os.path.isdir(os.path.join(_conda, 'lib', 'R')):
        os.environ['R_HOME'] = os.path.join(_conda, 'lib', 'R')
        os.environ['PATH'] = os.path.join(_conda, 'bin') + os.pathsep + os.environ['PATH']

import numpy as np
import scipy.sparse as sp
import anndata as ad
import torch
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
)


# =============================================================================
# Synthetic data
# =============================================================================

def make_synthetic_spatial(
    n_per_side: int = 30,
    n_genes: int = 1500,
    n_markers_per_domain: int = 50,
    marker_rate: float = 15.0,
    baseline_rate: float = 1.0,
    seed: int = 0,
) -> ad.AnnData:
    """Generate one synthetic visium-like slice on an n_per_side x n_per_side grid.

    Layout:
      - N = n_per_side ** 2 spots, integer (x, y) coordinates on the grid
      - The grid is split by its two midlines into 4 quadrants Q0..Q3, used
        as the ground-truth domain labels
      - Gene indices [d*n_markers_per_domain, (d+1)*n_markers_per_domain)
        are markers for domain d: Poisson rate = marker_rate inside Qd,
        baseline_rate everywhere else
      - All non-marker genes are Poisson(baseline_rate) everywhere
      - With the defaults, the marker-to-baseline rate ratio is 15

    Returns an AnnData with:
      .X                       sparse counts, shape (N, n_genes)
      .obsm['spatial']         (N, 2) float32 coordinates (required by GenOT)
      .obs['Ground Truth']     categorical, values in {'Q0','Q1','Q2','Q3'}
    """
    rng = np.random.default_rng(seed)

    # Integer grid coordinates
    xs, ys = np.meshgrid(np.arange(n_per_side), np.arange(n_per_side))
    coords = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    n_spots = coords.shape[0]

    # Quadrant label = 2-bit code from the two midline tests
    half = n_per_side / 2.0
    domain = (coords[:, 0] >= half).astype(int) * 2 + (coords[:, 1] >= half).astype(int)
    domain_names = np.array([f'Q{d}' for d in domain])

    # Rate matrix: baseline everywhere, then lift each quadrant's marker
    # block to marker_rate for spots inside that quadrant
    rates = np.full((n_spots, n_genes), baseline_rate, dtype=np.float32)
    assert 4 * n_markers_per_domain <= n_genes, "n_markers_per_domain too large for n_genes"
    for d in range(4):
        gi0 = d * n_markers_per_domain
        gi1 = gi0 + n_markers_per_domain
        rates[domain == d, gi0:gi1] = marker_rate

    counts = rng.poisson(rates).astype(np.float32)

    var_names = np.array([f'gene_{i:04d}' for i in range(n_genes)])
    obs_names = np.array([f's{seed}_{i:04d}' for i in range(n_spots)])

    adata = ad.AnnData(
        X=sp.csr_matrix(counts),
        obs={'Ground Truth': domain_names},
        var={'gene_symbol': var_names},
    )
    adata.obs_names = obs_names
    adata.var_names = var_names
    adata.obsm['spatial'] = coords
    adata.obs['Ground Truth'] = adata.obs['Ground Truth'].astype('category')
    return adata


# =============================================================================
# Task A: spatial domain identification
# =============================================================================

def run_identification(device: torch.device) -> tuple[bool, dict]:
    """Run the SOMDE -> Encoder -> mclust pipeline and score it against
    the synthetic ground truth.

    Environment pieces this exercises (anything broken here -> FAIL):
      - somde + somoclu (the latter MUST be the conda-forge build that can
        actually train; the pip wheel can be a "load only" shell that fails
        at retrain time)
      - GenOT main package must be importable (handled via sys.path above)
      - PyTorch (optionally CUDA; falls back to CPU which is slower)
      - rpy2 must reach R, and R must have library(mclust) available
    """
    print('\n=== Task A: spatial domain identification ===')
    t_a = time.time()

    # One synthetic 900-spot x 1500-gene slice
    adata = make_synthetic_spatial(seed=0)
    print(f'[A.synthetic] N={adata.n_obs} G={adata.n_vars} '
          f'domains={list(adata.obs["Ground Truth"].cat.categories)}')

    from GenOT.utils import find_hvg_somde, clustering
    from GenOT import genot

    # --- HVG selection via SOMDE --------------------------------------------
    # n_node=5 makes SomNode use somn = int(sqrt(N / n_node)) = int(sqrt(180)) = 13,
    # i.e. a 13x13 SOM. On N=900 that is enough statistical power for the q<0.05
    # filter to return non-empty results. A too-large n_node (= too few SOM nodes)
    # yields zero significant genes and the smoke test fails immediately.
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # quiet POT / scanpy / matplotlib deprecations
        hvg = find_hvg_somde(adata, save_output=False, n_node=5, n_retrain=30)
    print(f'[A.find_hvg_somde] {time.time()-t0:.2f}s -> {len(hvg)} HVGs')
    if len(hvg) == 0:
        return False, {'reason': 'SOMDE returned 0 HVGs'}

    # Keep top-K HVGs so the Encoder stays small
    k = min(300, len(hvg))
    adata = adata[:, hvg[:k]].copy()

    # --- GenOT Encoder ------------------------------------------------------
    # Important implicit constraint: GenOT.utils.clustering() runs an internal
    # PCA(n_components=16) on obsm['emb'], so the Encoder's pca_n must be >= 16
    # or sklearn raises ValueError. The constraint is undocumented; this smoke
    # test serves to surface it.
    t0 = time.time()
    enc = genot.Encoder(adata, device=device, pca_n=16, epochs=200)
    adata = enc.train_encoder()
    print(f'[A.train_encoder] {time.time()-t0:.2f}s emb shape={adata.obsm["emb"].shape}')
    if 'emb' not in adata.obsm:
        return False, {'reason': 'Encoder did not produce obsm["emb"]'}

    # --- Clustering via mclust (rpy2 -> R) ----------------------------------
    t0 = time.time()
    n_clusters = len(adata.obs['Ground Truth'].cat.categories)  # = 4
    clustering(adata, n_clusters=n_clusters, method='mclust', refinement=True)
    print(f'[A.mclust] {time.time()-t0:.2f}s pred_clusters={adata.obs["domain"].nunique()}')

    # --- Metrics ------------------------------------------------------------
    true = adata.obs['Ground Truth']
    pred = adata.obs['domain']
    metrics = {
        'ARI': adjusted_rand_score(true, pred),
        'NMI': normalized_mutual_info_score(true, pred),
        'HOM': homogeneity_score(true, pred),
        'COM': completeness_score(true, pred),
        'wall_s': time.time() - t_a,
    }
    print('[A.metrics]')
    for k_, v in metrics.items():
        print(f'  {k_}: {v:.4f}' if isinstance(v, float) else f'  {k_}: {v}')

    # Synthetic signal is clean enough that any working pipeline should hit
    # ARI ~ 1.0. The 0.30 floor is intentionally loose so random GCN init
    # jitter is tolerated; anything below it means the chain genuinely broke.
    ARI_FLOOR = 0.30
    ok = metrics['ARI'] >= ARI_FLOOR
    if ok:
        print('[A] PASS')
    else:
        ari_v = metrics['ARI']
        print(f'[A] FAIL: ARI {ari_v:.3f} < floor {ARI_FLOOR}')
    return ok, metrics


# =============================================================================
# Task B: spatiotemporal interpolation
# =============================================================================

def run_interpolation(
    device: torch.device,
    n_per_side: int = 20,
    n_genes: int = 800,
    n_markers_per_domain: int = 50,
    decoder_epochs: int = 100,
) -> tuple[bool, dict]:
    """Run the DualEncoder -> spatial barycenter -> Decoder -> embedding
    barycenter -> decoded gene expression pipeline.

    Setup:
      - adata1, adata2: two reference slices used to learn the embedding
        space
      - adata3: the target slice. Its only role is to provide
        num_barycenters (the number of interpolated points) and to act as
        ground truth for scoring. It does NOT participate in training.
      - PASTE2 alignment is intentionally skipped: all three slices share
        the same coordinate grid, so they are already aligned. PASTE2 is a
        separate external dependency and is not exercised by this smoke test.

    Pass metric (discrimination_ratio):
      For each interpolated point i:
        - From its barycenter spatial coordinate Xb_s[i], decide which
          quadrant d it falls in (this becomes the ground-truth domain)
        - in_signal[i]  = mean predicted expression in the marker block
                          for domain d
        - out_signal[i] = mean predicted expression in the marker blocks
                          of the other three domains
      discrimination_ratio = mean(in_signal) / mean(out_signal)

      - A chain that learned nothing -> ratio ~ 1.0
      - A perfectly faithful chain -> approaches marker_rate / baseline = 15
      - Typical observed value: 2.5 ~ 3.5, which confirms the OT barycenter
        chain ties embedding structure to spatial structure
    """
    print('\n=== Task B: spatiotemporal interpolation ===')
    t_b = time.time()

    # Three slices with the same synthetic schema but different seeds:
    # marker layout is identical, count noise differs.
    adata1 = make_synthetic_spatial(n_per_side=n_per_side, n_genes=n_genes,
                                    n_markers_per_domain=n_markers_per_domain, seed=1)
    adata2 = make_synthetic_spatial(n_per_side=n_per_side, n_genes=n_genes,
                                    n_markers_per_domain=n_markers_per_domain, seed=2)
    adata3 = make_synthetic_spatial(n_per_side=n_per_side, n_genes=n_genes,
                                    n_markers_per_domain=n_markers_per_domain, seed=3)
    print(f'[B.synthetic] adata1=N{adata1.n_obs} adata2=N{adata2.n_obs} '
          f'adata3=N{adata3.n_obs}  G={adata1.n_vars}')

    # Remember the marker gene blocks for each domain. We need them in the
    # final scoring step to compute in- vs out-domain marker expression.
    marker_blocks = {d: slice(d * n_markers_per_domain, (d + 1) * n_markers_per_domain)
                     for d in range(4)}

    from GenOT.utils import normalize_sparse
    from GenOT import genot
    from GenOT.OTutils import (
        compute_spatial_barycenter,
        compute_emb_barycenter,
        update_embedding_barycenter,
    )

    # --- Normalize ----------------------------------------------------------
    t0 = time.time()
    adata1 = normalize_sparse(adata1)
    adata2 = normalize_sparse(adata2)
    adata3 = normalize_sparse(adata3)
    print(f'[B.normalize] {time.time()-t0:.2f}s')

    # --- DualEncoder on the two reference slices ----------------------------
    # pca_n=16 matches Tutorial 6 defaults
    t0 = time.time()
    dual = genot.DualEncoder(adata1, adata2, device=device, pca_n=16)
    adata1, adata2 = dual.train_encoder()
    emb_dim = adata1.obsm['emb'].shape[1]
    print(f'[B.DualEncoder] {time.time()-t0:.2f}s '
          f'emb1={adata1.obsm["emb"].shape} emb2={adata2.obsm["emb"].shape}')
    if 'emb' not in adata1.obsm or 'emb' not in adata2.obsm:
        return False, {'reason': 'DualEncoder did not produce emb on both slices'}

    n_bary = adata3.n_obs  # number of interpolated points = target slice size

    # --- Fused GW barycenter on spatial coordinates -------------------------
    # max_iter=5 cuts the FGW iterations down. The smoke test does not need
    # full convergence; it only needs the call to run and return useful output.
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Xb_s, s_plans = compute_spatial_barycenter(
            adata1, adata2, num_barycenters=n_bary, max_iter=5)
    print(f'[B.spatial_barycenter] {time.time()-t0:.2f}s Xb_s={Xb_s.shape} '
          f'#plans={len(s_plans)}')

    # --- Decoder: embedding -> gene expression ------------------------------
    t0 = time.time()
    dec = genot.Decoder(input_size=emb_dim, output_size=adata1.X.shape[1])
    trained_dec = dec.train_decoder(
        adata1, adata2, dec,
        epochs=decoder_epochs, batch_size=128, learning_rate=1e-2,
        device=str(device),
    )
    print(f'[B.Decoder.train] {time.time()-t0:.2f}s epochs={decoder_epochs}')

    # --- Fused GW barycenter on embeddings ----------------------------------
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Xb_e, e_plans = compute_emb_barycenter(
            adata1, adata2, num_barycenters=n_bary, max_iter=5)
    print(f'[B.emb_barycenter] {time.time()-t0:.2f}s Xb_e={Xb_e.shape} '
          f'#plans={len(e_plans)}')

    # --- Cross-calibrate emb barycenter using both transport plans ---------
    t0 = time.time()
    Xb_e_updated = update_embedding_barycenter(Xb_s, Xb_e, s_plans, e_plans)
    print(f'[B.update_emb_bary] {time.time()-t0:.2f}s shape={Xb_e_updated.shape}')

    # --- Decode emb back to predicted gene expression at the interp points -
    t0 = time.time()
    trained_dec.eval()
    with torch.no_grad():
        emb_tensor = torch.tensor(Xb_e_updated, dtype=torch.float32, device=device)
        reconstructed = trained_dec(emb_tensor).cpu().numpy()
    print(f'[B.decode] {time.time()-t0:.2f}s reconstructed={reconstructed.shape}')

    # Shape / numerical sanity
    if reconstructed.shape != (n_bary, adata1.X.shape[1]):
        return False, {'reason': f'shape mismatch: {reconstructed.shape} '
                                  f'!= ({n_bary}, {adata1.X.shape[1]})'}
    if not np.isfinite(reconstructed).all():
        return False, {'reason': 'reconstructed expression contains non-finite values'}

    # Tutorial 6 cell 32 does the same thresholding: zero out weak signal,
    # keep only strong markers per gene.
    thresholds = reconstructed.max(axis=0) / 2
    reconstructed_thr = reconstructed.copy()
    for gi, t in enumerate(thresholds):
        col = reconstructed_thr[:, gi]
        col[col < t] = 0

    # --- Metric: marker discrimination ratio --------------------------------
    # Use the barycenter's spatial coordinate to assign each interp point a
    # ground-truth quadrant, then compare predicted expression in that
    # quadrant's markers vs the other three quadrants' markers.
    half = n_per_side / 2.0
    bary_domain = ((Xb_s[:, 0] >= half).astype(int) * 2
                   + (Xb_s[:, 1] >= half).astype(int))

    in_signal = np.zeros(n_bary)
    out_signal = np.zeros(n_bary)
    for i in range(n_bary):
        d = int(bary_domain[i])
        in_signal[i] = reconstructed_thr[i, marker_blocks[d]].mean()
        # Exclude the noise-tail (non-marker) genes from the comparison so
        # the ratio is in-domain markers vs out-of-domain markers only.
        other_marker_mask = np.zeros(reconstructed_thr.shape[1], dtype=bool)
        for d2 in range(4):
            if d2 != d:
                other_marker_mask[marker_blocks[d2]] = True
        out_signal[i] = reconstructed_thr[i, other_marker_mask].mean()
    eps = 1e-9
    discrim = (in_signal.mean() + eps) / (out_signal.mean() + eps)

    metrics = {
        'recon_shape': reconstructed.shape,
        'recon_mean': float(reconstructed.mean()),
        'recon_std': float(reconstructed.std()),
        'in_domain_marker_mean': float(in_signal.mean()),
        'out_domain_marker_mean': float(out_signal.mean()),
        'discrimination_ratio': float(discrim),
        'wall_s': time.time() - t_b,
    }
    print('[B.metrics]')
    for k_, v in metrics.items():
        print(f'  {k_}: {v:.4f}' if isinstance(v, float) else f'  {k_}: {v}')

    # Typical run gets 2.5 ~ 3.5. The 1.5 floor leaves room for FGW init
    # randomness but anything <= 1.0 means the OT chain failed to tie
    # spatial structure to gene structure, which the smoke test must catch.
    DISCRIM_FLOOR = 1.5
    ok = metrics['discrimination_ratio'] >= DISCRIM_FLOOR
    if ok:
        print('[B] PASS')
    else:
        print(f'[B] FAIL: discrim {discrim:.3f} < floor {DISCRIM_FLOOR}')
    return ok, metrics


# =============================================================================
# main
# =============================================================================

def main() -> int:
    print('=== GenOT smoke test (identification + interpolation) ===')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  torch={torch.__version__}  cuda={torch.cuda.is_available()}')

    a_ok, _ = run_identification(device)
    b_ok, _ = run_interpolation(device)

    print()
    print('=== summary ===')
    print(f'Task A (identification): {"PASS" if a_ok else "FAIL"}')
    print(f'Task B (interpolation):  {"PASS" if b_ok else "FAIL"}')
    if a_ok and b_ok:
        print('[smoke] ALL PASS')
        return 0
    print('[smoke] FAIL')
    return 1


if __name__ == '__main__':
    sys.exit(main())
