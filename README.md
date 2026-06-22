# GenOT: Generative optimal transport enables spatiotemporal interpolation and generation in cross-platform spatial transcriptomics

## Introduction
Spatial transcriptomics (ST) technologies have revolutionized the analysis of spatial gene expression patterns within tissues. However, existing computational methods still face challenges in integrating spatial information and generating cross-heterogeneous sample data. To address this, we developed GenOT - a spatial data generation framework based on graph self-supervised contrastive learning and optimal transport theory. The framework employs a multimodal feature learning architecture to dynamically identify important genes  and hierarchically aggregate spatial neighborhood information, achieving high-precision spatial domain clustering and biologically interpretable feature extraction. The core innovation of GenOT lies in introducing an optimal transport barycenter-based interpolation algorithm, which mathematically models cross-sample spatial distribution differences to reconstruct spatiotemporal continuous gene expression dynamics. Experiments on multiple datasets including human dorsolateral prefrontal cortex (10x visium), mouse embryonic development (Stereo-seq), and olfactory bulb/hippocampal tissues (Slide-seqV2) demonstrate that GenOT significantly outperforms existing methods in spatial domain identification, cross-technology platform integration, and developmental trajectory reconstruction. This provides an innovative tool for tissue structure analysis at single-cell resolution and developmental process modeling.
![image](https://github.com/wrab12/GenOT/blob/main/GenOT.jpg)

## Repository Structure

```
GenOT/
├── GenOT/                  # Core Python package
│   ├── __init__.py         # Package entry point; exports Encoder / DualEncoder / Decoder, etc.
│   ├── genot.py            # Main classes: Encoder (single-slice representation), DualEncoder (dual-slice integration), Decoder (embedding → expression reconstruction)
│   ├── model.py            # Network modules: K-hop aggregation (CombUnweighted), MGCN/MGCN2, Discriminator, AvgReadout
│   ├── preprocess.py       # Preprocessing: normalization, KNN graph construction, contrastive labels, PCA features, adjacency normalization
│   ├── OTutils.py          # Optimal transport: FGW barycenter interpolation, EMD transport plans, mapping expression back to target coordinates
│   ├── utils.py            # Clustering (mclust/leiden, etc.), spatial alignment (PASTE2/ICP), marker-gene color alignment
│   └── plotting.py         # Visualization functions
│
├── Tutorial/               # 8 Jupyter tutorials reproducing the paper's experiments
│   ├── Tutorial 1_DLPFC.ipynb                          # Spatial domain identification (10x Visium)
│   ├── Tutorial 2_MOSTA.ipynb                          # Spatial domain identification (Stereo-seq)
│   ├── Tutorial 3_Mouse_Hippocampus.ipynb             # Mouse hippocampus
│   ├── Tutorial 4_Mouse_Olfactory(...).ipynb          # Olfactory bulb (Stereo-seq & Slide-seqV2)
│   ├── Tutorial 5_Mouse_Brain_Merge_...ipynb          # Anterior–posterior brain slice merging
│   ├── Tutorial 6_DLPFC_interpolation.ipynb           # Spatiotemporal interpolation
│   ├── Tutorial 7_MOSTA_integration.ipynb             # Cross-slice integration
│   └── Tutorial 8_Diff_Tech_MOSTA_integration.ipynb   # Cross-platform integration
│
├── Data/                   # Data instructions (no actual data; download links only)
│   └── README.md
│
├── docs/                   # Sphinx documentation source for the ReadTheDocs site
│   ├── source/
│   ├── Figures/
│   └── ...
│
├── somde/                  # Bundled SOMDE dependency (spatially variable gene detection)
│
├── GenOT.jpg               # Model overview figure (referenced in README)
├── requirements.txt        # Python dependencies
├── smoke_test.py           # End-to-end integration smoke test on synthetic data
├── readthedocs.yaml        # ReadTheDocs build configuration
├── LICENSE                 # MIT license
└── README.md
```

## Environment
First, clone and navigate to the repository.
```bash
git clone https://github.com/wrab12/GenOT
cd GenOT
```
This process can take several minutes, depending on network speed.

Create and activate a virtual environment using python 3.9 with `conda`,
```bash
# conda
conda create -n GenOT python=3.9
conda activate GenOT
```

Install dependencies and the local library with `pip`.
```bash
pip install -r requirements.txt
```
This process usually takes around 5 minutes.

Next, you need to install `torch` and `torch_geometric`. The installation depends on your computer's CUDA version. Please follow these steps:

1. Run `nvidia-smi` to check your CUDA version.
2. Install the appropriate version of `torch` according to your CUDA version. Refer to the official guide: [PyTorch Installation](https://pytorch.org/get-started/locally/).
3. After installing `torch`, install `torch_geometric`.
4. For detailed instructions, visit [PyTorch Geometric Installation Guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

Specifically, you can use the following command:
```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch_geometric
```
Replace `${TORCH}` and `${CUDA}` with your specific PyTorch and CUDA versions.

For more details, see the official documentation for [PyTorch](https://pytorch.org/get-started/locally/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

You also need to install R from the official [CRAN website](https://cran.r-project.org/), then install the `mclust` package inside R:
```r
install.packages("mclust")
```
The SOMDE-based gene selection also requires `somoclu`, which should be installed from conda-forge (the pip wheel may lack the compiled library):
```bash
conda install -c conda-forge somoclu
```

## Testing

After setting up the environment, verify your installation with the included smoke test. It runs both main GenOT pipelines (spatial domain identification and spatiotemporal interpolation) end-to-end on a small synthetic dataset in about a minute, and exits non-zero if anything is broken.

```bash
cd GenOT          # repo root (the folder containing the GenOT/ package)
python smoke_test.py
```

A successful run prints `[smoke] ALL PASS` and exits with code 0. 

## Datasets

- Download all datasets from this [Google Drive link](https://drive.google.com/drive/folders/1Id4p7bpOruKgPL-sy2iT72D13P2H_w9P?usp=drive_link).


## Documentation

See detailed documentation and examples at [https://GenOT.readthedocs.io/en/latest/index.html](https://GenOT.readthedocs.io/en/latest/index.html).

## Acknowledgements
We thank the developers of GraphST ([GitHub](https://github.com/JinmiaoChenLab/GraphST)), Somde ([GitHub](https://github.com/WhirlFirst/somde)), Paste2 ([GitHub](https://github.com/raphael-group/paste2)), and POT ([GitHub](https://github.com/PythonOT/POT)) for their valuable tools and resources.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

Wang, R., Liu, X., Zhuo, L. et al. GenOT: generative optimal transport enables spatiotemporal interpolation and generation in cross-platform spatial transcriptomics. *Genome Biology* (2026). https://doi.org/10.1186/s13059-026-04166-z



