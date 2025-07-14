# GenOT: Optimal Transport-Driven Generation of Cross-Platform Spatial Transcriptomic Data and Spatiotemporal Continuous Modeling

## Introduction
Spatial transcriptomics (ST) technologies have revolutionized the analysis of spatial gene expression patterns within tissues. However, existing computational methods still face challenges in integrating spatial information and generating cross-heterogeneous sample data. To address this, we developed GenOT - a spatial data generation framework based on graph self-supervised contrastive learning and optimal transport theory. The framework employs a multimodal feature learning architecture to dynamically identify important genes  and hierarchically aggregate spatial neighborhood information, achieving high-precision spatial domain clustering and biologically interpretable feature extraction. The core innovation of GenOT lies in introducing an optimal transport barycenter-based interpolation algorithm, which mathematically models cross-sample spatial distribution differences to reconstruct spatiotemporal continuous gene expression dynamics. Experiments on multiple datasets including human dorsolateral prefrontal cortex (10x visium), mouse embryonic development (Stereo-seq), and olfactory bulb/hippocampal tissues (Slide-seqV2) demonstrate that GenOT significantly outperforms existing methods in spatial domain identification, cross-technology platform integration, and developmental trajectory reconstruction. This provides an innovative tool for tissue structure analysis at single-cell resolution and developmental process modeling.
![image](https://github.com/wrab12/GenOT/blob/main/GenOT.png)
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

You also need to install R. You can download the installer and run it from the official CRAN website: https://cran.r-project.org/

## Datasets

- Download all datasets from this [Google Drive link](https://drive.google.com/drive/folders/1Id4p7bpOruKgPL-sy2iT72D13P2H_w9P?usp=drive_link).


## Documentation

See detailed documentation and examples at [https:/GenOT.readthedocs.io/en/latest/index.html](https://GenOT.readthedocs.io/en/latest/index.html).

## Acknowledgements
We thank the developers of GraphST ([GitHub](https://github.com/JinmiaoChenLab/GraphST)), Somde ([GitHub](https://github.com/WhirlFirst/somde)), Paste2 ([GitHub](https://github.com/raphael-group/paste2)), and POT ([GitHub](https://github.com/PythonOT/POT)) for their valuable tools and resources.

## References



