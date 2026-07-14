# 3D Gaussian Inverse Rendering with Diffusion Priors

This repository contains the official implementation of **3D Gaussian Inverse Rendering with Diffusion Priors**.

[![Project Page](https://img.shields.io/badge/Project-Website-blue?style=flat-square)](https://PuuTzzA.github.io/PBR-3DGS/)
[![Paper](https://img.shields.io/badge/Paper-PDF-red?style=flat-square)](https://PuuTzzA.github.io/PBR-3DGS/static/paper.pdf)

---

## Method Overview

<p align="center">
  <img src="docs/static/images/method_readme.png" alt="Method Overview" width="600">
</p>

Our framework performs Physically Based Rendering (PBR) reconstruction and decomposition (Albedo, Normals, and Roughness) from multi-view images under unknown lighting conditions. By leveraging 2D Diffusion Priors, we constrain the inverse rendering problem, producing high-fidelity material parameters and enabling photorealistic novel-view synthesis and relighting under novel environment maps.

For a detailed mathematical description and technical breakdown of our pipeline, material model, and prior extraction techniques, please refer to [METHOD.md](METHOD.md).

---

## Setup (Short)

For the detailed environment installation and troubleshooting guide (compiling custom CUDA submodules, NumPy 2 compatibility, and PyTorch builds), see the [Detailed Setup Guide](SETUP.md).

### Quickstart

1. **Clone the repository recursively**:
   ```bash
   git clone --recursive --shallow-submodules https://github.com/PuuTzzA/PBR-3DGS.git
   cd PBR-3DGS
   ```

2. **Create and activate the environment**:
   ```bash
   conda env create -f environment.yml
   conda activate gir
   ```

3. **Install the custom CUDA submodules**:
   Please follow the compilation steps documented in [SETUP.md](SETUP.md) to successfully build `diff-gaussian-rasterization`, `simple-knn`, and `envlight` for your system configuration.

---

## Running the Code

### 1. Training & Experiments
To train the model on a single scene dataset:
```bash
cd GIR
python train.py \
  -s /path/to/dataset \
  -m ./output_model \
  --eval \
  --port 6009
```

To run systematic experiment sets across various priors, baselines, and parameterizations:
```bash
# Run from the GIR directory
python run_experiments.py
```

### 2. Evaluation & Relighting
To evaluate the rendered novel views and calculate core image quality metrics (PSNR, SSIM, LPIPS):
```bash
# Run from the root directory
python evaluate_model.py --model_path ./GIR/output_model
```

To quantitatively evaluate the decomposed material parameters (Albedo, Normals, etc.):
```bash
python evaluate_materials.py --model_path ./GIR/output_model
```

To render relighted scenes using novel HDR environment maps:
```bash
python relight_all.py --model_path ./GIR/output_model --env_path /path/to/env_map.hdr
```

---

## GIR Baseline Integration
This codebase is developed as an extension of the Gaussian Inverse Rendering (GIR) framework. For details, licenses, and documentation specific to the original GIR base codebase, please refer to the base project guide at [GIR/README.md](GIR/README.md).

---

## Acknowledgments
This repository builds upon [GIR](https://3dgir.github.io/) and leverages priors from [DiffusionRenderer](https://research.nvidia.com/labs/toronto-ai/DiffusionRenderer/). We thank the authors for their open-source contributions.