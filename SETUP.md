Clone with this:
```
git clone --recursive --shallow-submodules https://github.com/PuuTzzA/PBR-3DGS.git
```

# GIR Local Setup Guide (Tested Working Setup)

This guide documents the setup that actually worked locally for GIR on a modern Linux system with:
- Conda
- modern NVIDIA drivers
- CUDA toolkit installed
- newer GCC versions
- modern PyTorch

The original README did **not** work directly due to:
- outdated PyTorch/CUDA versions
- NumPy 2 incompatibility
- CUDA extension build issues
- missing `simple-knn` patch
- pip build isolation issues
- incorrect CUDA path assumptions

This guide reflects the fixes that were required to get the project compiling successfully.

---

# Requirements

- Linux (Ubuntu recommended)
- NVIDIA GPU
- NVIDIA driver installed
- CUDA toolkit installed (`nvcc` available)
- Conda / Miniconda installed
- GIR repository already cloned

Verify CUDA:

```bash
which nvcc
nvcc --version
```

---

# 1. Create Conda Environment

If `environment.yml` is available:

```bash
conda env create -f environment.yml
conda activate gir
```

Otherwise:

```bash
conda create -n gir python=3.10 -y
conda activate gir
```

---

# 2. Install System Dependencies

Install required build tools:

```bash
sudo apt update

sudo apt install -y \
    build-essential \
    ninja-build \
    gcc-11 \
    g++-11
```

Use GCC 11 explicitly:

```bash
export CC=gcc-11
export CXX=g++-11
```

This is important because newer GCC versions may break CUDA extension compilation.

---

# 3. Install Python Dependencies

IMPORTANT:
GIR currently does not work properly with NumPy 2.x.

```bash
pip install --upgrade pip setuptools wheel

pip install \
    ninja \
    setuptools==69.5.1 \
    "numpy<2.0.0" \
    tqdm \
    plyfile \
    imageio[full]
```

---

# 4. Install PyTorch

The original repository versions (`torch 1.12 + cu116`) did not work reliably on a modern system.

The following versions worked:

```bash
pip install \
  torch==2.2.2 \
  torchvision==0.17.2 \
  torchaudio==2.2.2 \
  --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output:

```text
True
```

---

# 5. Configure CUDA Paths

One issue encountered was that CUDA paths were incorrectly assumed.

Find the actual CUDA installation:

```bash
readlink -f $(which nvcc)
```

Example output:

```text
/usr/lib/nvidia-cuda-toolkit/bin/nvcc
```

Set CUDA environment variables accordingly:

```bash
export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit

export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export CUDACXX=$CUDA_HOME/bin/nvcc
```

Verify:

```bash
ls $CUDA_HOME/bin/nvcc
```

---

# 6. Go to GIR Repository

```bash
cd GIR
```

---

# 7. Build diff-gaussian-rasterization

IMPORTANT:
`pip install .` alone failed because pip build isolation could not access torch.

This worked:

```bash
cd submodules/diff-gaussian-rasterization

rm -rf build *.egg-info dist

pip install . --no-cache-dir --no-build-isolation
```

---

# 8. Build simple-knn

`simple-knn` failed initially due to a missing `<float.h>` include.

Go to the submodule:

```bash
cd ../simple-knn
```

Apply patch:

```bash
find . -type f \( -name "*.cu" -o -name "*.cpp" -o -name "*.h" \) \
  -exec sed -i '1i #include <float.h>' {} +
```

Clean old builds:

```bash
rm -rf build *.egg-info dist
```

Install:

```bash
pip install . --no-cache-dir --no-build-isolation
```

---

# 9. Build envlight

```bash
cd ../envlight

rm -rf build *.egg-info dist

pip install . --no-build-isolation
```

---

# 10. Install nvdiffrast

```bash
cd ../../

git clone https://github.com/NVlabs/nvdiffrast.git

cd nvdiffrast

pip install . --no-build-isolation
```

---

# 11. Download FreeImage Backend

```bash
python
```

```python
import imageio
imageio.plugins.freeimage.download()
```

---

# 12. Run Training

From the GIR root:

```bash
cd ~/GIR
```

Example:

```bash
python train.py \
  -s /path/to/dataset \
  -m ./output_model \
  --eval \
  --port 6009
```

---

# Common Problems Encountered

## NumPy 2.x Errors

Example:

```text
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

Fix:

```bash
pip install "numpy<2.0.0"
```

---

## Torch Missing During Build

Example:

```text
ModuleNotFoundError: No module named 'torch'
```

Cause:
pip build isolation.

Fix:

```bash
pip install . --no-build-isolation
```

---

## nvcc Not Found

Example:

```text
No such file or directory: '/usr/local/cuda/bin/nvcc'
```

Cause:
incorrect CUDA path assumptions.

Fix:
use the real path from:

```bash
readlink -f $(which nvcc)
```

and export `CUDA_HOME` accordingly.

---

# Every New Terminal Session

Activate environment:

```bash
conda activate gir
```

Re-export CUDA paths:
(Note): this only has to be done sometimes though
```bash
export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit

export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export CUDACXX=$CUDA_HOME/bin/nvcc
```
