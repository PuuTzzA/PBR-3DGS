# GIR Windows Setup Guide:

This guide documents how to set up and run GIR on **Windows** with:
- Conda / Miniconda
- NVIDIA GPU with modern drivers
- CUDA Toolkit installed
- Visual Studio Build Tools (MSVC compiler)
- PyTorch with CUDA 12.1

The Linux README was used as the starting point. Below are all Windows-specific adaptations.

---

## Requirements

- Windows 10/11
- NVIDIA GPU with recent drivers
- [CUDA Toolkit 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive) installed (make sure `nvcc` is on your PATH)
- [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the **"Desktop development with C++"** workload installed
- [Conda / Miniconda](https://docs.anaconda.com/miniconda/) installed
- GIR repository already cloned

### Verify prerequisites

Open a **PowerShell** or **Developer Command Prompt** and run:

```powershell
nvcc --version
cl
```

Both commands should be found. If `cl` is not found, you need to either:
- Open a **"Developer PowerShell for VS 2022"** (from Start menu), **or**
- Run this before proceeding (adjust path if needed):

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1"
```

---

## 1. Create Conda Environment

Using the provided Windows environment file:

```powershell
conda env create -f environment_windows.yml
conda activate gir
```

Or manually:

```powershell
conda create -n gir python=3.10 -y
conda activate gir
```

---

## 2. Install Python Dependencies

If you created the environment manually (not from `environment_windows.yml`), install these:

```powershell
pip install --upgrade pip setuptools wheel

pip install `
    ninja `
    setuptools==69.5.1 `
    "numpy<2.0.0" `
    tqdm `
    plyfile `
    "imageio[full]" `
    scipy `
    matplotlib `
    pillow `
    pyyaml `
    tensorboard `
    einops `
    tabulate
```

---

## 3. Install PyTorch (CUDA 12.1)

```powershell
pip install `
    torch==2.2.2 `
    torchvision==0.17.2 `
    torchaudio==2.2.2 `
    --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output: `True`

---

## 4. Configure CUDA Environment Variables

Find your CUDA installation (typically `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1`):

```powershell
where nvcc
```

Set environment variables for the current session:

```powershell
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
$env:CUDA_PATH = $env:CUDA_HOME
$env:PATH = "$env:CUDA_HOME\bin;$env:PATH"
```

> **Tip:** You can also set these permanently via **System Properties → Environment Variables**.

Verify:

```powershell
Test-Path "$env:CUDA_HOME\bin\nvcc.exe"
```

---

## 5. Navigate to GIR Directory

```powershell
cd D:\adl4cv\PBR-3DGS\GIR
```

---

## 6. Build diff-gaussian-rasterization

> **Important:** You must use `--no-build-isolation` so that the build process can find PyTorch.

```powershell
cd submodules\diff-gaussian-rasterization

# Clean old builds if any
if (Test-Path build) { Remove-Item -Recurse -Force build }
if (Test-Path *.egg-info) { Remove-Item -Recurse -Force *.egg-info }
if (Test-Path dist) { Remove-Item -Recurse -Force dist }

pip install . --no-cache-dir --no-build-isolation
```

### Troubleshooting: `cl.exe` not found

If the build fails because MSVC is not found, make sure you are in a **Developer PowerShell for VS 2022** or have run:

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1"
```

### Troubleshooting: CUDA version mismatch

Make sure the CUDA Toolkit version matches what PyTorch was built with (CUDA 12.1). You can check:

```powershell
python -c "import torch; print(torch.version.cuda)"
```

---

## 7. Build simple-knn

On Windows, `simple-knn` already has a Windows-specific compiler flag (`/wd4624`) in its `setup.py`. However, you may still need to add `#include <float.h>` to source files if the build fails.

```powershell
cd ..\simple-knn
```

### Apply patch (if build fails with missing `FLT_MAX` or `float.h`)

Run this Python one-liner to prepend `#include <float.h>` to all `.cu`, `.cpp`, and `.h` files:

```powershell
python -c "
import glob, os
for ext in ['*.cu', '*.cpp', '*.h']:
    for f in glob.glob(ext):
        content = open(f, 'r').read()
        if '#include <float.h>' not in content:
            open(f, 'w').write('#include <float.h>\n' + content)
            print(f'Patched {f}')
"
```

Clean and install:

```powershell
if (Test-Path build) { Remove-Item -Recurse -Force build }
if (Test-Path *.egg-info) { Remove-Item -Recurse -Force *.egg-info }
if (Test-Path dist) { Remove-Item -Recurse -Force dist }

pip install . --no-cache-dir --no-build-isolation
```

---

## 8. Build envlight

```powershell
cd ..\envlight

if (Test-Path build) { Remove-Item -Recurse -Force build }
if (Test-Path *.egg-info) { Remove-Item -Recurse -Force *.egg-info }
if (Test-Path dist) { Remove-Item -Recurse -Force dist }

pip install . --no-build-isolation
```

---

## 9. Install nvdiffrast

```powershell
cd ..\..

git clone https://github.com/NVlabs/nvdiffrast.git

cd nvdiffrast

pip install . --no-build-isolation
```

> **Note:** If nvdiffrast was already cloned into the repo, just `cd` into it and install.

---

## 10. Download FreeImage Backend

```powershell
python -c "import imageio; imageio.plugins.freeimage.download()"
```

---

## 11. Run Training

Navigate back to the GIR root:

```powershell
cd D:\adl4cv\PBR-3DGS\GIR
```

Example training command:

```powershell
python train.py `
    -s D:\path\to\dataset `
    -m .\output_model `
    --eval `
    --port 6009
```

---

## Every New Terminal Session

1. **Activate the conda environment:**

```powershell
conda activate gir
```

2. **Set CUDA paths (if not set permanently):**

```powershell
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
$env:CUDA_PATH = $env:CUDA_HOME
$env:PATH = "$env:CUDA_HOME\bin;$env:PATH"
```

3. **Ensure MSVC is available** (if not using Developer PowerShell):

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1"
```

---

## Common Problems on Windows

### `cl.exe` not found / No C++ compiler

**Cause:** Visual Studio Build Tools not installed or not activated in the current shell.

**Fix:** Install **Visual Studio 2022 Build Tools** with the **"Desktop development with C++"** workload, then open a **Developer PowerShell for VS 2022** or run `Launch-VsDevShell.ps1`.

### `nvcc fatal: Unsupported gpu architecture 'compute_...'`

**Cause:** PyTorch CUDA version and system CUDA Toolkit version mismatch.

**Fix:** Install CUDA Toolkit 12.1 to match the PyTorch build.

### NumPy 2.x Errors

```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

**Fix:**

```powershell
pip install "numpy<2.0.0"
```

### `ModuleNotFoundError: No module named 'torch'` during build

**Cause:** pip build isolation prevents finding the installed torch.

**Fix:** Always use `--no-build-isolation` when building CUDA extensions:

```powershell
pip install . --no-build-isolation
```

### `OSError: [WinError 126]` when importing CUDA extensions

**Cause:** CUDA runtime DLLs not on PATH.

**Fix:** Make sure `CUDA_HOME\bin` is on your PATH (see step 4).

### Long path errors

**Cause:** Windows default path length limit of 260 characters.

**Fix:** Enable long paths in the registry:

```powershell
# Run as Administrator
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Then restart your terminal.

