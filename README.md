# Causal Depthwise Conv1D (1.7.0) for Legacy CUDA on Windows

A Windows-native implementation of causal depthwise 1D convolution, adapted to support legacy NVIDIA GPU architectures within a modern PyTorch environment.

The project focuses on retaining CUDA support for older GPUs, including Kepler, Maxwell, Pascal and subsequent architectures.

## Features

- Supports FP32, FP16 and BF16 (subject to hardware and kernel constraints).
- Kernel sizes 2, 3 and 4.
- Native Windows CUDA compilation using MSVC and NVIDIA NVCC.
- Explicit CUDA architecture targeting based on the installed PyTorch build.
- Reproducible wheel builds without requiring editable installation.

## 1. Reference build configuration

The following configuration is the reference environment used for development and testing.

| Component | Version / configuration |
|---|---|
| PyTorch | 2.7.1 |
| Python | 3.11 |
| CUDA Toolkit | 11.8 |
| cuDNN | 8.7.0 |
| Visual Studio | 2019 / MSVC v142 |
| Intel oneAPI | 2024.2.1.101 |
| NVIDIA driver | 472.50 |
| Target architectures | 3.5; 3.7; 5.0; 5.2; 6.0; 6.1; 7.0; 7.5 |

The goal is to retain support for legacy GPU architectures while running a modern PyTorch stack on Windows.

**Build architecture discovery**

Unlike conventional CUDA extension build configurations, this project does not require CUDA architectures to be manually maintained in `setup.py`.

Instead, the build script:

1. Initializes the Visual Studio compiler environment.
2. Selects the intended Python environment.
3. Queries the installed PyTorch installation using `torch.cuda.get_arch_list()`.
4. Converts the reported architecture identifiers into `TORCH_CUDA_ARCH_LIST`.
5. Passes the resulting architecture list to the extension build.

Consequently, the architecture targets are derived from the actual PyTorch installation rather than being independently hardcoded in the package configuration.

`setup.py` is responsible for interpreting the supplied configuration and compiling the extension accordingly. The build script determines the environment and the targets.

## 2. High-performance prebuilt wheels
- for Pytorch see https://github.com/theIvanR/torch-on-clunkers
- for causal-conv1d see

> For additional Kepler GPU performance modifications, see [Kepler on Steroids](https://drive.google.com/file/d/1WeO1uz6DdOiAQooTUG2ouHa9ajZYDG0p/view?usp=drive_link).

## 3. Prerequisites

The reference development environment uses:

- Git
- Miniconda
- Visual Studio 2019 with the MSVC v142 C++ toolchain
- NVIDIA CUDA Toolkit 11.8
- NVIDIA driver 472.50
- PyTorch 2.7.1 (see link above)
- Intel oneAPI 2024.2.1.101
- cuDNN 8.7.0

**Important:** For legacy NVIDIA driver compatibility, install the CUDA Toolkit without replacing your existing display driver. Verify that the installed CUDA toolkit, driver and cuDNN combination is compatible before proceeding.

Install cuDNN into the appropriate CUDA directories according to its Windows installation instructions.

The CUDA compiler (`nvcc`) and Microsoft C++ compiler (`cl`) must both be available to the build process.

## 4. Create the build environment

Open an Anaconda Prompt and create a dedicated Python environment:

```bat
conda create -n py311_debug python=3.11
conda activate py311_debug
```

Install the required PyTorch build and verify that it is available:

```bat
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_arch_list())"
```

The installed PyTorch build must include the CUDA architectures intended for the extension.

## 5. Build the extension

Clone the repository:

```bat
git clone https://github.com/theIvanr/causal-conv1d.git
cd causal-conv1d
```

Run the Windows build script:

```bat
build_cida_win.bat
```

The script performs the following operations:

- Initializes the Visual Studio 2019 x64 compiler environment.
- Selects the configured Python interpreter.
- Verifies the PyTorch and CUDA environment.
- Checks for the availability of MSVC and NVCC.
- Detects the CUDA architectures compiled into the installed PyTorch build.
- Exports those architectures through `TORCH_CUDA_ARCH_LIST`.
- Builds a distributable wheel using `pip wheel`.

The build uses:

```bat
python -m pip wheel . --no-build-isolation --no-deps --no-cache-dir -w dist -v
```

This produces a wheel in the `dist` directory without installing the package or modifying the active Python environment.

The architecture selection is handled by the build script; `setup.py` simply consumes the supplied configuration.

## 7. Install the wheel

After a successful build, install the generated wheel:

```bat
python -m pip install --no-deps dist\causal_conv1d-*.whl
```

OPTIONAL: Run the test suite to verify the extension.

```bat
python test_causal_conv1d.py
```

## 8. Usage

```python
from causal_conv1d import causal_conv1d_fn
```

Function interface:

```python
def causal_conv1d_fn(x, weight, bias=None, activation=None):
    """
    x:          (batch, dim, seqlen)
    weight:     (dim, width)
    bias:       (dim,)
    activation: None, "silu" or "swish"

    Returns:
        out:    (batch, dim, seqlen)
    """
```

The operation is equivalent to:

```python
import torch.nn.functional as F

F.conv1d(
    x,
    weight.unsqueeze(1),
    bias,
    padding=width - 1,
    groups=dim
)[..., :seqlen]
```

## 9. AMD GPUS? 
- coming later
