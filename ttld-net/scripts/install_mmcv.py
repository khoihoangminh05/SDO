#!/usr/bin/env python3
"""Detect torch/CUDA version and print or run the correct MMCV install command."""

from __future__ import annotations

import platform
import subprocess
import sys


def mmcv_install_cmd() -> tuple[str | None, str]:
    try:
        import torch
    except ImportError:
        return None, "PyTorch not installed"

    torch_ver = ".".join(torch.__version__.split(".")[:2])
    cuda_ver = torch.version.cuda

    if not torch.cuda.is_available() or not cuda_ver:
        return None, "CUDA PyTorch required (conda install pytorch-cuda)"

    # mmcv wheels: cu118, cu121, cu124 ...
    major_minor = cuda_ver.split(".")[:2]
    if major_minor[0] == "11":
        cuda_tag = "cu118"
    elif float(cuda_ver) >= 12.4:
        cuda_tag = "cu124"
    else:
        cuda_tag = "cu121"

    index_url = (
        f"https://download.openmmlab.com/mmcv/dist/{cuda_tag}/torch{torch_ver}/index.html"
    )
    cmd = f'pip install "mmcv==2.1.0" -f {index_url}'
    return cmd, index_url


def main() -> int:
    print("MMCV Install Helper\n" + "=" * 40)
    print(f"Python:   {sys.version.split()[0]}")
    print(f"Platform: {platform.system()}")

    try:
        import torch

        print(f"PyTorch:  {torch.__version__}")
        print(f"CUDA:     {torch.cuda.is_available()} ({torch.version.cuda})")
        if torch.cuda.is_available():
            print(f"GPU:      {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("ERROR: pip/conda install pytorch first")
        return 1

    cmd, info = mmcv_install_cmd()
    if cmd is None:
        print(f"\n❌ {info}")
        if platform.system() == "Windows":
            print("   MMCV Deformable Attention chỉ cài trên Linux + GPU.")
            print("   Máy dev Windows: python test_env.py --dev")
        return 1

    print(f"\nWheel index: {info}")
    print(f"\nRun:\n  {cmd}\n")
    print('Verify:\n  python -c "from mmcv.ops import MultiScaleDeformableAttention; print(\'OK\')"')

    if "--run" in sys.argv:
        print("\nInstalling mmcv...")
        return subprocess.call(cmd, shell=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
