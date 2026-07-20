# GPU Server — Hết dung lượng khi cài môi trường

## Lỗi thường gặp

```
OSError: [Errno 28] No space left on device
CondaEnvException: Pip failed
```

## Nguyên nhân

1. **Ổ đĩa home đầy** — PyTorch + CUDA cần **10–15 GB** trống
2. **Cài torch 2 lần** — conda + pip cùng lúc (đã fix trong `environment.yml` mới)

## Bước 1 — Kiểm tra dung lượng

```bash
df -h ~
df -h /tmp
du -sh ~/miniconda3 ~/anaconda3 2>/dev/null
du -sh ~/.cache/pip
```

## Bước 2 — Giải phóng dung lượng

```bash
# Xóa cache conda + pip
conda clean -a -y
pip cache purge
rm -rf ~/.cache/pip

# Xóa env lỗi (nếu create fail giữa chừng)
conda env remove -n ttld-net -y

# Xóa env cũ không dùng
conda env list
conda env remove -n <ten-env-cu> -y

# Xóa runs/checkpoints cũ (nếu có)
rm -rf ~/runs ~/.cache/torch
```

Cần **ít nhất 15 GB trống** trên partition chứa `~`.

## Bước 3 — Cài lại (không duplicate torch)

```bash
cd ~/SDO/ttld-net
conda env create -f environment.yml
conda activate ttld-net

# MMCV (sau khi torch OK)
python scripts/install_mmcv.py --run

# Ultralytics — chỉ khi cần Phase 1
pip install ultralytics

# Gate
python test_env.py --strict-gpu
```

## Nếu vẫn thiếu chỗ

**Option A** — Cài env vào ổ khác có nhiều dung lượng:

```bash
export CONDA_ENVS_PATH=/data/conda/envs   # ổ /data thường rộng hơn
conda env create -f environment.yml
```

**Option B** — Chỉ cài tối thiểu Phase 0:

```bash
conda create -n ttld-net python=3.10 -y
conda activate ttld-net
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements-server.txt
python scripts/install_mmcv.py --run
```

## Verify sau khi xong

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
python -c "from mmcv.ops import MultiScaleDeformableAttention; print('MMCV OK')"
python scripts/check_numpy_torch.py
```

---

## Polars SIGILL / Illegal instruction

```
The following required CPU features were not detected: avx, avx2, ...
Illegal instruction (core dumped)
```

Ultralytics dùng `polars` — bản mặc định cần AVX2. CPU server cũ hoặc VM không có → crash.

```bash
conda activate ttld-net
pip uninstall polars polars-runtime-32 -y
pip install 'polars[rtcompat]>=0.20.0'
python -c "import polars as pl; print(pl.__version__, pl.DataFrame({'a':[1]}))"
```

## NumPy + PyTorch mismatch

```
RuntimeError: Numpy is not available
```

```bash
pip install "numpy>=1.23.5,<2.0" --force-reinstall
python scripts/check_numpy_torch.py
```
