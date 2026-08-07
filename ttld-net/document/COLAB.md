# TTLD-Net on Google Colab (Pro / Pro+)

Optional limited-budget training when a dedicated GPU server is unavailable.

> **Paper protocol** uses full YAML epochs / dataset (no `--fast` / `--proplus`).  
> Colab profiles **subset the data and cap batches** — disclose this in any report.

| Flag | Typical use | Approx. cost / model |
|------|-------------|----------------------|
| `--fast` | Smoke / debug | ~1 h |
| `--proplus` | Practical ablation on A100/L4 | ~3–5 h |

Defined in `utils/fast_train.py`.

---

## One-time packaging (Windows)

```powershell
cd D:\LapTrinh\System\SDO
powershell -ExecutionPolicy Bypass -File ttld-net\scripts\pack_colab.ps1
```

Upload to Drive `SDO_train/`:

- `ttld_colab_code.zip`
- `bstld_train.zip` (+ splits if any)
- `bstld_val.zip`

Open `scripts/ttld_net_colab.ipynb` with a GPU runtime.

---

## Training

```bash
cd /content/SDO/ttld-net

# M1 only
python scripts/run_ablation.py --configs m1_shallow --device 0 --proplus

# Continue from M2
bash scripts/continue_m2.sh --proplus
# or: python scripts/run_ablation.py --configs m2_focal --device 0 --proplus
```

**After each model finishes**, sync to Drive immediately (notebook CELL 8).  
Colab `/content` is ephemeral — checkpoints are lost on runtime reset if unsynced.

Startup sanity checks:

- `[YOLO26Backbone] pretrained … matched`
- `Optimizer params: backbone=… other=…` (backbone ≫ 0)

---

## Expected behaviour (M1, `--proplus`)

Healthy run (order of magnitude): AP50 ≈ 0.2+, recall ≈ 0.5+, thousands of TP at conf 0.05.  
Precision stays low by design for the high-recall Stage-1 generator.
