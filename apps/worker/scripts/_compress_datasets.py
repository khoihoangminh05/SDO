"""Run 7z compression for both train and val datasets in parallel."""
import subprocess, os, time

z7     = r"C:\Program Files\7-Zip\7z.exe"
worker = r"d:\LapTrinh\System\SDO\apps\worker"

jobs = [
    {
        "name": "val_sample",
        "src":  os.path.join(worker, "datasets", "dataset_val_sample"),
        "out":  os.path.join(worker, "bstld_val.zip"),
        "log":  os.path.join(worker, "7z_val.log"),
    },
    {
        "name": "train",
        "src":  os.path.join(worker, "datasets", "dataset_train_rgb"),
        "out":  os.path.join(worker, "bstld_train.zip"),
        "log":  os.path.join(worker, "7z_train.log"),
    },
]

procs = []
for j in jobs:
    cmd = [z7, "a", "-tzip", "-mx=1", "-v2g", j["out"], j["src"]]
    lf  = open(j["log"], "w", encoding="utf-8")
    p   = subprocess.Popen(cmd, stdout=lf, stderr=lf)
    procs.append((j, p, lf))
    print(f"Started {j['name']} (pid={p.pid})")

print("Both jobs running. Waiting for completion...")
t0 = time.time()

for j, p, lf in procs:
    p.wait()
    lf.close()
    elapsed = time.time() - t0

    with open(j["log"], encoding="utf-8") as f:
        lines = f.readlines()

    summary = [l.strip() for l in lines if any(k in l for k in ("Ok", "Error", "error", "size", "Archive"))]
    print(f"\n[{j['name']}] exit={p.returncode} ({elapsed:.0f}s)")
    for s in summary[-6:]:
        print(" ", s)

    d    = os.path.dirname(j["out"])
    base = os.path.basename(j["out"])
    parts = sorted(f for f in os.listdir(d) if f.startswith(base))
    for pf in parts:
        sz = os.path.getsize(os.path.join(d, pf))
        print(f"  {pf}  →  {sz/1e9:.2f} GB")

print("\nAll done.")
