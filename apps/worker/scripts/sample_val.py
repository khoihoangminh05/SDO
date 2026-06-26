"""
Sample a manageable subset of val images for Colab training.
Usage (from apps/worker/):
    python scripts/sample_val.py --count 2000 --seed 42
Output: datasets/dataset_val_sample/  (~3GB → ~500MB after 7z)
"""
import argparse, os, random, shutil

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--seed",  type=int, default=42)
    ap.add_argument("--src",   default="datasets/dataset_test_rgb/rgb/test")
    ap.add_argument("--dst",   default="datasets/dataset_val_sample/rgb/val")
    args = ap.parse_args()

    src = args.src
    dst = args.dst

    # Tìm toàn bộ PNG trong src
    all_pngs = []
    for root, _, files in os.walk(src):
        for f in files:
            if f.lower().endswith(".png"):
                all_pngs.append(os.path.join(root, f))

    print(f"Total val PNGs found: {len(all_pngs)}")

    random.seed(args.seed)
    sampled = random.sample(all_pngs, min(args.count, len(all_pngs)))
    print(f"Sampling {len(sampled)} images -> {dst}")

    os.makedirs(dst, exist_ok=True)
    copied, no_label = 0, 0
    for png_path in sampled:
        txt_path = os.path.splitext(png_path)[0] + ".txt"
        dst_png = os.path.join(dst, os.path.basename(png_path))
        dst_txt = os.path.splitext(dst_png)[0] + ".txt"
        shutil.copy2(png_path, dst_png)
        if os.path.exists(txt_path):
            shutil.copy2(txt_path, dst_txt)
        else:
            # background sample — tạo file rỗng
            open(dst_txt, "w").close()
            no_label += 1
        copied += 1

    total_bytes = sum(
        os.path.getsize(os.path.join(dst, f))
        for f in os.listdir(dst) if f.endswith(".png")
    )
    print(f"Done: {copied} copied, {no_label} background (no label)")
    print(f"Sample size: {total_bytes/1e9:.2f} GB")
    print(f"\nNext: update bstld.yaml val → 'dataset_val_sample/rgb/val'")

if __name__ == "__main__":
    main()
