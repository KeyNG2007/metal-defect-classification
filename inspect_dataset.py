from pathlib import Path
from collections import Counter
from PIL import Image
import pandas as pd

ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = ROOT / "data" / "raw" / "NEU-DET"

rows = []
bad_files = []
for split in ("train", "validation"):
    split_dir = IMAGE_ROOT / split / "images"
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            try:
                with Image.open(image_path) as image:
                    rows.append({
                        "split": split,
                        "label": class_dir.name,
                        "filename": image_path.name,
                        "width": image.width,
                        "height": image.height,
                        "mode": image.mode,
                    })
            except Exception as exc:
                bad_files.append((str(image_path), str(exc)))

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit(f"没有找到图片，请检查目录：{IMAGE_ROOT}")

output_dir = ROOT / "outputs"
output_dir.mkdir(exist_ok=True)
df.to_csv(output_dir / "image_info.csv", index=False, encoding="utf-8-sig")

print("图片总数：", len(df))
print("类别：", sorted(df["label"].unique()))
print("\n按数据划分和类别统计：")
print(pd.crosstab(df["label"], df["split"]))
print("\n图片尺寸：")
print(df[["width", "height"]].describe())
print("\n图片模式：")
print(Counter(df["mode"]))
print("\n损坏图片：", len(bad_files))
for path, error in bad_files:
    print(path, error)
