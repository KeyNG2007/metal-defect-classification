from pathlib import Path
from collections import defaultdict
import hashlib

from PIL import Image


ROOT = Path(__file__).resolve().parent
data_root = ROOT / "data/raw/NEU-DET"
groups = defaultdict(list)
image_count = 0

for split in ["train", "validation"]:
    image_dir = data_root / split / "images"
    if not image_dir.is_dir():
        raise SystemExit(f"找不到图片目录：{image_dir}")

    for path in sorted(image_dir.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue

        with Image.open(path) as image:
            image = image.convert("RGB")
            content = str(image.size).encode() + image.tobytes()

        fingerprint = hashlib.sha256(content).hexdigest()
        groups[fingerprint].append((split, str(path.relative_to(ROOT))))
        image_count += 1

if image_count == 0:
    raise SystemExit("没有找到图片，请检查数据目录。")

duplicates = [items for items in groups.values() if len(items) > 1]
cross_split = [
    items for items in duplicates
    if len({split for split, path in items}) > 1
]

lines = [
    f"已解码图片数：{image_count}",
    f"完全相同图片组数：{len(duplicates)}",
    f"跨训练/验证集重复组数：{len(cross_split)}",
    "检查范围：转换为 RGB 后像素完全相同；不排除近似图片或同一场景泄漏。",
]

for items in duplicates:
    lines.append("\n重复组：")
    for split, path in items:
        lines.append(f"{split}: {path}")

report = "\n".join(lines)
output_dir = ROOT / "outputs"
output_dir.mkdir(exist_ok=True)
(output_dir / "duplicate_check.txt").write_text(report, encoding="utf-8")
print(report)
print("\n报告已保存：outputs/duplicate_check.txt")
