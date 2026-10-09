from pathlib import Path
import argparse
import torch
from PIL import Image
from torchvision import transforms
from model import SmallResNet
ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="预测金属表面缺陷类别")
parser.add_argument("image", help="要预测的图片路径")
args = parser.parse_args()
image_path = Path(args.image)
if not image_path.is_file():
    raise SystemExit(f"找不到图片：{image_path}")

checkpoint = torch.load(
    ROOT / "checkpoints/best_model.pth",
    map_location="cpu",
    weights_only=True,
)
model = SmallResNet(
    num_classes=len(checkpoint["class_to_idx"])
)
model.load_state_dict(checkpoint["model_state"])
model.eval()
transform = transforms.Compose([
    transforms.Resize(
        (checkpoint["image_size"], checkpoint["image_size"])
    ),
    transforms.ToTensor(),
])
with Image.open(image_path) as image:
    tensor = transform(image.convert("RGB"))
tensor = tensor.unsqueeze(0)
with torch.no_grad():
    scores = model(tensor)
    predicted_id = scores.argmax(dim=1).item()
id_to_class = {
    number: name
    for name, number in checkpoint["class_to_idx"].items()
}
print("图片：", image_path.name)
print("所在文件夹：", image_path.parent.name)
print("预测类别：", id_to_class[predicted_id])
