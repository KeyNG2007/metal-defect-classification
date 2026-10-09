from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from model import SmallResNet
ROOT = Path(__file__).resolve().parents[1]
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
folder = ROOT / "data/raw/NEU-DET/validation/images/crazing"
image_path = next(folder.glob("*.jpg"))
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
print("真实类别：", image_path.parent.name)
print("预测类别：", id_to_class[predicted_id])