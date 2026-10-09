from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import SmallResNet
ROOT = Path(__file__).resolve().parents[1]

checkpoint = torch.load(
    ROOT / "checkpoints/best_model.pth",
    map_location="cpu",
    weights_only=True,
)
transform = transforms.Compose([
    transforms.Resize(
        (checkpoint["image_size"], checkpoint["image_size"])
    ),
    transforms.ToTensor(),
])
dataset = datasets.ImageFolder(
    ROOT / "data/raw/NEU-DET/validation/images",
    transform=transform,
)
assert dataset.class_to_idx == checkpoint["class_to_idx"]
loader = DataLoader(
    dataset, batch_size=16, shuffle=False, num_workers=0
)
model = SmallResNet(num_classes=len(dataset.classes))
model.load_state_dict(checkpoint["model_state"])
model.eval()
matrix = np.zeros((6, 6), dtype=int)
sample_index = 0
with torch.no_grad():
    for images, labels in loader:
        predictions = model(images).argmax(dim=1)
        for true_label, predicted_label in zip(labels, predictions):
            matrix[true_label.item(), predicted_label.item()] += 1
            if true_label.item() == 1 and predicted_label.item() == 3:
                print(
                    "夹杂误判为麻点：",
                    dataset.samples[sample_index][0],
                )
            sample_index += 1
accuracy = np.trace(matrix) / matrix.sum()
print(f"最佳模型验证准确率：{accuracy:.2%}")
for i, name in enumerate(dataset.classes):
    recall = matrix[i, i] / matrix[i].sum()
    print(f"{name}：识别正确 {matrix[i, i]}/"
          f"{matrix[i].sum()}，召回率 {recall:.2%}")
output_dir = ROOT / "outputs"
output_dir.mkdir(exist_ok=True)
pd.DataFrame(
    matrix,
    index=dataset.classes,
    columns=dataset.classes,
).to_csv(output_dir / "confusion_matrix.csv")
fig, ax = plt.subplots(figsize=(9, 7))
image = ax.imshow(matrix, cmap="Blues")
fig.colorbar(image, ax=ax)
ax.set_xticks(range(6))
ax.set_yticks(range(6))
ax.set_xticklabels(dataset.classes, rotation=45, ha="right")
ax.set_yticklabels(dataset.classes)
ax.set_xlabel("Predicted Class")
ax.set_ylabel("True Class")
ax.set_title("Validation Confusion Matrix")
for row in range(6):
    for col in range(6):
        ax.text(
            col, row, str(matrix[row, col]),
            ha="center", va="center",
            color="white" if matrix[row, col] > matrix.max() / 2
            else "black",
        )
plt.tight_layout()
plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
plt.show()