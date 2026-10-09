from pathlib import Path
import pandas as pd
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import SmallResNet
ROOT = Path(__file__).resolve().parents[1]
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
train_data = datasets.ImageFolder(
    ROOT / "data/raw/NEU-DET/train/images",
    transform=transform,
)
val_data = datasets.ImageFolder(
    ROOT / "data/raw/NEU-DET/validation/images",
    transform=transform,
)
assert train_data.class_to_idx == val_data.class_to_idx
train_loader = DataLoader(
    train_data, batch_size=16, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_data, batch_size=16, shuffle=False, num_workers=0
)
model = SmallResNet(num_classes=6)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
save_dir = ROOT / "checkpoints"
save_dir.mkdir(exist_ok=True)
best_accuracy = -1.0
history = []
for epoch in range(15):
    start = time.time()
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        scores = model(images)
        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * labels.size(0)
        correct += (scores.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    train_loss = loss_sum / total
    train_accuracy = correct / total
    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            scores = model(images)
            loss = criterion(scores, labels)

            val_loss_sum += loss.item() * labels.size(0)
            val_correct += (
                scores.argmax(dim=1) == labels
            ).sum().item()
            val_total += labels.size(0)
    val_loss = val_loss_sum / val_total
    val_accuracy = val_correct / val_total
    history.append({
    "epoch": epoch + 1,
    "train_loss": train_loss,
    "train_accuracy": train_accuracy,
    "val_loss": val_loss,
    "val_accuracy": val_accuracy,
})
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save({
            "model_state": model.state_dict(),
            "class_to_idx": train_data.class_to_idx,
            "image_size": 64,
            "epoch": epoch + 1,
            "val_accuracy": val_accuracy,
        }, save_dir / "best_model.pth")
    print(
        f"第 {epoch + 1} 轮 | "
        f"训练损失 {train_loss:.4f} | "
        f"训练准确率 {train_accuracy:.2%} | "
        f"验证损失 {val_loss:.4f} | "
        f"验证准确率 {val_accuracy:.2%} | "
        f"耗时 {time.time() - start:.1f} 秒",
        flush=True,
    )
print("最佳模型已保存：", save_dir / "best_model.pth")
output_dir = ROOT / "outputs"
output_dir.mkdir(exist_ok=True)
pd.DataFrame(history).to_csv(
    output_dir / "training_history.csv",
    index=False,
)