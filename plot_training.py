import pandas as pd
import matplotlib.pyplot as plt
history = pd.read_csv("outputs/training_history.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(history["epoch"], history["train_loss"], label="Train")
axes[0].plot(history["epoch"], history["val_loss"], label="Validation")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[1].plot(
    history["epoch"], history["train_accuracy"], label="Train"
)
axes[1].plot(
    history["epoch"], history["val_accuracy"], label="Validation"
)
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (0–1)")
axes[1].legend()
plt.tight_layout()
plt.savefig("outputs/training_curves.png", dpi=150)
plt.show()