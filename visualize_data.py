import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("outputs/image_info.csv")

counts = data["label"].value_counts()

print(counts)

counts.plot(kind="bar")

plt.title("Metal Surface Defect Distribution")
plt.xlabel("Defect Type")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.savefig("outputs/class_distribution.png")
plt.show()