"""Train the CNN on MNIST and save the weights for the app.

Run once:  python train.py
Creates:   digits_cnn.pt  (and confusion_matrix.png if matplotlib is installed)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import DigitsCNN

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

normalize = transforms.Normalize((0.5,), (0.5,))  # must match app.py
# Light augmentation on the training set only: makes the model more robust to
# the off-centre / slanted / different-sized digits people draw on a canvas.
train_tf = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    normalize,
])
test_tf = transforms.Compose([transforms.ToTensor(), normalize])

train_ds = datasets.MNIST("data", train=True, download=True, transform=train_tf)
test_ds = datasets.MNIST("data", train=False, download=True, transform=test_tf)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256)

model = DigitsCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

EPOCHS = 8
for epoch in range(EPOCHS):
    model.train()
    running = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        loss = criterion(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item()
    print(f"Epoch [{epoch + 1}/{EPOCHS}]  loss: {running / len(train_loader):.4f}")

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        preds = model(images.to(device)).argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)
preds = torch.cat(all_preds).numpy()
labels = torch.cat(all_labels).numpy()

print(f"\nTest accuracy: {100 * (preds == labels).mean():.2f}%")

confusion = np.zeros((10, 10), dtype=int)
for t, p in zip(labels, preds):
    confusion[t, p] += 1
print("Per-digit accuracy:", {d: f"{100 * confusion[d, d] / confusion[d].sum():.1f}%" for d in range(10)})

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    for i in range(10):
        for j in range(10):
            ax.text(j, i, confusion[i, j], ha="center", va="center",
                    color="white" if confusion[i, j] > confusion.max() / 2 else "black", fontsize=7)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=150)
    print("Saved confusion_matrix.png")
except ImportError:
    pass

torch.save(model.cpu().state_dict(), "digits_cnn.pt")
print("Saved digits_cnn.pt")
