import torch
import torch.nn as nn
from torchvision import models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

from utils.dataset_loader import (
    test_loader
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
model = models.resnet18(weights=None)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)

model.load_state_dict(
    torch.load(
        "models/best_model.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()
all_labels = []
all_predictions = []
with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = (
            torch.sigmoid(outputs) >= 0.5
        ).float()

        all_predictions.extend(
            predictions.cpu().numpy().flatten()
        )

        all_labels.extend(
            labels.numpy()
        )
accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions
)

recall = recall_score(
    all_labels,
    all_predictions
)

f1 = f1_score(
    all_labels,
    all_predictions
)
print("\n========== Test Results ==========")

print(f"Accuracy : {accuracy*100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
cm = confusion_matrix(
    all_labels,
    all_predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Cat", "Dog"]
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

import os

os.makedirs("screenshots", exist_ok=True)

plt.savefig(
    "screenshots/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

print("Confusion Matrix saved successfully!")

plt.close()