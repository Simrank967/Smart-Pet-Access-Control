import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from utils.dataset_loader import (
    train_loader,
    val_loader,
    class_names
)

# =====================================================
# Device Configuration
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# =====================================================
# Load Pretrained ResNet18
# =====================================================

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze pretrained layers
for param in model.parameters():
    param.requires_grad = False

# Replace final classification layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)

model = model.to(device)

# =====================================================
# Loss and Optimizer
# =====================================================

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=0.001
)

# =====================================================
# Training Parameters
# =====================================================

EPOCHS = 10

best_accuracy = 0.0

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# =====================================================
# Training Loop
# =====================================================

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = (torch.sigmoid(outputs) >= 0.5).float()

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

    train_accuracy = 100 * correct / total
    avg_loss = running_loss / len(train_loader)

    # =================================================
    # Validation
    # =================================================

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)

            predictions = (torch.sigmoid(outputs) >= 0.5).float()

            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = 100 * val_correct / val_total

    # =================================================
    # Save Best Model
    # =================================================

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "models/best_model.pth"
        )

        print("✅ Best model saved!")

    # =================================================
    # Epoch Results
    # =================================================

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"| Loss: {avg_loss:.4f} "
        f"| Train Accuracy: {train_accuracy:.2f}% "
        f"| Validation Accuracy: {val_accuracy:.2f}%"
    )

# =====================================================
# Training Finished
# =====================================================

print("\n========================================")
print("🎉 Training Completed Successfully!")
print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
print("Best model saved to models/best_model.pth")
print("========================================")