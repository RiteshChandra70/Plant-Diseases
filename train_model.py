import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_B0_Weights

# -------------------------------
# Device Configuration
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Using device:", device)

# -------------------------------
# Data Transformations
# -------------------------------
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------------
# Dataset and DataLoaders
# -------------------------------
train_data = datasets.ImageFolder('dataset/train', transform=transform_train)
val_data = datasets.ImageFolder('dataset/val', transform=transform_val)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)

print("✅ Loaded dataset successfully!")
print(f"Classes: {train_data.classes}")
print(f"Total Training Images: {len(train_data)}, Validation Images: {len(val_data)}")

# -------------------------------
# Model Setup (EfficientNet)
# -------------------------------
model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

# Unfreeze last few layers for fine-tuning
for name, param in model.features.named_parameters():
    if "6" in name or "7" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Replace final layer with custom classifier
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_features, len(train_data.classes))
)

# -------------------------------
# Training Setup
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
model.to(device)

# -------------------------------
# Training Loop
# -------------------------------
best_acc = 0.0
num_epochs = 15

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i+1}/{len(train_loader)}] "
                  f"Loss: {running_loss / (i+1):.4f}")

    # ---------------------------
    # Validation phase
    # ---------------------------
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = 100 * correct / total
    avg_loss = running_loss / len(train_loader)

    print(f"✅ Epoch [{epoch+1}/{num_epochs}] Completed | "
          f"Train Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}%")

    # Save best model
    if acc > best_acc:
        torch.save(model.state_dict(), 'plant_disease_model.pth')
        best_acc = acc
        print("💾 Model Saved! (Best so far)")

print(f"🎯 Training Finished! Best Validation Accuracy: {best_acc:.2f}%")