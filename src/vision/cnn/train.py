import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# === CONFIG ===
DATA_DIR = "data/processed/train1"
MODEL_SAVE_PATH = "models/food_classifier.pt"
LABEL_MAP_PATH = "models/label_map.json"
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === TRANSFORMASI GAMBAR DENGAN AUGMENTASI ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

from PIL import Image
def safe_loader(path):
    try:
        return Image.open(path).convert('RGB')
    except:
        print(f"[WARNING] Gagal load gambar: {path}")
        return Image.new("RGB", (224, 224), (0, 0, 0))  # dummy image

# === LOAD DATASET ===
dataset = datasets.ImageFolder(DATA_DIR, transform=transform, loader=safe_loader)
num_classes = len(dataset.classes)

# Simpan label map ke JSON
import json
os.makedirs("models", exist_ok=True)
with open(LABEL_MAP_PATH, "w") as f:
    json.dump(dataset.classes, f)

# Split train/val (80/20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

# === LOAD PRETRAINED MODEL ===
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === TRAINING LOOP ===
print("Mulai training...\n")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # === VALIDASI ===
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total * 100
    print(f"Epoch {epoch+1} - Loss: {train_loss / len(train_loader):.4f} | Val Accuracy: {acc:.2f}%")

# === SIMPAN MODEL ===
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\n Model disimpan ke: {MODEL_SAVE_PATH}")
