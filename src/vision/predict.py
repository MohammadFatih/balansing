import torch
from torchvision import models, transforms
from PIL import Image
import os
import json

# === KONFIGURASI ===
MODEL_PATH = "models/food_classifier.pt"
LABEL_PATH = "models/label_map.json"
IMAGE_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === TRANSFORMASI GAMBAR ===
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === LOAD LABEL MAP ===
if not os.path.exists(LABEL_PATH):
    raise FileNotFoundError(f"File label_map.json tidak ditemukan di: {LABEL_PATH}")

with open(LABEL_PATH, "r") as f:
    label_map = json.load(f)

# === LOAD MODEL ===
def load_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(label_map))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# === FUNGSI PREDIKSI GAMBAR ===
def predict_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        label_idx = predicted.item()

    return label_map[label_idx]
