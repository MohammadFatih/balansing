# src/vision/predict.py

import torch
from torchvision import models, transforms
from PIL import Image
import os

MODEL_PATH = "models/food_classifier.pt"
NUM_CLASSES = 9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_labels = [
    "Berpati", "Sayur_Hijau", "Sayur_Buah_VitA", "Buah_Lain",
    "Daging_Organ", "Daging_Ikan", "Telur", "Kacang_Biji", "Susu_Olahan"
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model CNN tidak ditemukan di: {MODEL_PATH}")
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

def predict_image(img_path):
    """
    Prediksi kategori makanan dari gambar input.
    """
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
        label_idx = pred.item()
        return class_labels[label_idx]
