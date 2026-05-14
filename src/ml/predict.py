# src/ml/predict.py

import joblib
import numpy as np
import os

MODEL_PATH = "models/stunting_predictor.pkl"

label_map = {
    0: "Normal",
    1: "Risiko Anemia",
    2: "Risiko Stunting",
    3: "Risiko CAS (Anemia + Stunting)"
}

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model belum ditemukan di: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

def predict_risiko_stunting(usia_bulan, berat_kg, tinggi_cm, jenis_kelamin, anemia, dds_score):
    jk_encoded = 1 if jenis_kelamin.upper() == 'L' else 0
    fitur = np.array([[usia_bulan, berat_kg, tinggi_cm, jk_encoded, anemia, dds_score]])
    hasil = model.predict(fitur)[0]
    return label_map[hasil]
