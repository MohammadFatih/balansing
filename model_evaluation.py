import os
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

"""
SCRIPT EVALUASI MODEL BALANSING-AI (FIXED & CLEAN VERSION)
---------------------------------------------------------
Fitur:
- Input path model (.pt) & data.yaml via input()
- Evaluasi otomatis (Precision, Recall, mAP50, mAP50-95)
- Confusion Matrix
- Save plot hasil prediksi ke folder pred_plots/
"""

def evaluate_model(model_path: str, data_yaml: str):
    print("[INFO] Loading model: {}".format(model_path))
    model = YOLO(model_path)

    print("[INFO] Running evaluation on validation set...")
    results = model.val(data=data_yaml, save_json=True, save_hybrid=True)

    metrics = {
        "precision": results.box.mp,
        "recall": results.box.mr,
        "map50": results.box.map50,
        "map50_95": results.box.map,
    }

    print("\n==================== MODEL PERFORMANCE ====================")
    for k, v in metrics.items():
        print("{}: {:.4f}".format(k, v))
    print("==========================================================\n")

    # CONFUSION MATRIX
    print("[INFO] Generating confusion matrix...")

    preds = results.results
    y_true, y_pred = [], []

    for res in preds:
        if res.boxes is None:
            continue

        pred_classes = res.boxes.cls.cpu().numpy().astype(int)
        y_pred.extend(pred_classes)

        # Load ground truth
        img_path = res.path
        if img_path is None:
            continue

        label_path = (
            img_path.replace("images", "labels")
            .replace(".jpg", ".txt")
            .replace(".png", ".txt")
        )

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                true_classes = [int(line.split()[0]) for line in f.readlines()]
                y_true.extend(true_classes)

    if len(y_true) > 0 and len(y_pred) > 0:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(xticks_rotation=45)
        plt.title("Confusion Matrix - Balansing AI Model")
        plt.tight_layout()

        os.makedirs("pred_plots", exist_ok=True)
        plt.savefig("pred_plots/confusion_matrix.png", dpi=300)
        plt.close()

        print("[INFO] Confusion matrix saved to pred_plots/confusion_matrix.png")
    else:
        print("[WARNING] Not enough data to generate confusion matrix.")

    return metrics


def predict_and_save(model_path: str, data_yaml: str):
    print("[INFO] Generating prediction plots...")
    model = YOLO(model_path)

    os.makedirs("pred_plots", exist_ok=True)

    model.predict(
        source="valid/images",
        save=True,
        project="pred_plots",
        name="pred_images"
    )

    print("[INFO] Prediction visualizations saved under pred_plots/pred_images/")


if __name__ == "__main__":
    print("=== BALANSING-AI MODEL EVALUATION ===")

    model_path = input("Masukkan path file model (.pt): ")
    data_path = input("Masukkan path file data.yaml: ")

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found: {}".format(model_path))

    if not os.path.exists(data_path):
        raise FileNotFoundError("data.yaml file not found: {}".format(data_path))

    evaluate_model(model_path, data_path)
    predict_and_save(model_path, data_path)

    print("[INFO] Evaluation finished.")
