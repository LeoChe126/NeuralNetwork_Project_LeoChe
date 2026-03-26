#-------------------------- Neural Network Project ----------------------------#

###--------------Model Evaluation------------------------


import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score,
                             roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay)
from model import HeartNet
from dataset import get_dataloaders

def evaluate(model_path="best_model.pth"):

    model = HeartNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    _, val_loader = get_dataloaders("data/heart.csv")

    all_probs  = []   # raw probabilities (for AUC)
    all_preds  = []   # binary predictions 0/1 (for accuracy, F1)
    all_labels = []   # true labels

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            probs = model(X_batch)              # shape (batch, 1), values 0–1
            preds = (probs >= 0.5).float()      # threshold at 0.5 → 0 or 1

            all_probs.append(probs.numpy())
            all_preds.append(preds.numpy())
            all_labels.append(y_batch.numpy())

    all_probs  = np.concatenate(all_probs).flatten()
    all_preds  = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC-ROC  : {auc:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Disease"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix — Heart Disease Prediction")
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    evaluate()