#-------------------------- Neural Network Project ----------------------------#

###-----------------training model---------------------

from utils import EarlyStopping
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model import HeartNet
from dataset import get_dataloaders


EPOCHS        = 100
BATCH_SIZE    = 32
LEARNING_RATE = 0.001

# Real data
train_loader, val_loader = get_dataloaders("data/heart.csv", BATCH_SIZE)

model     = HeartNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []
val_losses   = []
early_stopping = EarlyStopping(patience=10)


# Training loop
for epoch in range(EPOCHS):

    # ── Training phase ───────────────────────────────────────────
    model.train()
    train_loss = 0.0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss   = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # ── Validation phase ─────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            output    = model(X_batch)
            val_loss += criterion(output, y_batch).item()

    # ── Calculate averages FIRST ─────────────────────────────────
    avg_train = train_loss / len(train_loader)
    avg_val   = val_loss   / len(val_loader)

    # ── Collect for plotting ──────────────────────────────────────
    train_losses.append(avg_train)
    val_losses.append(avg_val)

    # ── Logging ───────────────────────────────────────────────────
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {avg_train:.4f} | "
              f"Val Loss: {avg_val:.4f}")

    # ── Early stopping (uses avg_val, so must come after) ─────────
    early_stopping(avg_val, model)
    if early_stopping.stop:
        print(f"Early stopping at epoch {epoch + 1}")
        break


# ── Loss curve ────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label="Train loss")
plt.plot(val_losses,   label="Val loss",  linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.savefig("loss_curve.png", dpi=150, bbox_inches="tight")
plt.show()
