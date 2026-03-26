#-------------------------- Neural Network Project ----------------------------#

###--------------Tunes Hyperparameters------------------------

# Simple grid search — run training with different configs
configs = [
    {"lr": 0.001, "dropout": 0.2, "hidden": 16},
    {"lr": 0.001, "dropout": 0.3, "hidden": 32},
    {"lr": 0.0005,"dropout": 0.3, "hidden": 16},
    {"lr": 0.01,  "dropout": 0.2, "hidden": 16},
]

results = []

for cfg in configs:
    model     = HeartNet(dropout=cfg["dropout"])
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    # ... run full training loop ...
    # ... collect final val loss + accuracy ...
    results.append({"config": cfg, "val_loss": avg_val, "accuracy": acc})

print(f"\n{'LR':<8} {'Dropout':<10} {'Val Loss':<12} {'Accuracy'}")
print("-" * 44)
for r in results:
    c = r["config"]
    print(f"{c['lr']:<8} {c['dropout']:<10} {r['val_loss']:<12.4f} {r['accuracy']:.4f}")
