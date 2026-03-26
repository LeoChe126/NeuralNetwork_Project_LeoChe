# NeuralNetwork_Project_LeoChe
## Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 86.7%  |
| F1 Score  | 0.852  |
| AUC-ROC   | 0.963  |

Validation set: 60 patients (20% stratified split)
Confusion matrix: 29 TN · 23 TP · 3 FP · 5 FN

Early stopping fired at epoch 51.
Best model weights saved from the epoch with lowest validation loss.

## Clinical threshold consideration
The default decision threshold of 0.5 produced 5 false negatives — patients
with heart disease predicted as healthy. In a real screening tool, a lower
threshold (e.g. 0.3) would reduce false negatives at the cost of more false
positives. With an AUC of 0.963, the model has strong discriminative power —
threshold tuning is a post-training decision that depends on the clinical
cost of each error type.
```

This one paragraph shows an interviewer you understand the difference between model performance and deployment decisions — that's a senior-level insight.

---

### Your project is complete. Here's what you built:
```
✓  3-layer feedforward neural network in PyTorch
✓  Real medical dataset — preprocessed, split, scaled correctly
✓  Training loop with dropout, early stopping, loss curves
✓  86.7% accuracy · 0.852 F1 · 0.963 AUC on held-out validation set
✓  Confusion matrix with clinical interpretation
✓  Modular codebase across 7 files
✓  Portfolio notebook + README
