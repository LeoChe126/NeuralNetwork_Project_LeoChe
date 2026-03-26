#-------------------------- Neural Network Project ----------------------------#

###--------------Necessary class for early stopping ------------------------
import torch

class EarlyStopping:

    def __init__(self, patience=10, min_delta=0.001):
        self.patience   = patience    # how many epochs to wait
        self.min_delta  = min_delta   # minimum improvement to count
        self.best_loss  = float("inf")
        self.counter    = 0
        self.stop       = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
                print(f"Early stopping triggered at patience={self.patience}")