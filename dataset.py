#-------------------------- Neural Network Project ----------------------------#

###--------------Data Handling------------------------

import torch 
import pandas as pd 
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class HeartDataset(Dataset): 
    def __init__(self, X, y):
        # store as float32 tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # (N,) → (N,1)

    def __len__(self):
        return len(self.X)      # telling dataloader how many samples there are 
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]   # returns one (patient, label) pair
    

def get_dataloaders(path="data/heart.csv", batch_size=32, test_size=0.2):
    # load csv, preprocess, splits
    
    ##load 
    df = pd.read_csv(path)
    X = df.drop("target", axis=1).values    #numpy array (297, 13) 
    y = df["target"].values                 # numpy array (297, )

    # split (before scaling)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Scale (after split, fit on train only, transform both)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)    #for getting mean and standard deviation from training set
    X_val = scaler.transform(X_val)     # applies the same scaler to val set 
    train_loader = DataLoader(HeartDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(HeartDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

    


    