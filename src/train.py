import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import sys
from torch.utils.data import Dataset, DataLoader, random_split

# Ensure the src directory is accessible for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import FetalHCModel

# 1. Custom Dataset Class
class FetalDataset(Dataset):
    def __init__(self, csv_file, processed_dir):
        # Expects a CSV with columns: filename, cx, cy, a, b, angle
        self.data_frame = pd.read_csv(csv_file)
        self.processed_dir = processed_dir

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get filename and swap .png to .npy since we are loading processed arrays
        raw_filename = self.data_frame.iloc[idx]['filename']
        npy_filename = raw_filename.replace('.png', '.npy')
        npy_path = os.path.join(self.processed_dir, npy_filename)

        # Load the preprocessed 512x512 array
        image_array = np.load(npy_path)

        # 1. Extract the 5 regression targets BEFORE tensor conversion
        cx, cy, a, b, angle = self.data_frame.iloc[idx][['cx', 'cy', 'a', 'b', 'angle']].values.astype(np.float32)

        # 2. DATA AUGMENTATION
        # 50% chance to flip the image horizontally
        if np.random.rand() > 0.5:
            # np.fliplr returns a view with negative strides, so .copy() is strictly required 
            # for PyTorch to accept the array later
            image_array = np.fliplr(image_array).copy() 
            cx = 512.0 - cx                             # Invert the X coordinate
            angle = 180.0 - angle                       # Invert the rotation angle

        # 3. PyTorch expects channel-first format: (Channels, Height, Width)
        image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)

        # 4. Re-pack the targets and normalize to [0, 1] range
        # cx, cy, a, b are bounded by target image shape (512.0)
        # angle is bounded by 180.0
        targets = np.array([cx / 512.0, cy / 512.0, a / 512.0, b / 512.0, angle / 180.0], dtype=np.float32)
        target_tensor = torch.tensor(targets, dtype=torch.float32)

        return image_tensor, target_tensor

# 2. Main Training Loop
def train_model(csv_path, processed_dir, batch_size=16, num_epochs=120, learning_rate=5e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Booting Advanced Training Engine on: {device}")

    dataset = FetalDataset(csv_file=csv_path, processed_dir=processed_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = FetalHCModel(pretrained=True).to(device)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # LEARNING RATE SCHEDULER 
    # Drops LR by 50% if Validation Loss stagnates for 5 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, targets in train_dataloader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_dataloader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_dataloader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
        val_epoch_loss = val_loss / len(val_dataloader)
        
        # STEP THE SCHEDULER 
        scheduler.step(val_epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f} | LR: {current_lr:.6f}")

    weights_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fetal_vision_weights.pth')
    torch.save(model.state_dict(), weights_path)
    print(f"Training Complete. Weights successfully saved to {weights_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CSV_PATH = os.path.join(base_dir, "data", "raw", "train_split.csv")
    PROCESSED_DIR = os.path.join(base_dir, "data", "processed")
    train_model(CSV_PATH, PROCESSED_DIR)