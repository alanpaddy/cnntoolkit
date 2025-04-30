import torch
from models.resnet_custom import get_resnet18
from utils.dataloader_utils import get_dataloaders
from utils.metrics import accuracy
import torch.nn as nn
import torch.optim as optim

def train():
    train_loader, _ = get_dataloaders()
    model = get_resnet18(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(2):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} complete.")

if __name__ == "__main__":
    train()
