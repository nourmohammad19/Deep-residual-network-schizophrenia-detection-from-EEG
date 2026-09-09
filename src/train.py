# train.py
import yaml # type: ignore
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from model import RezNet
from data import *

from engine import train_one_epoch, evaluate


def main():
    # Load Config
    with open("./configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    
    full_dataset = OptimizedDataset("optimized_tensors")

    
    total_size = len(full_dataset)
    train_size = int(config['data']['train_test_ratio'] * total_size)
    test_size = total_size - train_size

    # Randomly split the dataset

    train_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    #  Create separate DataLoaders for each split
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    batch_size = config['data']['batch_size']

    # Setup Model, Loss, Optimizer
    device = torch.device("mps")
    model = RezNet(3,2,1)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    best_loss = float('inf')
    
    for epoch in range(config['training']['epochs']):
        print(f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, batch_size)
        # val_loss, val_acc = evaluate(model, val_loader, criterion, device) 
        
        print(f"Train Loss: {train_loss:.4f}")
        
        # Save Best Model
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), f"{config['training']['save_dir']}/best_model.pth")

if __name__ == "__main__":
    main()