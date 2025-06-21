

import time
from tqdm import tqdm
import torch
from classifier import AgeClassifier
from losses import combined_loss
from data_loader import get_data_loaders
from config import Config
from torchvision.models import ResNet50_Weights

def train_model():
    cfg = Config()

    
    train_loader, val_loader = get_data_loaders(cfg.DATA_DIR, cfg.BATCH_SIZE, cfg.VAL_SPLIT)

   
    model = AgeClassifier(weights=ResNet50_Weights.DEFAULT if cfg.PRETRAINED else None).to(cfg.DEVICE)

    
    criterion = lambda pred, target: combined_loss(pred, target, cfg.NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(cfg.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{cfg.NUM_EPOCHS}")
        start_time = time.time()

        
        model.train()
        epoch_train_loss = 0.0
        for images, ages in tqdm(train_loader, desc="Training"):
            images, ages = images.to(cfg.DEVICE), ages.to(cfg.DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, ages)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * images.size(0)

        train_loss = epoch_train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for images, ages in tqdm(val_loader, desc="Validation"):
                images, ages = images.to(cfg.DEVICE), ages.to(cfg.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, ages)
                epoch_val_loss += loss.item() * images.size(0)

        val_loss = epoch_val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {time.time() - start_time:.2f}s")

        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), cfg.MODEL_SAVE_PATH)
            print(f"New best model saved with Val Loss: {val_loss:.4f}")

    print("\nTraining complete!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    return model, train_losses, val_losses

if __name__ == "__main__":
    train_model()
