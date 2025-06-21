import torch
from classifier import AgeClassifier
from data_loader import get_data_loaders
from config import Config

def evaluate_model(model_path):
    cfg = Config()
    
    
    model = AgeClassifier(
        weights=False, 
        num_classes=cfg.NUM_CLASSES
    ).to(cfg.DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    
    _, test_loader = get_data_loaders(
        cfg.DATA_DIR,
        cfg.BATCH_SIZE,
        val_split=0.2
    )
    
    
    ...

if __name__ == "__main__":
    evaluate_model("best_model.pth")