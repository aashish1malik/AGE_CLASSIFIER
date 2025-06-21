
import os
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class AgeFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir).resolve()
        self.transform = transform
        self.samples = []

        for age_dir in os.listdir(self.root_dir):
            age_path = self.root_dir / age_dir
            if age_path.is_dir() and age_dir.isdigit():
                age = int(age_dir)
                image_files = list(age_path.glob("*.jpg")) + list(age_path.glob("*.jpeg")) + list(age_path.glob("*.png"))
                self.samples.extend([(img_path, age) for img_path in image_files])

        if not self.samples:
            raise RuntimeError(f"No valid images found in {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, age = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(age, dtype=torch.float)  

def get_data_loaders(data_dir, batch_size=32, val_split=0.2):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    full_dataset = AgeFolderDataset(root_dir=data_dir, transform=transform)
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0 if os.name == 'nt' else 4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0 if os.name == 'nt' else 4)

    return train_loader, val_loader
