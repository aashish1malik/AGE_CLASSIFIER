




import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class AgeClassifier(nn.Module):
    def __init__(self, weights=ResNet50_Weights.DEFAULT, num_classes=100):
        super(AgeClassifier, self).__init__()
        self.backbone = resnet50(weights=weights)
        
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        
        self.fc1 = nn.Linear(in_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_classes)  

    def forward(self, x):
        x = self.backbone(x)       
        x = self.fc1(x)            
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)            
        return x
