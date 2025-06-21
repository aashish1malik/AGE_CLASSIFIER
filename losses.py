import torch
import torch.nn as nn
import torch.nn.functional as F

class OrdinalRegressionLoss(nn.Module):
    def __init__(self, num_classes):
        super(OrdinalRegressionLoss, self).__init__()
        self.num_classes = num_classes
        
    def forward(self, predictions, targets):
       
        ord_targets = torch.zeros_like(predictions)
        for i in range(self.num_classes - 1):
            ord_targets[:, i] = (targets > i).float()
        
        
        loss = F.binary_cross_entropy_with_logits(predictions, ord_targets)
        return loss

def combined_loss(predictions, targets, num_classes, alpha=0.5):
    
    mae_loss = F.l1_loss(torch.sum(torch.sigmoid(predictions), dim=1), targets.float())
    ord_loss = OrdinalRegressionLoss(num_classes)(predictions, targets)
    return alpha * mae_loss + (1 - alpha) * ord_loss