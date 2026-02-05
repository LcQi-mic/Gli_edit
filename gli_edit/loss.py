import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss


class WeightedDiceLoss(nn.Module):
    def __init__(self, weight_wt=0.1, weight_tc=0.5, weight_et=0.4) -> None:
        super().__init__()
        self.dice_loss = DiceCELoss(include_background=True, batch=True, sigmoid=True, squared_pred=True)
        self.weight_wt = weight_wt
        self.weight_tc = weight_tc
        self.weight_et = weight_et
        
    def forward(self, x, y):
        x_wt = x[:, 0].unsqueeze(1)
        x_tc = x[:, 1].unsqueeze(1)
        x_et = x[:, 2].unsqueeze(1)
        
        y_wt = y[:, 0].unsqueeze(1)
        y_tc = y[:, 1].unsqueeze(1)
        y_et = y[:, 2].unsqueeze(1)
        
        return self.weight_wt * self.dice_loss(x_wt, y_wt) + self.weight_tc * self.dice_loss(x_tc, y_tc) + self.weight_et * self.dice_loss(x_et, y_et)
        

class ConvBlock(nn.Module):
    def __init__(self, channel):
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channel, 1, 1, 1, 0, bias=False)
        self.conv2 = nn.Conv2d(1, 1, 1, 1, 0, bias=False)
        
        for param in self.conv1.parameters():
            param.requires_grad = False
            
        for param in self.conv2.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        return x
            

class Perceptual_loss(nn.Module):
    def __init__(self, network=None, channel=3):
        super(Perceptual_loss, self).__init__()

        # pretrained network
        if network is not None:
            self.net = network
        else:
            self.net = ConvBlock(channel)
            
        self.loss = nn.L1Loss()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        feat_x, feat_y = self.net(x), self.net(y)
        feat_x = F.normalize(feat_x, dim=1)
        feat_y = F.normalize(feat_y, dim=1)

        return self.loss(feat_x, feat_y)
    
    
if __name__ =="__main__":
    inps = torch.randn((64, 1, 256, 256))

    loss = Perceptual_loss(channel=1)
    
    a = loss(inps, inps)
    
    print(a)