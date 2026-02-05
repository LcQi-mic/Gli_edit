import torch
from torch.nn import functional as F

class Loss(torch.nn.Module):
    def __init__(self, alpha_1=1., alpha_2=1.):
        super().__init__()
        self.l1_loss = torch.nn.SmoothL1Loss()
        self.cossim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
    
    def __call__(self, f_0, f_1, f_2, t_0, t_1, t_2):
        f_0 = F.normalize(f_0, dim=1)
        f_1 = F.normalize(f_1, dim=1)
        f_2 = F.normalize(f_2, dim=1)
        t_0 = F.normalize(t_0, dim=1)
        t_1 = F.normalize(t_1, dim=1)
        t_2 = F.normalize(t_2, dim=1)
        
        distill_loss = (self.l1_loss(f_0, t_0) + self.l1_loss(f_1, t_1) + self.l1_loss(f_2, t_2)) / 3
        
        dist_regularization = torch.abs(self.cossim(f_0, f_2) - self.cossim(f_1, f_0) - self.cossim(f_2, f_1)).mean()
        
        total_loss = self.alpha_1 * distill_loss + self.alpha_2 * dist_regularization

        return total_loss, (distill_loss, dist_regularization)
    
    

