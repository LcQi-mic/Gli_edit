import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageImageContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):

        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss
    
    
class GliomaMRIContrastiveLoss(torch.nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()

        self.batch_size = batch_size
        self.register_buffer("temp", torch.tensor(temperature).to(torch.device("cuda")))
        self.register_buffer("neg_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to('cuda')).float())

    def forward(self, x_i, x_j):        
        z = torch.cat([x_i, x_j], dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim_ij = torch.diag(sim, self.batch_size)
        sim_ji = torch.diag(sim, -self.batch_size)
        pos = torch.cat([sim_ij, sim_ji], dim=0)
        nom = torch.exp(pos / self.temp)
        denom = self.neg_mask * torch.exp(sim / self.temp)
        return torch.sum(-torch.log(nom / torch.sum(denom, dim=1))) / (2 * self.batch_size)
    
    
class Loss(torch.nn.Module):
    def __init__(self, batch_size, temperature=0.05, alpha_1=1., alpha_2=1.):
        super().__init__()
        
        self.L2ILoss = LanguageImageContrastiveLoss()
        self.G2ILoss = GliomaMRIContrastiveLoss(batch_size, temperature=temperature)
        
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
    
    def forward(self, image_features, gli_features, text_features, logit_scale):
        L2Iloss = self.L2ILoss(image_features, text_features, logit_scale)
        G2Iloss = self.G2ILoss(gli_features, image_features)
        
        total_loss = self.alpha_1 * L2Iloss + self.alpha_2 * G2Iloss
        
        return total_loss, (L2Iloss, G2Iloss)
        