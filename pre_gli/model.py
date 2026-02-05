import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class Projector(nn.Module):
    def __init__(self, mid_width, width) -> None:
        super().__init__()
        
        self.linear = nn.Linear(512, mid_width)
        
        self.projector = nn.Sequential(
            nn.Linear(mid_width * 3, mid_width),
            nn.Linear(mid_width, width)
        )        
        
    def forward(self, f_wt, f_tc, f_et):
        f_wt = self.linear(f_wt.to(torch.int32))
        f_tc = self.linear(f_tc.to(torch.int32))
        f_et = self.linear(f_et.to(torch.int32))
        
        f = torch.concat([f_wt, f_tc, f_et], dim=1)
        
        f = self.projector(f)
        return f


class GliCLIP(nn.Module):
    def __init__(
        self,
        init_logit_scale: float = np.log(1 / 0.07),
        img_feature_dim: int=768,
        style_dim: int=256,
        text_dim: int=512,
        gli_encoder=None
    ):
        super().__init__()

        self.gli_encoder = gli_encoder

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.mri_projector = nn.Linear(img_feature_dim, style_dim)
        self.gli_projector = nn.Linear(img_feature_dim, style_dim)
        self.text_projector = nn.Linear(text_dim, style_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)

    def encode_image(self, f_image, normalize: bool = True):
        x = self.pool(f_image)
        x = x.flatten(start_dim=2, end_dim=3).squeeze(-1)
        x = self.mri_projector(x)
        return F.normalize(x, dim=-1) if normalize else x

    def encode_gli(self, gli, normalize: bool = True):
        f_gli = self.gli_encoder(gli)
        x = self.pool(f_gli)
        x = x.flatten(start_dim=2, end_dim=3).squeeze(-1)
        x = self.gli_projector(x)
        return F.normalize(x, dim=-1) if normalize else x

    def encode_text(self, f_text, normalize: bool = True):
        f = self.text_projector(f_text)
        return F.normalize(f, dim=-1) if normalize else f

    # 用于测试和可视化
    def get_logits(self, gli, text):
        gli_features = self.encode_gli(gli)
        text_features = self.encode_text(text)
        
        gli_features = self.logit_scale.exp() * gli_features @ text_features.T

        text_logits = gli_features.T
        return gli_features, text_logits

    def forward(self, f_image, gli, f_text):
        image_features = self.encode_image(f_image, normalize=True) 
        gli_features = self.encode_gli(gli, normalize=True) 
        text_features = self.encode_text(f_text, normalize=True) 

        return image_features, gli_features, text_features, self.logit_scale.exp()