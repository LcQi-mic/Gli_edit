from collections import OrderedDict
from typing import Callable, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

import clip


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)
    

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    

class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
            batch_first: bool = True,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=batch_first)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: None,
            v_x: None,
            attn_mask: None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(
            q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask
        )[0]

    def forward(
            self,
            q_x: torch.Tensor,
            attn_mask,
    ):
        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=None, v_x=None, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            batch_first: bool = True,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.batch_first = batch_first
        self.grad_checkpointing = False
        
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                batch_first=batch_first,
            )
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()    # NLD -> LND
        for r in self.resblocks:
            x = r(x, attn_mask=attn_mask)
        if not self.batch_first:
            x = x.transpose(0, 1)    # LND -> NLD
        return x

class TextEncoder(nn.Module):
    def __init__(self, 
                width,
                layers,
                heads,
                mlp_ratio: float = 4.0,
                ls_init_value: float = None,
                act_layer: Callable = nn.GELU,
                norm_layer: Callable = LayerNorm,
                batch_first: bool = True,
                style_dim: int = 512
                ) -> None:
        super().__init__()
        # vocab_size = 49408
        self.token_embedding = nn.Embedding(49408, width)
        self.positional_embedding = nn.Parameter(torch.empty(77, width))
        self.ln_final = LayerNorm(width)
        
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value,
            act_layer,
            norm_layer,
            batch_first,
        )
        
        self.text_projection = nn.Parameter(torch.empty(width, style_dim))
        
    @property       
    def dtype(self):
        return self.token_embedding.weight.dtype

    def initialize_parameters(self):
        
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
    
    # hacked from clip
    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
 
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x
    
class DistillationModel(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        style_dim: int = 128,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        batch_first: bool = True,
        teacher_model: str = "ViT-B/16"
    ):
        super().__init__()
        
        # student model
        self.s = TextEncoder(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value,
            act_layer,
            norm_layer,
            batch_first,
            style_dim
        )

        # teacher model
        self.t, _ = clip.load(teacher_model) 

    def get_t_feature(self, token):
        f = self.t.encode_text(token.to(torch.int32))
        return f
    
    def forward(self, token):
        f =  self.s(token.to(torch.int32))
        return f
    
    
if __name__ == "__main__":
    transformer = TextEncoder(width=512, layers=4, heads=8)
    transformer.initialize_parameters()
    model, preprocess = clip.load("ViT-B/16")
    model.cpu().eval()
    
    text = ["Hello world", "Hello world", "Hello world"]
    text_tokens = clip.tokenize(text)
    
    print(text_tokens.size())

    clip_features = model.encode_text(text_tokens)
    print(clip_features.size())
    
    transformer_features = transformer(text_tokens)
    print(clip_features.size())
    
