from math import log2

from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from kornia.filters import filter2d
import math
    
def exists(val):
    return val is not None

def noise(n, latent_dim, device):
    return torch.randn(n, latent_dim).to(device)

def noise_list(n, layers, latent_dim, device):
    return [(noise(n, latent_dim, device), layers)]

def mixed_list(n, layers, latent_dim, device):
    tt = int(torch.rand(()).numpy() * layers)
    return noise_list(n, tt, latent_dim, device) + noise_list(n, layers - tt, latent_dim, device)

def text_list(n, layers, latent_dim, device, f_text):
    tt = 2
    return [(f_text, tt)] + noise_list(n, layers - tt, latent_dim, device)

def gli_list(n, layers, latent_dim, device, f_text, f_gli):
    tt = 4
    return [(f_text, 2)] + [(f_gli, 2)] + noise_list(n, layers - tt, latent_dim, device)

def mri_list(n, layers, latent_dim, device, f_text, f_gli, f_mri):
    tt = 6
    return [(f_text, 2)] + [(f_gli, 2)] + [(f_mri, 2)] + noise_list(n, layers - tt, latent_dim, device)

def latent_to_w(style_vectorizer, latent_descr):
    return [(style_vectorizer(z), num_layers) for z, num_layers in latent_descr]

def styles_def_to_tensor(styles_def):
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)

def image_noise(n, im_size, device):
    return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.).to(device)

def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)

class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)
    
    
class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        if bias:
            self.bias = nn.Parameter(torch.zeros(channel))

        else:
            self.bias = None

        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, inputs):
        return fused_leaky_relu(inputs, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(inputs, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    if bias is not None:
        rest_dim = [1] * (inputs.ndim - bias.ndim - 1)
        return (
            F.leaky_relu(
                inputs + bias.view(1, bias.shape[0], *rest_dim), negative_slope=negative_slope
            )
            * scale
        )

    else:
        return F.leaky_relu(inputs, negative_slope=negative_slope) * scale


class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)
    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f [None, :, None]
        return filter2d(x, f, normalized=True)
    
    
class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out

# attention
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x

class ChanNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = ChanNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding = 0, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, dim_head = 64, heads = 8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.nonlin = nn.GELU()
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, 3, padding = 1, bias = False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, fmap):
        h, x, y = self.heads, *fmap.shape[-2:]
        q, k, v = (self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = h), (q, k, v))

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)

        out = self.nonlin(out)
        return self.to_out(out)
    
attn_and_ff = lambda chan: nn.Sequential(*[
    Residual(PreNorm(chan, LinearAttention(chan))),
    Residual(PreNorm(chan, nn.Sequential(nn.Conv2d(chan, chan * 2, 1), leaky_relu(), nn.Conv2d(chan * 2, chan, 1))))
])

# stylegan2 classes

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul = 1, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)

class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth, lr_mul = 0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)

class MRIBlock(nn.Module):
    def __init__(self, latent_dim, input_channel, out_channel, upsample):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)

        self.conv = Conv2DMod(input_channel, out_channel, 1, demod=False)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False),
            Blur()
        ) if upsample else None

    def forward(self, x, prev_mri, istyle):
        style = self.to_style(istyle)
        x = self.conv(x, style)

        if exists(prev_mri):
            x = x + prev_mri

        if exists(self.upsample):
            x = self.upsample(x)

        return x

class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps = 1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x

class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, cls = 1, upsample = True, upsample_mri = True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(input_channels, filters, 3)
        
        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = leaky_relu()
        self.to_mri = MRIBlock(latent_dim, filters, cls, upsample_mri)

    def forward(self, x, prev_mri, istyle, inoise):
        if exists(self.upsample):
            x = self.upsample(x)

        inoise = inoise[:, :x.shape[2], :x.shape[3], :]
        noise1 = self.to_noise1(inoise).permute((0, 3, 2, 1))
        noise2 = self.to_noise2(inoise).permute((0, 3, 2, 1))

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)

        mri = self.to_mri(x, prev_mri, istyle)
        return x, mri

class Generator(nn.Module):
    def __init__(
            self,
            image_size,
            style_dim,
            network_capacity=16,
            fmap_max = 512,
            cls=1,
            style_depth=8,
            task="gli_gen",
            attn_layers = [],
            image_feature_dim=384
    ):
        super().__init__()
        self.style_dim = style_dim
        self.num_layers = int(log2(image_size) - 1)
        
        filters = [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][::-1]
        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]

        in_out_pairs = zip(filters[:-1], filters[1:])

        self.to_initial_block = nn.ConvTranspose2d(style_dim, init_channels, 4, 1, 0, bias=False)
        
        self.conv1 = nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.blocks = nn.ModuleList([])
        self.attns = nn.ModuleList([])
        
        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            
            num_layer = self.num_layers - ind
            attn_fn = attn_and_ff(in_chan) if num_layer in attn_layers else None

            self.attns.append(attn_fn)
            
            block = GeneratorBlock(
                style_dim,
                in_chan,
                out_chan,
                cls=cls,
                upsample = not_first,
                upsample_mri = not_last
            )
            self.blocks.append(block)
            
        self.S = StyleVectorizer(style_dim, depth=style_depth, lr_mul=0.1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gli_projector = nn.Linear(image_feature_dim, style_dim)
        self.img_projector = nn.Linear(image_feature_dim, style_dim)
        self.mri_initial = nn.Conv2d(image_feature_dim,
                                     style_dim,
                                     1,
                                     1)
        
        self.task = task
        
    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        for block in self.blocks:
            nn.init.zeros_(block.to_noise1.weight)
            nn.init.zeros_(block.to_noise2.weight)
            nn.init.zeros_(block.to_noise1.bias)
            nn.init.zeros_(block.to_noise2.bias)

    def make_image_noise(self, x):
        batch_size, _, image_size, _ = x.size()
        return image_noise(batch_size, image_size, device=x.device)

    def get_w(self, x, f_text=None, f_gli=None, f_mri=None):   
        batch_size, _, _, _ = x.size()   

        f_gli = self.pool(f_gli)
        f_gli = f_gli.flatten(start_dim=2, end_dim=3).squeeze(-1)
        f_gli = self.gli_projector(f_gli)
        
        f_mri = self.pool(f_mri)
        f_mri = f_mri.flatten(start_dim=2, end_dim=3).squeeze(-1)
        f_mri = self.img_projector(f_mri)
        
        if self.task == "gli_gen":
            style = text_list(batch_size, self.num_layers, self.style_dim, device=x.device, f_text=f_text)
        elif self.task == "gli_edit":
            style = gli_list(batch_size, self.num_layers, self.style_dim, device=x.device, f_text=f_text, f_gli=f_gli)
        elif self.task == "mri_gen":
            style = mri_list(batch_size, self.num_layers, self.style_dim, device=x.device, f_text=f_text, f_gli=f_gli, f_mri=f_mri)
        else:
            style = mixed_list(batch_size, self.num_layers, self.style_dim, device=x.device)
        
        w = latent_to_w(self.S, style)
        w = styles_def_to_tensor(w)

        return w
    
    def forward(self, styles, input_noise, f_mri=None):
        """_summary_

        Args:
            styles (_type_): [b, num_layer, style_dim]
            input_noise (_type_): _description_

        Returns:
            _type_: _description_
        """
        if f_mri is None:
            avg_style = styles.mean(dim=1)[:, :, None, None]
            x = self.to_initial_block(avg_style)
        else:
            x = self.mri_initial(f_mri)
        
        mri = None
        
        styles = styles.transpose(0, 1)
        x = self.conv1(x)

        for style, block, attn in zip(styles, self.blocks, self.attns):
            if exists(attn):
                x = attn(x)
            x, mri = block(x, mri, style, input_noise)

        return mri


class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride = (2 if downsample else 1))

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding=1),
            leaky_relu()
        )

        self.downsample = nn.Sequential(
            Blur(),
            nn.Conv2d(filters, filters, 3, padding = 1, stride = 2)
        ) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        if exists(self.downsample):
            x = self.downsample(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x
    
    
class Discriminator(nn.Module):
    def __init__(self, 
                 image_size, 
                 network_capacity = 16, 
                 attn_layers = [], 
                 fmap_max = 512,
                 cls=1):
        super().__init__()
        num_layers = int(log2(image_size) - 1)
        num_init_filters = cls

        blocks = []
        filters = [num_init_filters] + [(network_capacity * 4) * (2 ** i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []
        attn_blocks = []
        quantize_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample = is_not_last)
            blocks.append(block)

            attn_fn = attn_and_ff(out_chan) if num_layer in attn_layers else None

            attn_blocks.append(attn_fn)

        self.blocks = nn.ModuleList(blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)
        self.quantize_blocks = nn.ModuleList(quantize_blocks)

        chan_last = filters[-1]
        latent_dim = 2 * 2 * chan_last

        self.final_conv = nn.Conv2d(chan_last, chan_last, 3, padding=1)
        self.flatten = Flatten()
        self.to_logit = nn.Linear(latent_dim, 1)

    def forward(self, x):
        b, *_ = x.shape

        for (block, attn_block) in zip(self.blocks, self.attn_blocks):
            x = block(x)

            if exists(attn_block):
                x = attn_block(x)

        x = self.final_conv(x)
        x = self.flatten(x)
        x = self.to_logit(x)
        return x.squeeze()
    
    
if __name__ == "__main__":
    gen = Generator(256, 512, attn_layers=[1,2,3,4,5,6,7,8])
    gen._init_weights()
    x = torch.randn((4, 3, 256, 256))
    
    f = torch.randn((4, 512))
    f_1 = torch.randn((4, 512))

    w = gen.get_w(x, f, f, f)

    noise = gen.make_image_noise(x)

    out = gen(w, noise)
    
    print(out.size())

    
    dis = Discriminator(256, attn_layers=[1,2,3,4,5,6,7,8])
    
    out = dis(out)
    
    print(out)
    


