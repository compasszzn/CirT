from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, trunc_normal_
from torch.jit import Final

import torch.nn.functional as F
from typing import Optional
from timm.layers import DropPath, use_fused_attn, Mlp

class CLinear(nn.Module):
  def __init__(self, in_channels, out_channels, **kwargs):
    super(CLinear, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels

    self.re_linear = nn.Linear(self.in_channels, self.out_channels, **kwargs)
    self.im_linear = nn.Linear(self.in_channels, self.out_channels, **kwargs)

    nn.init.xavier_uniform_(self.re_linear.weight)
    nn.init.xavier_uniform_(self.im_linear.weight)

  def forward(self, x):  
    x = torch.view_as_real(x)
    x_re = x[..., 0]
    x_im = x[..., 1]

    out_re = self.re_linear(x_re) - self.im_linear(x_im)
    out_im = self.re_linear(x_im) + self.im_linear(x_re)

    out = torch.stack([out_re, out_im], -1) 
    out = torch.view_as_complex(out)

    return out

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.dim = dim
        self.attn_bias = nn.Parameter(torch.zeros(121, 121, 2), requires_grad=True)

        # self.qkv = CLinear(dim, dim * 3, bias=qkv_bias)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        # self.q = CLinear(dim // 2 + 1, dim, bias=qkv_bias)
        # self.k = CLinear(dim // 2 + 1, dim, bias=qkv_bias)
        # self.v = CLinear(dim // 2 + 1, dim, bias=qkv_bias)


        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = CLinear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # v = self.v(x).reshape(B, self.num_heads, N, self.head_dim)

        # x = torch.fft.rfft(x, norm="forward")

        q = self.q(x).reshape(B, self.num_heads, N, self.head_dim)
        k = self.k(x).reshape(B, self.num_heads, N, self.head_dim)
        v = self.v(x).reshape(B, self.num_heads, N, self.head_dim)

        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv.unbind(0)

        # q = torch.fft.fft(q)
        # k = torch.fft.fft(k)
        # v = torch.fft.fft(v)

        # qkv = self.qkv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # q, k = qkv.unbind(0)
        
        # q, k = self.q_norm(q), self.k_norm(k)

        # if self.fused_attn:
        #     x = F.scaled_dot_product_attention(
        #         q, k, v,
        #         dropout_p=self.attn_drop.p if self.training else 0.,
        #     )
        # else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1) 

        # attn = torch.view_as_real(attn)
        # attn = attn + self.attn_bias
        # attn = torch.norm(attn, p=2, dim=-1)
        attn = attn.softmax(dim=-1)
        
        # attn = attn.softmax(dim=-2)
        # attn = torch.view_as_complex(attn)

        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        # x = torch.fft.ifft(x, self.dim).to(torch.float)
        # x = torch.fft.fft(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        # x = torch.fft.irfft(x, self.dim, norm="forward")
        
        # x = torch.fft.ifft(x, self.dim)
        # x = torch.view_as_real(x)
        # x = torch.norm(x, p=2, dim=-1)
        return x


class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma



class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class PatchEmbed(nn.Module):
    def __init__(
            self,
            img_size=[121, 240],
            in_chans=63,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_patches = img_size[1]
        self.flatten = flatten

        # self.proj = CLinear(in_chans, embed_dim)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=[242, 1], stride=1, bias=bias)
        # self.proj = CConv2d(in_chans, embed_dim, kernel_size=[1, 240], stride=1, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class SphFormer(nn.Module):
    def __init__(
        self,
        img_size=[242, 120],
        input_size=63,
        patch_size=124,
        embed_dim=256,
        depth=8,
        decoder_depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1
    ):
        super().__init__()

        # TODO: remove time_history parameter
        self.img_size = img_size
        self.patch_size = img_size[0]
        self.input_size = input_size
        self.token_embeds = PatchEmbed(img_size, input_size, embed_dim)
        self.num_patches = self.token_embeds.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=True)
        self.clinear = CLinear(img_size[0], img_size[0])
        # self.pos_embed = PosEmbed(embed_dim=embed_dim)    

        # --------------------------------------------------------------------------

        # ViT backbone
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    # drop=drop_rate,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # --------------------------------------------------------------------------

        # prediction head
        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(embed_dim, self.input_size * 2 * self.img_size[0]))
        self.head = nn.Sequential(*self.head)

        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # token embedding layer
        w = self.token_embeds.proj.weight.data
        trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        """
        x: (B, L, V * patch_size)
        return imgs: (B, V, H, W)
        """
        b = x.shape[0]
        # p = self.patch_size
        p = self.img_size[0]
        c = self.input_size
        h = self.img_size[0] // p 
        w = self.img_size[1] // 1
        assert h * w == x.shape[1]

        x = x.reshape(shape=(b, 1, w, p, c * 2))
        x = torch.einsum("nhwpc->nhcwp", x)
        x = x.reshape(shape=(b, 1, c * 2, w, p))
        # x = torch.view_as_complex(x)
        # x = self.invsht(x)
        # x = torch.fft.irfft(x, 240)
        imgs = x.reshape(shape=(b, 2, c, w, p))
        return imgs

    def forward_encoder(self, x: torch.Tensor):
        # x: `[B, V, H, W]` shape.
        B, V, H, W = x.shape
        # x = x.reshape(B * V, H, W)
        # x = torch.fft.rfft(x)
        # x = self.clinear(x)
        # x = torch.fft.irfft(x, 240)
        # print(x.shape)
        
        # print(torch.nanmean(torch.abs(x - torch.fft.irfft(torch.fft.rfft(x)))))
        # ddd

        # x = self.sht(x).transpose(1, 2).reshape(B, V, H, H).to(torch.float)
        # x = torch.view_as_real(self.sht(x).reshape(B, V, H, H))
        # print(x.shape)
        # ddd

        # tokenize each variable separately
        x = self.token_embeds(x)

        # pos_embed = self.pos_embed()
        # add pos embedding
        # x = x + self.pos_embed

        x = self.pos_drop(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward(self, x):
        
        x = torch.cat([x[:,:,:,:120],torch.flip(x[:,:,:,120:], dims=[2])],dim=2)
        out_transformers = self.forward_encoder(x) + self.pos_embed # B, L, D
        preds = self.head(out_transformers)  # B, L, V*p*p
        preds = self.unpatchify(preds)
        preds = torch.cat([preds[:,:,:,:,:121],torch.flip(preds[:,:,:,:,121:], dims=[4])],dim=3).transpose(3, 4)
        return preds

