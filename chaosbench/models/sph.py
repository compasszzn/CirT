from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed, trunc_normal_
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
        self.num_patches = img_size[0]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=[1, 240], stride=1, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class PosEmbed(nn.Module):
    def __init__(
            self,
            kernel_size=1,
            embed_dim=256,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        lats = torch.linspace(-90, 90, 121)
        lons = torch.linspace(-180, 178.5, 240)
        grid = torch.meshgrid(lats, lons)
        self.position = torch.stack(grid).reshape(240, 1, 121, 2).cuda()
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=[121, 2], stride=kernel_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self):
        x = self.proj(self.position)
        x = x.flatten(2).transpose(0, 2).transpose(1, 2)
        x = self.norm(x)
        return x


class SphFormer(nn.Module):
    def __init__(
        self,
        img_size=[121, 240],
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
        self.patch_size = img_size[1]
        self.input_size = input_size
        self.token_embeds = PatchEmbed(img_size, input_size, embed_dim)
        self.num_patches = self.token_embeds.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=True)
        # self.pos_embed = PosEmbed(embed_dim=embed_dim)
        self.lmax = 160
        

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
        self.head.append(nn.Linear(embed_dim, self.input_size * 2 * self.img_size[1]))
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
        p = self.img_size[1]
        c = self.input_size
        h = self.img_size[0] // 1 
        w = self.img_size[1] // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(b, h, 1, p, c * 2))
        x = torch.einsum("nhwpc->nhcwp", x)
        x = x.reshape(shape=(b, 1, c * 2, h, p))
        # x = torch.view_as_complex(x)
        # x = self.invsht(x)
        # x = torch.fft.irfft(x, 240)
        imgs = x.reshape(shape=(b, 2, c, h, p))
        return imgs

    def forward_encoder(self, x: torch.Tensor):
        # x: `[B, V, H, W]` shape.
        B, V, H, W = x.shape
        # x = x.reshape(B * V, H, W)
        # x = torch.fft.rfft(x).to(torch.float)
        # print(x.shape)
        
        print(torch.nanmean(torch.abs(x - torch.fft.irfft(torch.fft.fft(x), 240))))
        

        # x = self.sht(x).transpose(1, 2).reshape(B, V, H, H).to(torch.float)
        # x = torch.view_as_real(self.sht(x).reshape(B, V, H, H))
        # print(x.shape)
        # ddd

        # tokenize each variable separately
        x = self.token_embeds(x)

        # pos_embed = self.pos_embed()
        # add pos embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward(self, x):
        
        # print(x.shape)
        out_transformers = self.forward_encoder(x)  # B, L, D
        preds = self.head(out_transformers)  # B, L, V*p*p
        preds = self.unpatchify(preds)
        return preds