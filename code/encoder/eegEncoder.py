import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from encoder.maskedAutoEncoder import CustomBlock, PatchEmbed1D
import utils as ut

class eegEncoder(nn.Module):
    def __init__(self, time_len=512, patch_size=4, embed_dim=512, in_chans=128,
                 depth=24, num_heads=16, norm_layer=nn.LayerNorm):
        super().__init__()
        self.patch_embed = PatchEmbed1D(
            time_len, patch_size, in_chans, embed_dim)

        num_patches = int(time_len / patch_size)

        self.num_patches = num_patches
        self.cls_token = nn.Parameter(torch.zeros(
            1, 1, embed_dim))  # torch.Size([1, 1, 1024])
        self.pos_embed = nn.Parameter(
            # torch.Size([1, 129, 1024])
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                CustomBlock(
                    embed_dim,
                    num_heads
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.embed_dim = embed_dim

        self.patch_size = patch_size
        self.num_patches = num_patches
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = ut.get_1d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.num_patches, cls_token=True)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        x = x + self.pos_embed[:, 1:, :]

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        # if self.global_pool:
        #     x = x.mean(dim=1, keepdim=True)

        x = self.norm(x)

        return x

    def forward(self, imgs):
        if imgs.ndim == 2:
            imgs = torch.unsqueeze(imgs, dim=0)  # N, n_seq, embed_dim
        latent = self.forward_encoder(imgs)  # N, n_seq, embed_dim
        return latent  # N, n_seq, embed_dim

    def load_checkpoint(self, state_dict):
        # if self.global_pool:
        #     state_dict = {k: v for k, v in state_dict.items() if (
        #         'mask_token' not in k and 'norm' not in k)}
        # else:
        state_dict = {k: v for k, v in state_dict.items()
                      if ('mask_token' not in k)}
        ut.interpolate_pos_embed(self, state_dict)

        m, u = self.load_state_dict(state_dict, strict=False)
        print('missing keys:', u)
        print('unexpected keys:', m)
        return

