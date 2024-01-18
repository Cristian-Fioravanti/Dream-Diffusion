from egg_ldm.plms import PLMSSampler
from encoder.maskedAutoEncoder import PatchEmbed1D
from timm.models.vision_transformer import Block
from omegaconf import OmegaConf
import torch
import utils as ut
from torch.utils.data import DataLoader
from einops import rearrange, repeat
from PIL import Image
from torchvision.utils import make_grid
import os
import torch.nn as nn
import numpy as np


class eLDM:
    # Latent Diffusion Model
    def __init__(self, metafile, num_voxels, device=torch.device('cpu'),
                 pretrain_root='../pretrains/',
                 logger=None, ddim_steps=250, global_pool=True, use_time_cond=False, clip_tune=True, cls_tune=False):
        # self.ckp_path = os.path.join(pretrain_root, 'model.ckpt')
        self.ckp_path = os.path.join(pretrain_root, 'models/v1-5-pruned.ckpt')
        self.config_path = os.path.join(pretrain_root, 'models/config15.yaml')
        config = OmegaConf.load(self.config_path)
        config.model.params.unet_config.params.use_time_cond = use_time_cond
        config.model.params.unet_config.params.global_pool = global_pool

        self.cond_dim = config.model.params.unet_config.params.context_dim

        model = ut.instantiate_from_config(config.model)
        print("Start loading ")
        pl_sd = torch.load(self.ckp_path, map_location="cpu")['state_dict']

        model.cond_stage_trainable = True
        model.cond_stage_model = cond_stage_model(
            metafile, num_voxels, self.cond_dim, global_pool=global_pool, clip_tune=clip_tune, cls_tune=cls_tune)

        model.ddim_steps = ddim_steps
        model.re_init_ema()
        if logger is not None:
            logger.watch(model, log="all", log_graph=False)

        model.p_channels = config.model.params.channels
        model.p_image_size = config.model.params.image_size
        model.ch_mult = config.model.params.first_stage_config.params.ddconfig.ch_mult

        self.device = device
        self.model = model

        self.model.clip_tune = clip_tune
        self.model.cls_tune = cls_tune

        self.ldm_config = config
        self.pretrain_root = pretrain_root
        self.egg_latent_dim = model.cond_stage_model.egg_latent_dim
        self.metafile = metafile

    def finetune(self, trainers, dataset, test_dataset, bs1, lr1,
                 output_path, config=None):
        config.trainer = None
        config.logger = None
        self.model.main_config = config
        self.model.output_path = output_path
        # self.model.train_dataset = dataset
        self.model.run_full_validation_threshold = 0.15
        # stage one: train the cond encoder with the pretrained one

        # # stage one: only optimize conditional encoders
        print('\n##### Stage One: only optimize conditional encoders #####')
        dataloader = DataLoader(dataset, batch_size=bs1,
                                shuffle=True, num_workers=4)
        test_loader = DataLoader(
            test_dataset, batch_size=bs1, shuffle=False, num_workers=4)
        self.model.unfreeze_whole_model()
        self.model.freeze_first_stage()
        # self.model.freeze_whole_model()
        # self.model.unfreeze_cond_stage()

        self.model.learning_rate = lr1
        self.model.train_cond_stage_only = True
        self.model.eval_avg = config.eval_avg

        trainers.fit(self.model, dataloader, val_dataloaders=test_loader)

        self.model.unfreeze_whole_model()

        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'config': config,
                'state': torch.random.get_rng_state()

            },
            os.path.join(output_path, 'checkpoint.pth')
        )

    @torch.no_grad()
    def generate(self, fmri_embedding, num_samples, ddim_steps, HW=None, limit=None, state=None, output_path=None):
        all_samples = []
        if HW is None:
            shape = (self.ldm_config.model.params.channels,
                     self.ldm_config.model.params.image_size, self.ldm_config.model.params.image_size)
        else:
            num_resolutions = len(
                self.ldm_config.model.params.first_stage_config.params.ddconfig.ch_mult)
            shape = (self.ldm_config.model.params.channels,
                     HW[0] // 2**(num_resolutions-1), HW[1] // 2**(num_resolutions-1))

        model = self.model.to(self.device)
        sampler = PLMSSampler(model)
        # sampler = DDIMSampler(model)
        if state is not None:
            torch.cuda.set_rng_state(state)

        with model.ema_scope():
            model.eval()
            # print((fmri_embedding.size)) #669
            for count, item in enumerate(fmri_embedding):
                if limit is not None:
                    if count >= limit:
                        break
                print(item)
                latent = item['eeg']
                gt_image = rearrange(
                    item['image'], 'h w c -> 1 c h w')  # h w c
                print(
                    f"rendering {num_samples} examples in {ddim_steps} steps.")

                c, re_latent = model.get_learned_conditioning(
                    repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                # c = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                 conditioning=c,
                                                 batch_size=num_samples,
                                                 shape=shape,
                                                 verbose=False)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp(
                    (x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                gt_image = torch.clamp((gt_image+1.0)/2.0, min=0.0, max=1.0)

                # put groundtruth at first
                all_samples.append(
                    torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0))
                if output_path is not None:
                    samples_t = (
                        255. * torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0).numpy()).astype(np.uint8)
                    for copy_idx, img_t in enumerate(samples_t):
                        img_t = rearrange(img_t, 'c h w -> h w c')
                        Image.fromarray(img_t).save(os.path.join(output_path,
                                                                 f'./test{count}-{copy_idx}.png'))

        # display as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=num_samples+1)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        model = model.to('cpu')
        return grid, (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8)


class cond_stage_model(nn.Module):
    def __init__(self, metafile, num_voxels=440, cond_dim=1280, global_pool=True, clip_tune=True, cls_tune=False):
        super().__init__()

        if metafile is not None: # Must Be not None Cri
            model = create_model_from_config(
                metafile['config'], num_voxels, global_pool)

            model.load_checkpoint(metafile['model'])
        # else:
        #     model = eeg_encoder(time_len=num_voxels, global_pool=global_pool)
        self.mae = model
        if clip_tune:
            self.mapping = mapping()
        if cls_tune:
            self.cls_net = classify_network()

        self.egg_seq_len = model.num_patches
        self.egg_latent_dim = model.embed_dim
        if global_pool == False:
            self.channel_mapper = nn.Sequential(
                nn.Conv1d(self.egg_seq_len,
                          self.egg_seq_len // 2, 1, bias=True),
                nn.Conv1d(self.egg_seq_len // 2, 77, 1, bias=True)
            )
        self.dim_mapper = nn.Linear(self.egg_latent_dim, cond_dim, bias=True)
        self.global_pool = global_pool

        # self.image_embedder = FrozenImageEmbedder()

    # def forward(self, x):
    #     # n, c, w = x.shape
    #     latent_crossattn = self.mae(x)
    #     if self.global_pool == False:
    #         latent_crossattn = self.channel_mapper(latent_crossattn)
    #     latent_crossattn = self.dim_mapper(latent_crossattn)
    #     out = latent_crossattn
    #     return out

    def forward(self, x):
        # n, c, w = x.shape
        latent_crossattn = self.mae(x)
        latent_return = latent_crossattn
        if self.global_pool == False:
            latent_crossattn = self.channel_mapper(latent_crossattn)
        latent_crossattn = self.dim_mapper(latent_crossattn)
        out = latent_crossattn
        return out, latent_return

    # def recon(self, x):
    #     recon = self.decoder(x)
    #     return recon

    def get_cls(self, x):
        return self.cls_net(x)

    def get_clip_loss(self, x, image_embeds):
        # image_embeds = self.image_embedder(image_inputs)
        target_emb = self.mapping(x)
        # similarity_matrix = nn.functional.cosine_similarity(target_emb.unsqueeze(1), image_embeds.unsqueeze(0), dim=2)
        # loss = clip_loss(similarity_matrix)
        loss = 1 - \
            torch.cosine_similarity(target_emb, image_embeds, dim=-1).mean()
        return loss


def create_model_from_config(config, num_voxels, global_pool):
    model = eeg_encoder(time_len=num_voxels, patch_size=config.patch_size, embed_dim=config.embed_dim,
                        depth=config.depth, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, global_pool=global_pool)
    return model


class mapping(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.AdaptiveAvgPool1d((1))
        self.maxpool = nn.Conv1d(128, 1, 1, stride=1)
        self.fc = nn.Linear(1024, 768)

    def forward(self, x):
        x = self.maxpool(x)
        x = x.squeeze(1)
        x = self.fc(x)
        return x


# class eeg_encoder(nn.Module):
#     def __init__(self, time_len=512, patch_size=4, embed_dim=1024, in_chans=128,
#                  depth=24, num_heads=16, mlp_ratio=1., norm_layer=nn.LayerNorm, global_pool=False):
#         super().__init__()
#         self.patch_embed = PatchEmbed1D(
#             time_len, patch_size, in_chans, embed_dim)

#         num_patches = int(time_len / patch_size)

#         self.num_patches = num_patches
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(torch.zeros(
#             1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

#         self.blocks = nn.ModuleList([
#             Block(embed_dim, num_heads, mlp_ratio,
#                   qkv_bias=True, norm_layer=norm_layer)
#             for i in range(depth)])
#         self.norm = norm_layer(embed_dim)

#         self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.embed_dim = embed_dim

#         self.patch_size = patch_size
#         self.num_patches = num_patches
#         self.global_pool = global_pool
#         self.initialize_weights()

#     def initialize_weights(self):
#         # initialization
#         # initialize (and freeze) pos_embed by sin-cos embedding
#         pos_embed = ut.get_1d_sincos_pos_embed(
#             self.pos_embed.shape[-1], self.num_patches, cls_token=True)
#         self.pos_embed.data.copy_(
#             torch.from_numpy(pos_embed).float().unsqueeze(0))

#         # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
#         w = self.patch_embed.proj.weight.data
#         torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
#         # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
#         torch.nn.init.normal_(self.cls_token, std=.02)
#         torch.nn.init.normal_(self.mask_token, std=.02)
#         # initialize nn.Linear and nn.LayerNorm
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             # we use xavier_uniform following official JAX ViT:
#             torch.nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv1d):
#             torch.nn.init.normal_(m.weight, std=.02)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)

#     def forward_encoder(self, x):
#         # embed patches
#         x = self.patch_embed(x)

#         # add pos embed w/o cls token
#         # print(x.shape)
#         # print(self.pos_embed[:, 1:, :].shape)
#         x = x + self.pos_embed[:, 1:, :]
#         # apply Transformer blocks
#         for blk in self.blocks:
#             x = blk(x)
#         # print(x.shape)
#         if self.global_pool:
#             x = x.mean(dim=1, keepdim=True)
#         # print(x.shape)
#         x = self.norm(x)
#         # print(x.shape)
#         return x

#     def forward(self, imgs):
#         if imgs.ndim == 2:
#             imgs = torch.unsqueeze(imgs, dim=0)  # N, n_seq, embed_dim
#         latent = self.forward_encoder(imgs)  # N, n_seq, embed_dim
#         return latent  # N, n_seq, embed_dim

#     def load_checkpoint(self, state_dict):
#         if self.global_pool:
#             state_dict = {k: v for k, v in state_dict.items() if (
#                 'mask_token' not in k and 'norm' not in k)}
#         else:
#             state_dict = {k: v for k, v in state_dict.items()
#                           if ('mask_token' not in k)}
#         ut.interpolate_pos_embed(self, state_dict)

#         m, u = self.load_state_dict(state_dict, strict=False)
#         print('missing keys:', u)
#         print('unexpected keys:', m)
#         return
# Cri

class classify_network(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.AdaptiveAvgPool1d((1))
        self.maxpool = nn.Conv1d(128, 1, 1, stride=1)
        self.fc = nn.Linear(1024, 40)

    def forward(self, x):
        x = self.maxpool(x)
        x = x.squeeze(1)
        x = self.fc(x)
        return x
