import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import utils as ut

"""
I parametri: time_len, patch_size, in_chans, e embed_dim 
specificano le dimensioni del problema e del modello. In particolare, time_len rappresenta la lunghezza della sequenza temporale, patch_size specifica la dimensione delle "patch" (o segmenti) della sequenza, in_chans è il numero di canali di input, e embed_dim è la dimensione dell'embedding per ogni patch.
num_patches rappresenta il numero di patch ottenute dividendo la lunghezza della sequenza temporale per la dimensione della patch.
"""


class PatchEmbed1D(nn.Module):
    def __init__(self, time_len=224, patch_size=1, in_chans=128, embed_dim=256):
        super().__init__()
        num_patches = time_len // patch_size  # 224
        self.patch_shape = patch_size
        self.time_len = time_len
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv1d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x, **kwargs):

        x = (
            # patch proiettate  nella dimensione dell'embedding embed_dim
            self.proj(x).transpose(1, 2).contiguous() #from torch.Size([1, 1024, 128]) torch.Size([1, 128, 1024])
        )  # put embed_dim at the last dimension

        return x


class CustomBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Self-attention layer
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)

        # Feedforward layer
        self.feedforward = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention and layer normalization
        # ogni elemento della sequenza viene utilizzato per calcolare i pesi rispetto a tutti gli altri elementi nella stessa sequenza
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feedforward and layer normalization
        ff_output = self.feedforward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x


class MaskedAutoEncoderEEG(nn.Module):
    def __init__(
        self,
        time_len=512,
        patch_size=4,
        embed_dim=1024,
        in_chans=128,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        norm_layer=nn.LayerNorm,
        focus_range=None,
        focus_rate=None,
        img_recon_weight=1.0,
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed1D(
            time_len, patch_size, in_chans, embed_dim)

        num_patches = int(time_len / patch_size)

        self.num_patches = num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # torch.Size([1, 1, 1024])
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False #torch.Size([1, 129, 1024])
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
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                CustomBlock(
                    decoder_embed_dim,
                    decoder_num_heads

                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, in_chans * patch_size, bias=True
        )  # encoder to decoder

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.focus_range = focus_range
        self.focus_rate = focus_rate
        self.img_recon_weight = img_recon_weight

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        pos_embed = ut.get_1d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.num_patches, cls_token=True
        )
        decoder_pos_embed = ut.get_1d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], self.num_patches, cls_token=True
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )  # trasforma gli embedding in tensor

        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )  # trasforma gli embedding in tensor

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(
            self._init_weights
        )  # _init_weights viene applicato per inizializzare i pesi e bias per i livelli lineari e la normalizzazione del livello all'interno del modello

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
            torch.nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # Prende un tensore di immagini e lo converte in un formato di patch
    def patchify(self, imgs):
        """
        imgs: (N, 1, num_voxels)
        imgs: [N, chan, T]
        x: (N, L, patch_size)
        x: [N, chan * 4, T/4]
        """
        p = self.patch_embed.patch_size
        assert imgs.ndim == 3 and imgs.shape[1] % p == 0

        # h = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1] // p, -1))
        return x

    # prende un tensore di patch e lo converte in un formato di immagine
    def unpatchify(self, x):
        """
        x: (N, L, patch_size)
        imgs: (N, 1, num_voxels)
        """
        p = self.patch_embed.patch_size
        h = x.shape[1]

        imgs = x.reshape(shape=(x.shape[0], -1, x.shape[2] // p))
        return imgs.transpose(1, 2)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        if self.focus_range is not None:
            len_mask = L - len_keep
            weights = [1 - self.focus_rate] * L
            weights[
                self.focus_range[0]
                // self.patch_size: self.focus_range[1]
                // self.patch_size
            ] = [self.focus_rate] * (
                self.focus_range[1] // self.patch_size
                - self.focus_range[0] // self.patch_size
            )
            weights = torch.tensor(weights).repeat(N, 1).to(x.device)
            ids_mask = torch.multinomial(weights, len_mask, replacement=False)

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        if self.focus_range is not None:
            for i in range(N):
                noise[i, ids_mask[i, :]] = 1.1  # set mask portion to 1.1

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        #x torch.Size([1, 1024, 128])

        # embed patches
        x = self.patch_embed(x)  # patch di input nel modello torch.Size([1, 128, 1024])

        # add pos embed with out the first column that is for the cls token
        x = (
            x + self.pos_embed[:, 1:, :] 
        )  # Aggiunge gli embedding posizionali senza il token di classe

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(
            x, mask_ratio
        )  # Applica una mascheratura casuale all'input e restituisce l'output mascherato
        #x torch.Size([1, 115, 1024])

        # Crea un token di classe con l'embedding posizionale torch.Size([1, 1, 1024])
        cls_token = (
            self.cls_token + self.pos_embed[:, :1, :]
        ) 

        # Espande il token di classe per adattarlo alle dimensioni dell'input torch.Size([1, 116, 1024])
        cls_tokens = cls_token.expand(
            x.shape[0], -1, -1
        )  
        # Aggiunge il token di classe all'inizio delle patch
        x = torch.cat(
            (cls_tokens, x), dim=1
        )  
        # x torch.Size([1, 116, 1024])

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(
                x
            )  # Applica una serie di blocchi Transformer (self.blocks) all'input
        # Normalizza l'output utilizzando la normalizzazione di layer
        # x torch.Size([1, 116, 1024])
        x = self.norm(x)
        
        return x, mask, ids_restore

    

    def forward_decoder(self, x, ids_restore=None):
        #x torch.Size([1, 116, 1024])

        # embed tokens
        x = self.decoder_embed(x) #torch.Size([1, 116, 512])
        # print('decoder embed')
        # print(x.shape)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )  # Concatena i token di decodifica e i token di maschera,
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token

        # x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle #Riordina la sequenza basandosi sugli indici per ripristinare l'ordine originale

        x = torch.cat(
            [x[:, :1, :], x_], dim=1
        )  # append cls token # Aggiunge nuovamente il token di classe alla sequenza
        # x = x_
        # add pos embed
        x = x + self.decoder_pos_embed  # Aggiunge embedding posizionali
        # x = x + self.decoder_pos_embed[:, 1:, :]

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)  # Applica una serie di blocchi
        x = self.decoder_norm(
            x
        )  # Normalizza l'output utilizzando la normalizzazione di layer
        # print(x.shape)
        # predictor projection
        # Implementando una trasformazione lineare (nn.Linear) Proietta l'output 
        x = self.decoder_pred(x)
        # print(x.shape)

        # remove cls token
        x = x[:, 1:, :]  # Rimuove il token di classe dall'output

        return x  # Restituisce l'output decodificato

  
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 1, num_voxels]
        imgs: [N, chan, T]
        pred: [N, L, p]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        imgs = imgs.transpose(1, 2)
        target = self.patchify(imgs)
        # target = imgs.transpose(1,2)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], Mean Squared Error per patch
        # loss = loss.mean()
        loss = (
            (loss * mask).sum() /
            mask.sum() if mask.sum() != 0 else (loss * mask).sum()
        )  # mean loss on removed patches
        return loss

    def forward(self, imgs,  mask_ratio=0.75):
        #imgs torch.Size([1, 128, 512])

        # latent = self.forward_encoder(imgs, mask_ratio)
        latent, mask, ids_restore = self.forward_encoder(
            imgs, mask_ratio
        )  # Esegue la codifica delle immagini di input masked

        # latent torch.Size([1, 116, 1024])
        
        pred = self.forward_decoder(
            latent, ids_restore
        )  # [N, L, p] # Esegue la decodifica
        # pred torch.Size([1, 128, 512])
       
        loss = self.forward_loss(
            imgs, pred, mask
        )  # Calcola la perdita basata sulle immagini originali e le predizioni decodificate
        # print(self.unpatchify(pred.transpose(1,2)).shape)

        return loss, pred, mask
