import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import datetime
import os
import math
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from transformers import CLIPTextModel
import torch.nn.functional as F
from einops import rearrange
from encoder.eegEncoder import eegEncoder
from config import Config_Generative_Model, Config_MBM_EEG
from dataset import create_EEG_dataset
from diffusers import AutoencoderKL, DDPMScheduler, LMSDiscreteScheduler, StableDiffusionPipeline, UNet2DConditionModel
import argparse
import utils as ut
import clip as CLIP
import gc

def main(config):
    device = "cpu"
    pretrain_model = torch.load('../dreamdiffusion/pretrains/eeg_pretrain/checkpoint.pth', map_location=device)
    metafile_config = pretrain_model['config']

    img_transform_train = transforms.Compose([
        ut.normalize,
        transforms.Resize((512, 512)),
        transforms.RandomCrop(size=(64, 64)),
        transforms.Normalize([0.5], [0.5]),
        ut.channel_last,
    ])

    img_transform_test = transforms.Compose([
        ut.normalize,
        transforms.Resize((512, 512)),
        ut.channel_last
    ])

    crop_transform = transforms.CenterCrop((224, 224))

    eeg_latents_dataset_train, eeg_latents_dataset_test = create_EEG_dataset(
        eeg_signals_path=config.eeg_signals_path,
        splits_path=config.splits_path,
        image_transform=[img_transform_train, img_transform_test],
        subject=config.subject,
    )

    data_len_eeg = eeg_latents_dataset_train.data_len

    encoder = eegEncoder(time_len=data_len_eeg, patch_size=metafile_config.patch_size, embed_dim=metafile_config.embed_dim,
                         depth=metafile_config.depth, num_heads=metafile_config.num_heads, mlp_ratio=metafile_config.mlp_ratio)

    encoder.load_checkpoint(pretrain_model['model'])

    unet = UNet2DConditionModel.from_pretrained(
       "runwayml/stable-diffusion-v1-5", subfolder="unet")
    vae = AutoencoderKL.from_pretrained(
       "runwayml/stable-diffusion-v1-5", subfolder="vae")

    scheduler = DDPMScheduler.from_pretrained(
       "runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    clip_model, _ = CLIP.load(name="ViT-L/14", device=device)
    projector1 = ProjectionLayerEmbedding(128*metafile_config.embed_dim, 59136, device)
    projection_layer = ProjectionLayer(29568, 768, device)

    vae.required_grad = False
    clip_model.required_grad = False
    projection_layer.required_grad = False

    accelerator = Accelerator()
    unet, encoder, vae, scheduler, projector1, clip_model, crop_transform = accelerator.prepare(
        unet, encoder, vae, scheduler, projector1, clip_model, crop_transform)

    optimizer = torch.optim.AdamW(list(unet.parameters()) +
                                  list(encoder.parameters()) +
                                  list(projector1.parameters()), lr=config.lr)
    try:
        # Training loop
        for epoch in range(config.num_epoch):
            current_dateTime = datetime.datetime.now()
            print("DataInizio: " + str(current_dateTime))
            print(f"Epoch: {epoch}")

            for step, batch in enumerate(eeg_latents_dataset_train):
                eeg = batch["eeg"].to(device)
                image = batch["image"].to(device)

                with accelerator.accumulate(unet, encoder, projector1):
                    embeddings = encoder(eeg).to(device)
                    hidden_states = projector1(embeddings).to(device)
                    del embeddings
                    image_for_encode = change_shape_for_encode(
                        image).to(device)

                    latents = vae.encode(image_for_encode).latent_dist.sample().to(device)
                    latents = latents * vae.config.scaling_factor

                    noise = torch.randn_like(latents).to(device)
                    timesteps = torch.randint(0, 1000, (1,),
                                            device=device).long()

                    noisy_latents = scheduler.add_noise(
                        latents, noise, timesteps).to(device)
                    del latents

                    model_pred = unet(noisy_latents, timesteps, hidden_states.unsqueeze(0), return_dict=False)[
                        0].to(device)
                    loss_unet = F.mse_loss(
                        model_pred, noise, reduction="mean").to(device)

                    del model_pred, noise
                    image_cropped = crop_transform(image_for_encode).to(device)
                    del image_for_encode
                    image_encoded = clip_model.encode_image(
                        image_cropped).to(device)

                    projection = projection_layer(hidden_states).to(device)
                    loss_clip = 1 - F.cosine_similarity(image_encoded, projection).to(device)

                    total_loss = loss_unet + loss_clip
                    del loss_unet, loss_clip
                    if not math.isfinite(total_loss.item()):
                        exit(1)

                    del eeg, image, timesteps

                    accelerator.backward(total_loss.to(device))
                    current_dateTime = datetime.datetime.now()
                    print(str(total_loss.item()) + " Step: " + str(step) + " " + str(current_dateTime))
                    del total_loss, projection
                    gc.collect()
                    torch.cuda.empty_cache()

                    optimizer.step()
                    optimizer.zero_grad()

            current_dateTime = datetime.datetime.now()
            print("DataFine: " + str(current_dateTime))
            if epoch % 5 == 0:
                save_model(unet, encoder, vae, clip_model,
                        projector1, config, config.output_path, epoch)
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        save_model(unet, encoder, vae, clip_model, projector1, config, config.output_path, epoch=None)
        exit(0)

def save_model(unet, encoder, vae, clip_model, projector1, config, output_path, epoch=None):
    torch.save(
        {
            'unet_state_dict': unet.state_dict(),
            'egg_encoder_state_dict': encoder.state_dict(),
            'vae_state_dict': vae.state_dict(),
            'clip_model_state_dict': clip_model.state_dict(),
            'projector1': projector1.state_dict(),
            'config': config,
            'state': torch.random.get_rng_state()
        },
        os.path.join(output_path, f'checkpoint_epoch_{epoch}.pth') if epoch is not None else os.path.join(
            output_path, 'manual_checkpoint.pth')
    )

class ProjectionLayer(nn.Module):
    def __init__(self, input_size, output_size, device):
        super(ProjectionLayer, self).__init__()
        self.device = device
        self.l1 =nn.Linear(768,384).to(device)
        self.projection = nn.Linear(input_size, output_size).to(device)

    def forward(self, x):
        x = self.l1(x).to(self.device)
        # Reshape the input tensor to have a single batch dimension
        x = self.projection(x.flatten()).to(self.device)
        # Apply the linear projection

        return torch.reshape(x,(1,768))
    
class ProjectionLayerEmbedding(nn.Module):
    def __init__(self, input_size, output_size, device):
        super(ProjectionLayerEmbedding, self).__init__()
        self.device = device
        self.projection = nn.Linear(input_size, output_size).to(device)

    def forward(self, x):
        # Reshape the input tensor to have a single batch dimension
        x = self.projection(x.flatten()).to(self.device)
        # Apply the linear projection

        return torch.reshape(x,(77,768))

def prepareOutputPath(config):
    output_path = os.path.join(
        config.output_path,
        "results",
        "generation",
        "%s" % (datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")),
    )
    os.makedirs(output_path, exist_ok=True)

    folder_init(config, output_path)
    setattr(config, "output_path", output_path)

    return config


def folder_init(config, output_path):
    create_readme(config, output_path)


def create_readme(config, path):
    print(config.__dict__)
    with open(os.path.join(path, "README.md"), "w+") as f:
        print(config.__dict__, file=f)


def update_config(args, config):
    for attr in config.__dict__:
        if hasattr(args, attr):
            if getattr(args, attr) is not None:
                setattr(config, attr, getattr(args, attr))
    return config


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Double Conditioning LDM Finetuning', add_help=False)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--root_path', type=str, default='../dreamdiffusion/')
    parser.add_argument('--pretrain_mbm_path', type=str)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--crop_ratio', type=float)
    parser.add_argument('--dataset', type=str)

    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--precision', type=int)
    parser.add_argument('--accumulate_grad', type=int)
    parser.add_argument('--global_pool', type=bool)

    parser.add_argument("--pretrain_gm_path", type=str)
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--ddim_steps", type=int)
    parser.add_argument("--use_time_cond", type=bool)
    parser.add_argument("--eval_avg", type=bool)

    return parser


def change_shape_for_encode(image):
    return rearrange(image, "h w c-> c h w").unsqueeze(0).to(dtype=torch.float32)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    config = Config_Generative_Model()
    config = update_config(args, config)

    prepareOutputPath(config)

    accelerator = Accelerator()
    config = accelerator.prepare(config)

    if config.checkpoint_path is not None:
        model_meta = torch.load(
            config.checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        ckp = config.checkpoint_path
        config = model_meta["config"]
        config.checkpoint_path = ckp
        print("Resuming from checkpoint: {}".format(config.checkpoint_path))

    main(config)
