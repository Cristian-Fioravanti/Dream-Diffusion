from diffusers import DDPMScheduler, UNet2DConditionModel
from PIL import Image
import torch
from pathlib import Path

from encoder.eegEncoder import eegEncoder
from config import Config_Generative_Model, Config_MBM_EEG
from dataset import create_EEG_dataset
from diffusers import AutoencoderKL, DDPMScheduler, LMSDiscreteScheduler, StableDiffusionPipeline, UNet2DConditionModel, PNDMScheduler

import torch
import torch.nn as nn
import argparse
import datetime
import os
import torchvision.transforms as transforms
import utils as ut
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from transformers import CLIPTextModel
import torch.nn.functional as F
import clip as CLIP
from einops import rearrange
from tqdm.auto import tqdm
        
class ProjectionLayerEmbedding(nn.Module):
    def __init__(self, input_size, output_size, device):
        self.device = device
        super(ProjectionLayerEmbedding, self).__init__()
        self.projection = nn.Linear(input_size, output_size).to(device)

    def forward(self, x):
        # Reshape the input tensor to have a single batch dimension
        x = self.projection(x.flatten()).to(self.device)
        # Apply the linear projection
        return torch.reshape(x, (77, 768))

def main(config):
    logging_dir = os.path.join(config.output_path, "accLog/")
    # Pretrianed model loading
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrain_model = torch.load('pretrains\eeg_pretrain\checkpoint.pth', map_location=device)
    metafile_config = pretrain_model['config']

    # Training image transformations
    img_transform_train = transforms.Compose(
        [
            ut.normalize,
            transforms.Resize((512, 512)),
            transforms.RandomCrop(size=(512, 512)),
            transforms.Normalize([0.5], [0.5]),
            # transforms.Resize((512, 512)),
            ut.channel_last,
        ]
    )

    # Testing image transformations
    img_transform_test = transforms.Compose(
        [ut.normalize, transforms.Resize((512, 512)), ut.channel_last]
    )

    # Definition of train and test datasets
    _, eeg_latents_dataset_test = create_EEG_dataset(
        eeg_signals_path=config.eeg_signals_path,
        splits_path=config.splits_path,
        image_transform=[img_transform_train, img_transform_test],
        subject=config.subject,
    )
    data_len_eeg = eeg_latents_dataset_test.data_len
    # Accelerator
    accelerator_project_config = ProjectConfiguration(
        project_dir=config.output_path, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=None,
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )
    # Finetuned model loading
    folder_path = "exps/results/generation/"
    model_name = "checkpoint_epoch.pth"
    model_path = Path(folder_path) / model_name
    generative_model = torch.load(model_path, map_location=device)

    # Definition of the models useful for generation
    encoder = eegEncoder(time_len=data_len_eeg, patch_size=metafile_config.patch_size, embed_dim=metafile_config.embed_dim,
                         depth=metafile_config.depth, num_heads=metafile_config.num_heads, mlp_ratio=metafile_config.mlp_ratio)

    scheduler = PNDMScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="unet")
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="vae")
    projector1 = ProjectionLayerEmbedding(128*metafile_config.embed_dim, 59136, device)

    # Loading of the models' states
    unet.load_state_dict(generative_model['unet_state_dict'])
    encoder.load_state_dict(generative_model['egg_encoder_state_dict'])
    vae.load_state_dict(generative_model['vae_state_dict'])
    projector1.load_state_dict(generative_model['projector1'])

    encoder.to(device)
    projector1.to(device)
    vae.to(device)
    unet.to(device)

    for step, batch in enumerate(eeg_latents_dataset_test):
        eeg = batch["eeg"].to(device)
        image = batch["image"].to(device)
        # EEG related image is saved
        image_real = Image.fromarray(
            ((image.cpu() / 2 + 0.5).clamp(0, 1) * 255).numpy().astype('uint8'))
        image_real_path = os.path.join(config.output_path,
                                    f'real/{step}.jpg')
        image_real.save(image_real_path)

        # EEG embedding are retrieved using encoder
        embeddings = encoder(eeg).to(device)
        # Embeddings are projected 
        hidden_states = projector1(embeddings).to(device)
        # Image is prepared for encoding
        image_for_encode = change_shape_for_encode(image).to(device)
        del embeddings, eeg, image
        
        # The VAE model is used to encode the image into latent data
        latents = vae.encode(image_for_encode).latent_dist.sample().to(device)
        latents = latents * vae.config.scaling_factor
        
        # A random noise is initialized
        input = torch.randn_like(latents).to(accelerator.device).to(device)
        del latents, image_for_encode
        
        # Definition of timesteps
        timesteps = 999
        scheduler.set_timesteps(timesteps)
        for t in tqdm(scheduler.timesteps):
            with torch.no_grad():
                # Generating image using UNet and Scheduler  
                noisy_residual = unet(input, t, hidden_states.unsqueeze(0), return_dict=False)[0].to(device)
                prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
                # Generated image is used as input of the next iteration
                input = prev_noisy_sample
        del timesteps, hidden_states, noisy_residual
        input = 1 / 0.18215 * input
        with torch.no_grad():
            # Generated image is decoded thanks to VAE decoder
            image_gen = vae.decode(input).sample

        # Generated image is saved
        image_gen = (image_gen / 2 + 0.5).clamp(0, 1)
        image_gen = image_gen.cpu().permute(0, 2, 3, 1).numpy()[0]
        image_gen = Image.fromarray((image_gen * 255).round().astype("uint8"))

        image_gen_path = os.path.join(config.output_path,
                                    f'test/{step}.jpg')

        image_gen.save(image_gen_path)

def change_shape_for_encode(image):
    return rearrange(image, "h w c-> c h w").unsqueeze(0).to(dtype=torch.float32)


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
    # config.output_path = output_path

    os.makedirs(os.path.join(output_path,'real/'))
    os.makedirs(os.path.join(output_path,'test/'))

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
            if getattr(args, attr) != None:
                setattr(config, attr, getattr(args, attr))
    return config

def get_args_parser():
    parser = argparse.ArgumentParser(
        'Double Conditioning LDM Finetuning', add_help=False)
    # project parameters
    parser.add_argument('--seed', type=int)
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--pretrain_mbm_path', type=str)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--crop_ratio', type=float)
    parser.add_argument('--dataset', type=str)

    # finetune parameters
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--precision', type=int)
    parser.add_argument('--accumulate_grad', type=int)
    parser.add_argument('--global_pool', type=bool)

    # diffusion sampling parameters
    parser.add_argument("--pretrain_gm_path", type=str)
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--ddim_steps", type=int)
    parser.add_argument("--use_time_cond", type=bool)
    parser.add_argument("--eval_avg", type=bool)

    # # distributed training parameters
    # parser.add_argument('--local_rank', type=int)

    return parser


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    config = Config_Generative_Model()
    config = update_config(args, config)

    if config.checkpoint_path is not None:
        model_meta = torch.load(config.checkpoint_path, map_location="cpu")
        ckp = config.checkpoint_path
        config = model_meta["config"]
        config.checkpoint_path = ckp
        print("Resuming from checkpoint: {}".format(config.checkpoint_path))

    prepareOutputPath(config)

    # logger = WandbLogger()
    config.logger = None  # logger
    main(config)
