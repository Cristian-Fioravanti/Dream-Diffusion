from encoder.eegEncoder import eegEncoder
from config import Config_Generative_Model, Config_MBM_EEG
from dataset import create_EEG_dataset
from diffusers import AutoencoderKL, DDPMScheduler,LMSDiscreteScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers import LMSDiscreteScheduler

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
# 1 Caricare il modello pretrained V
# 2 Prendere labeledEEG gli facciamo l'encoder per ricavare la noise 
# 3 aggiungiamo a questa noise una randomNoise
# 4 passare tutto alla uNET per poi calcolarci la loss A

# 5 prendo l'immagine dell'egg relativo, lo croppo
# 6 uso l'encoder di clip per trovare l'embedding dell'immagine
# 7 utilizzo una projection per poter confrontare i due embedding trovati
# 8 calcolo la cosine-similarity tra i due embeddign, ricevendo un altra loss B

# 10 loss.backward()
# 11 optimizer.step()
# 11 optimizer.zero_grad()

def main(config):
    ## 1 Caricare il modello pretrained
    pretrain_model = torch.load('..\dreamdiffusion\pretrains\eeg_pretrain\checkpoint.pth', map_location="cpu")
    metafile_config = pretrain_model['config']
  

    ## 2 Prendere labeledEEG gli facciamo l'encoder per ricavare la noise
    # Training image transformations
    img_transform_train = transforms.Compose(
        [
            ut.normalize,
            transforms.Resize((512, 512)),
            transforms.RandomCrop(size=(64, 64)),
            transforms.Normalize([0.5], [0.5]),
            # transforms.Resize((512, 512)),
            ut.channel_last,
        ]
    )

    # Testing image transformations
    img_transform_test = transforms.Compose(
        [ut.normalize, transforms.Resize((512, 512)), ut.channel_last]
    )
    crop_transform = transforms.CenterCrop((224,224))
        
    eeg_latents_dataset_train, eeg_latents_dataset_test = create_EEG_dataset(
        eeg_signals_path=config.eeg_signals_path,
        splits_path=config.splits_path,
        image_transform=[img_transform_train, img_transform_test],
        subject=config.subject,
    )
    data_len_eeg = eeg_latents_dataset_train.data_len
    #Carico Encoder
    encoder = eegEncoder(time_len=data_len_eeg, patch_size=metafile_config.patch_size, embed_dim=metafile_config.embed_dim,
                        depth=metafile_config.depth, num_heads=metafile_config.num_heads)
   
    encoder.load_checkpoint(pretrain_model['model'])
    # Accelerator da vedere
    accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision='fp16')
   
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

    # scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    scheduler =  DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    clip_model, _ = CLIP.load(name="ViT-B/32", device="cpu")    
    projector1 = nn.Linear(1024, 768)
    projection_layer = ProjectionLayerH(128 * 768, 512)   
    
    vae.required_grad=False
    clip_model.required_grad=False
    projection_layer.required_grad=False

    unet.train()
    encoder.train()
    projector1.train()
    optimizer = torch.optim.AdamW(list(unet.parameters()) + 
                                  list(encoder.parameters()) + 
                                  list(projector1.parameters()),lr=config.lr)
    
    unet,encoder, vae,scheduler,optimizer,projector1,clip_model,crop_transform =  accelerator.prepare(unet,encoder, vae,scheduler,optimizer,projector1,clip_model,crop_transform)
    ## For Epoch
    train_loss = 0.0
    for epoch in range(0, config.num_epoch):
        current_dateTime = datetime.datetime.now()
        print("DataInizio: "+ str(current_dateTime))
        print(f"Epoch: {epoch}")
        for step, batch in enumerate(eeg_latents_dataset_train ):
            eeg = batch["eeg"]
            image = batch["image"]
            with accelerator.accumulate(unet, encoder,projector1):
                embeddings = encoder(eeg) #from torch.Size([128, 512]) to torch.Size([1, 128, 1024])
                hidden_states = projector1(embeddings)
                image_for_encode = change_shape_for_encode(image)
                
                # Convert images to latent space
                latents = vae.encode(image_for_encode).latent_dist.sample() # need torch.Size([1, 3, 64, 64])
                latents = latents * vae.config.scaling_factor

                # 3 aggiungiamo a questa noise una randomNoise
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents).to(accelerator.device)

                #bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, 1000, (1,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
              
                # 4 passare tutto alla uNET per poi calcolarci la loss A
                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, hidden_states, return_dict=False)[0] #RuntimeError: mat1 and mat2 shapes cannot be multiplied (128x1024 and 768x320)
                loss_unet = F.mse_loss(model_pred,noise,reduction="mean")
                
                # 5 prendo l'immagine dell'egg relativo, lo croppo    
                image_cropped= crop_transform(image_for_encode)
                # 6 uso l'encoder di clip per trovare l'embedding dell'immagine
                image_encoded = clip_model.encode_image(image_cropped)
                
                # 7 utilizzo una projection per poter confrontare i due embedding trovati
                projection = projection_layer(hidden_states)
                # 8 calcolo la cosine-similarity tra i due embeddign, ricevendo un altra loss B

                # hidden_states.shape torch.Size([1, 128, 768])
                # projection   .shape torch.Size([1, 128, 512])
                # image_encoded.shape torch.Size([1, 512])
                loss_clip = 1 - F.cosine_similarity(image_encoded,projection)
                
                # 9 somme le due loss (A + B)
                total_loss = loss_unet + loss_clip
                print(f"{total_loss.item()} Step: {str(step)}")
                
                accelerator.backward(total_loss)
                optimizer.step()
                optimizer.zero_grad()
        current_dateTime = datetime.datetime.now()
        print("DataFine: "+ str(current_dateTime))
        torch.save(
                    {
                       'unet_state_dict': unet.state_dict(),
                        'egg_encoder_state_dict': encoder.state_dict(),
                        'vae_state_dict': vae.state_dict(),
                        # 'scheduler_state_dict': scheduler.state_dict(),
                        'clip_model_state_dict': clip_model.state_dict(),
                        'projector1': projector1.state_dict(),
                        'config': config,
                        'state': torch.random.get_rng_state()

                    },
                    os.path.join(config.output_path, 'checkpoint.pth'))                      
                
                
class ProjectionLayerH(nn.Module):
    def __init__(self, input_size, output_size):
        super(ProjectionLayerH, self).__init__()
        self.projection = nn.Linear(input_size, output_size)

    def forward(self, x):
        # Reshape the input tensor to have a single batch dimension
        x = x.view(1, -1)
        # Apply the linear projection
        x = self.projection(x)
        return x



def save_model(config, epoch, model, optimizer, loss_scaler, checkpoint_paths):
    os.makedirs(checkpoint_paths, exist_ok=True)
    to_save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'scaler': loss_scaler.state_dict(),
        'config': config,
    }
    torch.save(to_save, os.path.join(checkpoint_paths, 'checkpoint.pth'))
    
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

    return config


def folder_init(config, output_path):
    # wandb.init( project='dreamdiffusion',
    #             group="stageB_dc-ldm",
    #             anonymous="allow",
    #             config=config,
    #             reinit=True)
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
    parser.add_argument('--root_path', type=str, default='../dreamdiffusion/')
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

def change_shape_for_encode(image):
    return rearrange(image, "h w c-> c h w").unsqueeze(0).to(dtype=torch.float32)

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
