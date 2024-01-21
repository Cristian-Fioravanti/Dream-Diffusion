from encoder.eegEncoder import eegEncoder
from config import Config_Generative_Model, Config_MBM_EEG
from dataset import create_EEG_dataset
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
import torch
import argparse
import datetime
import os
import torchvision.transforms as transforms
import utils as ut
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
# 1 Caricare il modello pretrained V
# 2 Prendere labeledEEG gli facciamo l'encoder per ricavare la noise 
# 3 aggiungiamo a questa noise una randomNoise
# 4 passare tutto alla uNET per poi calcolarci la loss A

# 5 prendo l'immagine dell'egg relativo, lo croppo
# 6 uso l'encoder di clip per trovare l'embedding dell'immagine
# 7 utilizzo una projection per poter confrontare i due embedding trovati
# 8 calcolo la cosine-similarity tra i due embeddign, ricevendo un altra loss B
# 9 somme le due loss (A + B)
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
        [
            ut.normalize,
            transforms.Resize((512, 512)),
            transforms.RandomCrop(size=(64, 64)),
            transforms.Normalize([0.5], [0.5]),
            ut.channel_last]
    )

    # Testing image transformations
    img_transform_test = transforms.Compose(
        [ut.normalize, transforms.Resize((512, 512)), ut.channel_last]
    )
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
    logging_dir = os.path.join(config.output_path, "accLog/")
    accelerator_project_config = ProjectConfiguration(project_dir=config.output_path, logging_dir=logging_dir)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=None,
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )
    # UNet da vedere
    unet = UNet2DConditionModel()
    # Projector da vedere
    projector = None
    # Vae da vedere
    vae = AutoencoderKL()

    ## For Epoch
    train_loss = 0.0
    for epoch in range(0, config.num_epoch):
        print(f"Epoch: {epoch}")
        for step, batch in enumerate(eeg_latents_dataset_train ):
            eeg = batch["eeg"]
            image = batch["image"]
            with accelerator.accumulate(unet, projector, encoder):
                embeddings = encoder(eeg)
                hidden_states = projector(embeddings)

                # Convert images to latent space
                latents = vae.encode(image).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
##################################################################################
                # 3 aggiungiamo a questa noise una randomNoise
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                # noise = torch.randn_like(latents).to(accelerator.device)

                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                if args.input_perturbation:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
    

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
