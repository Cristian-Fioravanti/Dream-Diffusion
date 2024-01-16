import numpy as np
import torch
from egg_ldm.eggLDM import eLDM
import utils as ut
import torchvision.transforms as transforms
from dataset import create_EEG_dataset
from torch.utils.data import DataLoader
import argparse
import pytorch_lightning as pl
import datetime
import os
from config import Config_Generative_Model, Config_MBM_EEG


def main(config):
    # Project setup
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Set up image transformations for training and testing
    crop_pix = int(config.crop_ratio * config.img_size)

    # Training image transformations
    img_transform_train = transforms.Compose(
        [
            ut.normalize,
            transforms.Resize((512, 512)),
            ut.random_crop(config.img_size - crop_pix, p=0.5),
            transforms.Resize((512, 512)),
            ut.channel_last,
        ]
    )

    # Testing image transformations
    img_transform_test = transforms.Compose(
        [ut.normalize, transforms.Resize((512, 512)), ut.channel_last]
    )

    # Create EEG dataset based on configuration
    if config.dataset == "EEG":
        eeg_latents_dataset_train, eeg_latents_dataset_test = create_EEG_dataset(
            eeg_signals_path=config.eeg_signals_path,
            splits_path=config.splits_path,
            image_transform=[img_transform_train, img_transform_test],
            subject=config.subject,
        )
        num_voxels = eeg_latents_dataset_train.data_len
    else:
        # If the dataset is not EEG, raise an exception (you might want to handle other datasets here)
        raise NotImplementedError

    # Load pretrained model metafile
    pretrain_mbm_metafile = torch.load(
        config.pretrain_mbm_path, map_location="cpu")

    # Create generative model (eLDM)
    generative_model = eLDM(
        pretrain_mbm_metafile,
        num_voxels,
        device=device,
        pretrain_root=config.pretrain_gm_path,
        logger=config.logger,
        ddim_steps=config.ddim_steps,
        global_pool=config.global_pool,
        use_time_cond=config.use_time_cond,
        clip_tune=config.clip_tune,
        cls_tune=config.cls_tune,
    )

    # Resume training if applicable
    if config.checkpoint_path is not None:
        model_meta = torch.load(config.checkpoint_path, map_location="cpu")
        generative_model.model.load_state_dict(model_meta["model_state_dict"])
        print("model resumed")

    # Create trainer for finetuning the model
    trainer = create_trainer(
        config.num_epoch,
        config.precision,
        config.accumulate_grad,
        config.logger,
        check_val_every_n_epoch=2,
    )

    generative_model.finetune(
        trainer,
        eeg_latents_dataset_train,
        eeg_latents_dataset_test,
        config.batch_size,
        config.lr,
        config.output_path,
        config=config,
    )

    # Generate images
    # Generate limited train images and generate images for subjects separately
    ut.generate_images(generative_model, eeg_latents_dataset_train,
                       eeg_latents_dataset_test, config)

    return


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


def create_trainer(
    num_epoch,
    precision=32,
    accumulate_grad_batches=2,
    logger=None,
    check_val_every_n_epoch=0,
):
    acc = "gpu" if torch.cuda.is_available() else "cpu"
    return pl.Trainer(
        accelerator=acc,
        max_epochs=num_epoch,
        logger=logger,
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
        enable_checkpointing=False,
        enable_model_summary=False,
        gradient_clip_val=0.5,
        check_val_every_n_epoch=check_val_every_n_epoch,
    )


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

    output_path = os.path.join(
        config.output_path,
        "results",
        "generation",
        "%s" % (datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")),
    )
    config.output_path = output_path
    os.makedirs(output_path, exist_ok=True)

    folder_init(config, output_path)

    # logger = WandbLogger()
    config.logger = None  # logger
    main(config)
