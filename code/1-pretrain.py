import math
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
import time
from load.load_dataset import loadEEG_5_95, loadOneEEG_5_95
from encoder.maskedAutoEncoder import MaskedAutoEncoderEEG
from utils import save_model
import timm.optim.optim_factory as optim_factory
import datetime
import matplotlib.pyplot as plt

from config import Config_MBM_EEG
from dataset import eeg_pretrain_dataset
import utils as ut

def main(config):
    config = prepareOutputPath(config)

    device = torch.device("cpu")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if config.load_all_dataset:
        loadEEG_5_95()
    if config.load_one_dataset:
        loadOneEEG_5_95()

    # create dataset and dataloader
    dataset_pretrain = eeg_pretrain_dataset(
        path="datasets/eegdataset/eeg",
    )

    dataloader_eeg = DataLoader(
        dataset_pretrain,
        batch_size=config.batch_size,
        pin_memory=True,
    )

    # create model
    config.time_len = dataset_pretrain.data_len
    model = MaskedAutoEncoderEEG(
        time_len=dataset_pretrain.data_len,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        decoder_embed_dim=config.decoder_embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        decoder_num_heads=config.decoder_num_heads,
        mlp_ratio=config.mlp_ratio,
        focus_range=config.focus_range,
        focus_rate=config.focus_rate,
        img_recon_weight=config.img_recon_weight,
    )
    model.to(device)
    model_without_ddp = model

    param_groups = optim_factory.add_weight_decay(
        model, config.weight_decay
    )  # used to manage weight decay during optimization.

    optimizer = torch.optim.AdamW(
        param_groups, lr=config.lr, betas=(0.9, 0.95)
    )  # AdamW optimizer instantiated

    start_time = time.time()
    print("Start Training the EEG MAE ... ...")

    for ep in range(config.num_epoch):
        print(f"Currently on Epoch {ep} ...")
        train_one_epoch(
            model,
            dataloader_eeg,
            optimizer,
            device,
            ep,
            config,
            model_without_ddp
        )
        if (
            ep % 20 == 0 or ep + 1 == config.num_epoch
        ) and config.local_rank == 0:  # and ep != 0
            # save models
            save_model(
                config,
                ep,
                model_without_ddp,
                optimizer,
                os.path.join(config.output_path, "checkpoints"),
            )
            # plot figures
            plot_recon_figures(
                model,
                device,
                dataset_pretrain,
                config.output_path,
                5,
                config,
                None,
                model_without_ddp,
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    return

@torch.no_grad()
def plot_recon_figures(
    model,
    device,
    dataset,
    output_path,
    num_figures=5,
    config=None,
    logger=None,
    model_without_ddp=None,
):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model.eval()
    fig, axs = plt.subplots(num_figures, 3, figsize=(30, 15))
    fig.tight_layout()
    axs[0, 0].set_title("Ground-truth")
    axs[0, 1].set_title("Masked Ground-truth")
    axs[0, 2].set_title("Reconstruction")

    for ax in axs:
        sample = next(iter(dataloader))["eeg"]
        sample = sample.to(device)
        _, pred, mask = model(sample, mask_ratio=config.mask_ratio)
        # sample_with_mask = model_without_ddp.patchify(sample.transpose(1,2))[0].to('cpu').numpy().reshape(-1, model_without_ddp.patch_size)
        sample_with_mask = (
            sample.to("cpu")
            .squeeze(0)[0]
            .numpy()
            .reshape(-1, model_without_ddp.patch_size)
        )
        # pred = model_without_ddp.unpatchify(pred.transpose(1,2)).to('cpu').squeeze(0)[0].unsqueeze(0).numpy()
        # sample = sample.to('cpu').squeeze(0)[0].unsqueeze(0).numpy()
        pred = model_without_ddp.unpatchify(
            pred).to("cpu").squeeze(0)[0].numpy()
        # pred = model_without_ddp.unpatchify(model_without_ddp.patchify(sample.transpose(1,2))).to('cpu').squeeze(0)[0].numpy()
        sample = sample.to("cpu").squeeze(0)[0].numpy()
        mask = mask.to("cpu").numpy().reshape(-1)

        cor = np.corrcoef([pred, sample])[0, 1]

        x_axis = np.arange(0, sample.shape[-1])
        # groundtruth
        ax[0].plot(x_axis, sample)
        # groundtruth with mask
        s = 0
        for x, m in zip(sample_with_mask, mask):
            if m == 0:
                ax[1].plot(x_axis[s: s + len(x)], x, color="#1f77b4")
            s += len(x)
        # pred
        ax[2].plot(x_axis, pred)
        ax[2].set_ylabel("cor: %.4f" % cor, weight="bold")
        ax[2].yaxis.set_label_position("right")

    fig_name = "reconst-%s" % (
        datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    fig.savefig(os.path.join(output_path, f"{fig_name}.png"))
    if logger is not None:
        logger.log_image("reconst", fig)
    plt.close(fig)

def update_config(args, config):
    for attr in config.__dict__:
        if hasattr(args, attr):
            if getattr(args, attr) != None:
                setattr(config, attr, getattr(args, attr))
    return config


def prepareOutputPath(config):
    output_path = os.path.join(
        config.root_path,
        "results",
        "eeg_pretrain",
        "%s" % (datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")),
    )

    setattr(config, "output_path", output_path)

    if config.local_rank == 0:
        os.makedirs(output_path, exist_ok=True)
        create_readme(config, output_path)
    return config


def get_args_parser():
    parser = argparse.ArgumentParser(
        "MBM pre-training for fMRI", add_help=False)

    # Training Parameters
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--num_epoch", type=int)
    parser.add_argument("--batch_size", type=int)

    # Model Parameters
    parser.add_argument("--mask_ratio", type=float)
    parser.add_argument("--patch_size", type=int)
    parser.add_argument("--embed_dim", type=int)
    parser.add_argument("--decoder_embed_dim", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--decoder_num_heads", type=int)
    parser.add_argument("--mlp_ratio", type=float)

    # Project setting
    parser.add_argument("--root_path", type=str)
    parser.add_argument("--seed", type=str)
    parser.add_argument("--roi", type=str)
    parser.add_argument("--aug_times", type=int)
    parser.add_argument("--num_sub_limit", type=int)

    parser.add_argument("--include_hcp", type=bool)
    parser.add_argument("--include_kam", type=bool)

    parser.add_argument("--use_nature_img_loss", type=bool)
    parser.add_argument("--img_recon_weight", type=float)

    # distributed training parameters
    parser.add_argument("--local_rank", type=int)

    return parser


def create_readme(config, path):
    print(config.__dict__)
    with open(os.path.join(path, "README.md"), "w+") as f:
        print(config.__dict__, file=f)


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
    epoch,
    config=None,
    model_without_ddp=None
):
    # Set the model to training mode
    model.train(True)
    optimizer.zero_grad()  # Resets the optimizer gradients
    total_loss = []  # Loss accumulator
    total_cor = []  # Correlations accumulator

    # Iterating through the dataloader
    data_dcit = next(iter(data_loader), None)
    # Adjusting the learning rate based on the number of iterations
    ut.adjust_learning_rate(optimizer, epoch, config)

    samples = data_dcit["eeg"]  # Samples of EEG from dataloader torch.Size([1, 128, 512])
    samples = samples.to(device)  # Move samples to device

    optimizer.zero_grad()  # Resets the optimizer gradients

    # enabling automatic autocasting mode to perform mixed precision operations during forward and backward pass of the model
    # execute forward passof the model 
    with torch.autocast(device_type="cpu"):
       loss, pred, _ = model(samples, mask_ratio=config.mask_ratio)  
    
    loss_value = loss.item()
    
    if not math.isfinite(loss_value):
        print(f"Loss is not finite: {loss_value}, stopping training at epoch {epoch}")
        sys.exit(1)

    # loss_scaler(loss, optimizer, parameters=model.parameters(), clip_grad=config.clip_grad) 
    # cal the cor
    print("Loss: "+str(loss_value))
    # Backward operation
    loss.backward(create_graph=False)   
    optimizer.step()
    
    pred = pred.to("cpu").detach()
    samples = samples.to("cpu").detach()
  
    # Transform tensor in image
    pred = model_without_ddp.unpatchify(pred)
  
    # The average correlation between the predictions (pred) and the input samples (samples) is calculated
    cor = torch.mean(
        torch.tensor(
            [
                torch.corrcoef(
                    torch.cat(
                        [p[0].unsqueeze(0), s[0].unsqueeze(0)], axis=0)
                )[0, 1]
                for p, s in zip(pred, samples)
            ]
        )
    ).item()
    optimizer.zero_grad()

    total_cor.append(cor)
    lr = optimizer.param_groups[0]["lr"]
    print(
        "train_loss_step:",
        np.mean(total_loss),
        "lr:",
        lr,
        "cor",
        np.mean(total_cor),
    )
    return


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    config = Config_MBM_EEG()
    config = update_config(args, config)
    main(config)
