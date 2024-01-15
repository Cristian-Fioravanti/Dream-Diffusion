import math
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
import time
from load.load_dataset import loadEEG_5_95, loadOneEEG_5_95
from scalar.scalarNative import NativeScalerWithGradNormCount
from encoder.maskedAutoEncoder import MaskedAutoEncoderEEG
from utils import save_model
import timm.optim.optim_factory as optim_factory
import datetime
import matplotlib.pyplot as plt

from config import Config_MBM_EEG
from dataset import eeg_pretrain_dataset
import utils as ut

# def fmri_transform(x, sparse_rate=0.2):
#     # x: 1, num_voxels
#     x_aug = copy.deepcopy(x)
#     idx = np.random.choice(x.shape[0], int(x.shape[0]*sparse_rate), replace=False)
#     x_aug[idx] = 0
#     return torch.FloatTensor(x_aug)


def main(config):
    print(config)
    config = prepareOutputPath(config)
    print(config)
    device = torch.device("cpu")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if config.load_all_dataset:
        loadEEG_5_95()
    if config.load_one_dataset:
        loadOneEEG_5_95()

    # create dataset and dataloader
    dataset_pretrain = eeg_pretrain_dataset(
        path="../dreamdiffusion/datasets/eegdataset/eeg",
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
        use_nature_img_loss=config.use_nature_img_loss,
    )
    model.to(device)
    model_without_ddp = model

    param_groups = optim_factory.add_weight_decay(
        model, config.weight_decay
    )  # utilizzato per gestire la decrescita del peso durante l'ottimizzazione.
    optimizer = torch.optim.AdamW(
        param_groups, lr=config.lr, betas=(0.9, 0.95)
    )  # istanziato l'ottimizzatore AdamW, lr=learning rate
    print(optimizer)
    loss_scaler = (
        NativeScalerWithGradNormCount()
    )  # fornisce strumenti per eseguire addestramento di reti neurali con precisione mista automaticamente. La classe GradScaler in AMP è progettata per facilitare l'uso della precisione mista. La precisione mista è una tecnica in cui alcune parti del modello vengono eseguite in precisione ridotta (ad esempio, FP16) per migliorare l'efficienza computazionale, mentre altre parti rimangono in precisione completa (ad esempio, FP32) per mantenere la stabilità numerica.

    cor_list = []  # lista per tracciare la correlazione durante l'addestramento
    start_time = time.time()
    print("Start Training the EEG MAE ... ...")
    img_feature_extractor = None
    preprocess = None
    # if config.use_nature_img_loss:
    #     from torchvision.models import resnet50, ResNet50_Weights
    #     from torchvision.models.feature_extraction import create_feature_extractor

    #     weights = ResNet50_Weights.DEFAULT
    #     preprocess = weights.transforms()
    #     m = resnet50(weights=weights)
    #     img_feature_extractor = (
    #         create_feature_extractor(m, return_nodes={f"layer2": "layer2"})
    #         .to(device)
    #         .eval()
    #     )
    #     for param in img_feature_extractor.parameters():
    #         param.requires_grad = False

    for ep in range(config.num_epoch):
        # if torch.cuda.device_count() > 1:
        #     sampler.set_epoch(ep)  # to shuffle the data at every epoch
        cor = train_one_epoch(
            model,
            dataloader_eeg,
            optimizer,
            device,
            ep,
            loss_scaler,
            None,
            config,
            start_time,
            model_without_ddp,
            img_feature_extractor,
            preprocess,
        )
        cor_list.append(cor)
        if (
            ep % 20 == 0 or ep + 1 == config.num_epoch
        ) and config.local_rank == 0:  # and ep != 0
            # save models
            # if True:
            save_model(
                config,
                ep,
                model_without_ddp,
                optimizer,
                loss_scaler,
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
    # if logger is not None:
    #     logger.log("max cor", np.max(cor_list), step=config.num_epoch - 1)
    #     logger.finish()
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


# @torch.no_grad()
# def plot_recon_figures2(
#     model,
#     device,
#     dataset,
#     output_path,
#     num_figures=5,
#     config=None,
#     logger=None,
#     model_without_ddp=None,
# ):
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
#     model.eval()
#     fig, axs = plt.subplots(num_figures, 2, figsize=(20, 15))
#     fig.tight_layout()
#     axs[0, 0].set_title("Ground-truth")
#     # axs[0,1].set_title('Masked Ground-truth')
#     axs[0, 1].set_title("Reconstruction")

#     for ax in axs:
#         sample = next(iter(dataloader))["eeg"]
#         sample = sample.to(device)
#         _, pred, mask = model(sample, mask_ratio=config.mask_ratio)
#         # sample_with_mask = model_without_ddp.patchify(sample.transpose(1,2))[0].to('cpu').numpy().reshape(-1, model_without_ddp.patch_size)
#         sample_with_mask = (
#             sample.to("cpu")
#             .squeeze(0)[0]
#             .numpy()
#             .reshape(-1, model_without_ddp.patch_size)
#         )
#         # pred = model_without_ddp.unpatchify(pred.transpose(1,2)).to('cpu').squeeze(0)[0].unsqueeze(0).numpy()
#         # sample = sample.to('cpu').squeeze(0)[0].unsqueeze(0).numpy()
#         pred = model_without_ddp.unpatchify(pred).to("cpu").squeeze(0)[0].numpy()
#         # pred = model_without_ddp.unpatchify(model_without_ddp.patchify(sample.transpose(1,2))).to('cpu').squeeze(0)[0].numpy()
#         sample = sample.to("cpu").squeeze(0)[0].numpy()
#         cor = np.corrcoef([pred, sample])[0, 1]

#         x_axis = np.arange(0, sample.shape[-1])
#         # groundtruth
#         ax[0].plot(x_axis, sample)

#         ax[1].plot(x_axis, pred)
#         ax[1].set_ylabel("cor: %.4f" % cor, weight="bold")
#         ax[1].yaxis.set_label_position("right")

#     fig_name = "reconst-%s" % (datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
#     fig.savefig(os.path.join(output_path, f"{fig_name}.png"))
#     if logger is not None:
#         logger.log_image("reconst", fig)
#     plt.close(fig)


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
    # config.output_path = output_path

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
    loss_scaler,
    log_writer=None,
    config=None,
    start_time=None,
    model_without_ddp=None,
    img_feature_extractor=None,
    preprocess=None,
):
    # Imposta il modello in modalità di addestramento
    model.train(True)
    optimizer.zero_grad()  # Azzera i gradienti dell'ottimizzatore
    total_loss = []  # Accumulatore per le perdite
    total_cor = []  # Accumulatore per le correlazioni
    accum_iter = (
        config.accum_iter
    )  # Numero di iterazioni prima di eseguire un'ottimizzazione

    # Iterazione attraverso il dataloader
    for data_iter_step, (data_dcit) in enumerate(data_loader):
        # Regolazione del tasso di apprendimento in base al numero di iterazioni
        if data_iter_step % accum_iter == 0:
            ut.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, config
            )

        samples = data_dcit["eeg"]  # Campioni di EEG dal dataloader

        img_features = None
        valid_idx = None
        if img_feature_extractor is not None:
            # Gestione delle immagini se è abilitata la perdita di immagine naturale
            images = data_dcit["image"]
            valid_idx = torch.nonzero(
                images.sum(dim=(1, 2, 3)) != 0).squeeze(1)
            img_feature_extractor.eval()
            with torch.no_grad():
                img_features = img_feature_extractor(
                    preprocess(images[valid_idx]).to(device)
                )["layer2"]

        samples = samples.to(device)  # Sposta i campioni sulla GPU

        optimizer.zero_grad()  # Azzera i gradienti dell'ottimizzatore
        with torch.cuda.amp.autocast(
            enabled=True
        ):  # abilitando la modalità di autocasting automatico per eseguire operazioni in precisione mista durante il forward e il backward pass del modello
            loss, pred, _ = model(
                samples, img_features, valid_idx=valid_idx, mask_ratio=config.mask_ratio
            )  # esegue il forward del modello
        # loss.backward()
        # norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        # optimizer.step()

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(
                f"Loss is {loss_value}, stopping training at step {data_iter_step} epoch {epoch}"
            )
            sys.exit(1)

        # loss /= accum_iter
        loss_scaler(
            loss, optimizer, parameters=model.parameters(), clip_grad=config.clip_grad
        )  # esegue __call__ di NativeScalerWithGradNormCount e modifica i gradienti

        # if (data_iter_step + 1) % accum_iter == 0:
        # cal the cor
        pred = pred.to("cpu").detach()
        samples = samples.to("cpu").detach()
        # pred = pred.transpose(1,2) #model_without_ddp.unpatchify(pred)
        # trasforma il tensore nell'immagine
        pred = model_without_ddp.unpatchify(pred)
        # print(pred.shape)
        # print(samples.shape)
        # for p, s in zip(pred, samples):
        #     print(p[0], s[0])
        #     print(torch.cat([p[0].unsqueeze(0), s[0].unsqueeze(0)],axis=0))
        #     print(torch.corrcoef(torch.cat([p[0].unsqueeze(0), s[0].unsqueeze(0)],axis=0)))
        #     print(torch.corrcoef(torch.cat([p[0].unsqueeze(0), s[0].unsqueeze(0)],axis=0))[0,1])

        # Viene calcolata la correlazione media tra le predizioni (pred) e i campioni di input (samples)
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

        total_loss.append(loss_value)
        total_cor.append(cor)
        if device == torch.device("cuda:0"):
            lr = optimizer.param_groups[0]["lr"]
            print(
                "train_loss_step:",
                np.mean(total_loss),
                "lr:",
                lr,
                "cor",
                np.mean(total_cor),
            )

    if log_writer is not None:
        lr = optimizer.param_groups[0]["lr"]
        log_writer.log("train_loss_step", np.mean(total_loss), step=epoch)
        log_writer.log("lr", lr, step=epoch)
        log_writer.log("cor", np.mean(total_cor), step=epoch)
        if start_time is not None:
            log_writer.log("time (min)", (time.time() -
                           start_time) / 60.0, step=epoch)
    if config.local_rank == 0:
        print(f"[Epoch {epoch}] loss: {np.mean(total_loss)}")

    return np.mean(total_cor)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    config = Config_MBM_EEG()
    config = update_config(args, config)
    main(config)
