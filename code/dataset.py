from torch.utils.data import Dataset
import numpy as np
import os
from scipy import interpolate
from einops import rearrange
import json
import csv
import torch
from pathlib import Path
import torchvision.transforms as transforms
from scipy.interpolate import interp1d
from typing import Callable, Optional, Tuple, Union
from natsort import natsorted
from glob import glob
import pickle
import utils as ut
from transformers import AutoProcessor
from PIL import Image


def file_ext(name: Union[str, Path]) -> str:
    return str(name).split(".")[-1]


def is_npy_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f"{ext}" == "npy"  # type: ignore


class eeg_pretrain_dataset(Dataset):
    def __init__(self, path="../dreamdiffusion/datasets/mne_data/"):
        super(eeg_pretrain_dataset, self).__init__()
        data = []
        images = []
        self.input_paths = [
            str(f)
            for f in sorted(Path(path).rglob("*"))
            if is_npy_ext(f) and os.path.isfile(f)
        ]

        assert len(self.input_paths) != 0, "No data found"
        self.data_len = 512
        self.data_chan = 128

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index):
        # Ottiene il percorso del file dati corrispondente all'indice
        data_path = self.input_paths[index]

        # Carica i dati EEG dal file numpy
        data = np.load(data_path)

        # Se la lunghezza dei dati è maggiore di self.data_len, estrae una sotto-sequenza casuale di lunghezza self.data_len
        if data.shape[-1] > self.data_len:
            idx = np.random.randint(0, int(data.shape[-1] - self.data_len) + 1)
            data = data[:, idx: idx + self.data_len]
        else:
            # Se la lunghezza dei dati è inferiore, esegue un'interpolazione lineare per adattare la lunghezza desiderata
            x = np.linspace(0, 1, data.shape[-1])
            x2 = np.linspace(0, 1, self.data_len)
            f = interp1d(x, data)
            data = f(x2)

        # Inizializza un array di zeri con dimensioni (self.data_chan, self.data_len)
        ret = np.zeros((self.data_chan, self.data_len))

        if self.data_chan > data.shape[-2]:
            # Se il numero di canali è maggiore rispetto a quelli nei dati effettivi, ripete i dati per riempire gli spazi mancanti
            for i in range((self.data_chan // data.shape[-2])):
                ret[i * data.shape[-2]: (i + 1) * data.shape[-2], :] = data
            if self.data_chan % data.shape[-2] != 0:
                # Aggiunge i dati rimanenti se il numero di canali non è un multiplo dei canali nei dati
                ret[-(self.data_chan % data.shape[-2]):, :] = data[
                    : (self.data_chan % data.shape[-2]), :
                ]
        elif self.data_chan < data.shape[-2]:
            # Se il numero di canali è inferiore, estrae una sotto-sequenza casuale di lunghezza self.data_chan
            idx2 = np.random.randint(
                0, int(data.shape[-2] - self.data_chan) + 1)
            ret = data[idx2: idx2 + self.data_chan, :]
        elif self.data_chan == data.shape[-2]:
            # Se il numero di canali è lo stesso, utilizza direttamente i dati
            ret = data

        # Normalizza i dati dividendo per 10
        ret = ret / 10

        # Converte l'array NumPy in un tensore PyTorch di tipo float
        ret = torch.from_numpy(ret).float()

        # Restituisce un dizionario con la chiave 'eeg' contenente il tensore rappresentante i dati
        return {"eeg": ret}


def create_EEG_dataset(eeg_signals_path='../dreamdiffusion/datasets/eeg_5_95_std.pth',
                       splits_path='../dreamdiffusion/datasets/block_splits_by_image_single.pth',
                       # splits_path = '../dreamdiffusion/datasets/block_splits_by_image_all.pth',
                       image_transform=ut.identity, subject=0):
    # if subject == 0:
    # splits_path = '../dreamdiffusion/datasets/block_splits_by_image_all.pth'
    if isinstance(image_transform, list):
        dataset_train = EEGDataset(
            eeg_signals_path, image_transform[0], subject)
        dataset_test = EEGDataset(
            eeg_signals_path, image_transform[1], subject)
    else:
        dataset_train = EEGDataset(eeg_signals_path, image_transform, subject)
        dataset_test = EEGDataset(eeg_signals_path, image_transform, subject)
    split_train = Splitter(dataset_train, split_path=splits_path,
                           split_num=0, split_name='train', subject=subject)
    split_test = Splitter(dataset_test, split_path=splits_path,
                          split_num=0, split_name='test', subject=subject)
    return (split_train, split_test)


class EEGDataset(Dataset):

    # Constructor
    def __init__(self, eeg_signals_path, image_transform=ut.identity, subject=4):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)

        # if opt.subject!=0:
        #     self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset']) ) if loaded['dataset'][i]['subject']==opt.subject]
        # else:
        # print(loaded)
        if subject != 0:
            self.data = [loaded['dataset'][i] for i in range(
                len(loaded['dataset'])) if loaded['dataset'][i]['subject'] == subject]
        else:
            self.data = loaded['dataset']
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        self.imagenet = '../dreamdiffusion/datasets/imageNet_images'
        self.image_transform = image_transform
        self.num_voxels = 440
        self.data_len = 512
        # Compute size
        self.size = len(self.data)
        self.processor = AutoProcessor.from_pretrained(
            "openai/clip-vit-large-patch14")
     # Get size

    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        # print(self.data[i])
        eeg = self.data[i]["eeg"].float().t()

        eeg = eeg[20:460, :]
        # 2023 2 13 add preprocess and transpose
        eeg = np.array(eeg.transpose(0, 1))
        x = np.linspace(0, 1, eeg.shape[-1])
        x2 = np.linspace(0, 1, self.data_len)
        f = interp1d(x, eeg)
        eeg = f(x2)
        eeg = torch.from_numpy(eeg).float()
        # 2023 2 13 add preprocess
        label = torch.tensor(self.data[i]["label"]).long()

        # Get label
        image_name = self.images[self.data[i]["image"]]
        image_path = os.path.join(self.imagenet, image_name.split('_')[
                                  0], image_name+'.JPEG')
        # print(image_path)
        image_raw = Image.open(image_path).convert('RGB')

        image = np.array(image_raw) / 255.0
        image_raw = self.processor(images=image_raw, return_tensors="pt")
        image_raw['pixel_values'] = image_raw['pixel_values'].squeeze(0)

        return {'eeg': eeg, 'label': label, 'image': self.image_transform(image), 'image_raw': image_raw}
        # Return
        # return eeg, label


class Splitter:

    def __init__(self, dataset, split_path, split_num=0, split_name="train", subject=4):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)

        self.split_idx = loaded["splits"][split_num][split_name] #669 index
        # Filter data
        
        #Mantieni gli indici che hanno la dimensione delle colonne del tensor tra 450 e 600
        # Compute size
        # self.split_idx = [i for i in self.split_idx if i <= len(
        #     self.dataset.data) and 450 <= self.dataset.data[i]["eeg"].size(1) <= 600] 

        self.size = len(self.split_idx)
        self.num_voxels = 440
        self.data_len = 512

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        return self.dataset[self.split_idx[i]]
