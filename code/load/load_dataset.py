import os
import torch
import numpy as np


def loadEEG_5_95():
    eeg_data = torch.load("datasets/eeg_5_95_std.pth")
    # save one file for example purposes - don't run this unless needed

    # Assuming you've already loaded eeg_data from the .pth file
    dataset = eeg_data["dataset"]

    # Define a base path to save
    base_save_path = "datasets/eegdataset/"

    # Loop through all items in the dataset
    for idx, tensor_item in enumerate(dataset):
        if idx < 1:
            # Loop over all attributes in the tensor_item
            for key, value in tensor_item.items():
                # Construct the subfolder path based on the attribute
                subfolder_path = os.path.join(base_save_path, key)

                # Check if the subfolder exists, if not, create it
                if not os.path.exists(subfolder_path):
                    os.makedirs(subfolder_path)

                # If the value is a torch.Tensor, convert it to a numpy array
                if isinstance(value, torch.Tensor):
                    ndarray = value.numpy()
                    try:
                        np.save(f"{subfolder_path}/0.npy", ndarray)
                    except Exception as e:
                        print(f"Error saving file at index 0: {e}")
                else:
                    # If the value is not a tensor, simply save it as it is
                    try:
                        np.save(f"{subfolder_path}/0.npy", np.array(value))
                    except Exception as e:
                        print(f"Error saving file at index 0: {e}")


def loadOneEEG_5_95():
    eeg_data = torch.load("datasets/eeg_5_95_std.pth")

    # save one file for example purposes - don't run this unless needed

    # Assuming you've already loaded eeg_data from the .pth file
    tensor_item = eeg_data["dataset"][0]

    # Define a base path to save
    base_save_path = "datasets/eegdataset/"

    # Loop over all attributes in the tensor_item
    for key, value in tensor_item.items():
        # Construct the subfolder path based on the attribute
        subfolder_path = os.path.join(base_save_path, key)

        # Check if the subfolder exists, if not, create it
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        # If the value is a torch.Tensor, convert it to a numpy array
        if isinstance(value, torch.Tensor):
            ndarray = value.numpy()
            try:
                np.save(f"{subfolder_path}/0.npy", ndarray)
            except Exception as e:
                print(f"Error saving file at index 0: {e}")
        else:
            # If the value is not a tensor, simply save it as it is
            try:
                np.save(f"{subfolder_path}/0.npy", np.array(value))
            except Exception as e:
                print(f"Error saving file at index 0: {e}")
