import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from dataloader import extract_ecg_file_list, extract_data

from os.path import abspath, realpath

class AtrialFibrillationDataset(Dataset):
    def __init__(self, split="train"):
        super().__init__()
        self.split = split

        datapath = 'data'
        if self.split == "train":
            print("Loading the training dataset..")
            x_train, y_train = extract_data(extract_ecg_file_list(datapath, "train.csv"))
            x_train = np.delete(x_train, 1, axis=2) # Input has two channels, take the first channel of the input
            x_train = x_train.reshape((len(x_train), 1, 5250)) # transpose the input: (len(data),5250,1) -> (len(data),1,5250)
            x_train = x_train.astype('float32') # cast the array from float64 to float32. to avoid RuntimeError: expected scalar type Double but found Float when performing batch norm and input quantization.
            self.X = x_train
            self.y = y_train

        elif self.split == "test":
            print("Loading the test dataset..")
            x_test, y_test = extract_data(extract_ecg_file_list(datapath, "test.csv"))
            x_test = np.delete(x_test, 1, axis=2)
            x_test = x_test.reshape((len(x_test), 1, 5250))
            x_test = x_test.astype('float32')
            self.X = x_test
            self.y = y_test

        elif self.split =="valid":
            print("Loading the validation dataset..")
            x_valid, y_valid = extract_data(extract_ecg_file_list(datapath, "validation.csv"))
            x_valid = np.delete(x_valid, 1, axis=2)
            x_valid = x_valid.reshape((len(x_valid), 1, 5250))
            x_valid = x_valid.astype('float32')
            self.X = x_valid
            self.y = y_valid

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])