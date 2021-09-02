
import h5py
import yaml
import numpy as np
import pandas as pd

from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    def __init__(self, split="train"):
        super().__init__()
        self.split = split

        #len(raw_data) is 60000 samples
        raw_data = datasets.MNIST("", train=True, download=True,
                                  transform=transforms.Compose([transforms.ToTensor()]))
        Xs = []
        ys = []

        for data in raw_data:
            X = data[0] # data[0]: features, data[1] : labels
            # print(X.shape) # torch.Size([1, 28, 28])
            tmp = data[1]  # labels: this will be a scalar value but we want it as a one_hot Vector
            y = torch.Tensor(np.eye(10)[tmp])
            Xs.append(X)
            ys.append(y)
            # print("y=",y)
            # print("argmax = ", torch.argmax(y))
            # plt.imshow(X.view(28, 28))
            # plt.show()

        X_train_val, X_test, y_train_val, y_test = train_test_split(Xs, ys, test_size=0.2, random_state=42)

        if self.split == "train":
            self.X = X_train_val
            self.y = y_train_val
        elif self.split == "test":
            self.X = X_test
            self.y = y_test


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])

