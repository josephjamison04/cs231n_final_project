from multiprocessing.spawn import _main
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import sampler

import torchvision.transforms as T

class FCNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FCNet, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def load_data(train =True,valid = True,test = False):
    # Load the raw CIFAR-10 data.
    data_dir = '/home/ubuntu/CS231N/data/split_datasets/'
    # data_dir = "../../data/"
    X_train,y_train,X_valid,y_valid,X_test,y_test = None, None, None, None, None, None
    if train:
        X_train = pd.read_pickle(data_dir + "train_data.pkl").to_numpy()
        y_train = pd.read_pickle(data_dir + "train_labels.pkl").to_numpy()
        y_train = y_train.flatten().astype(np.int64)
        print('Training data shape: ', X_train.shape)
        print('Training labels shape: ', y_train.shape)
    if valid:
        X_valid = pd.read_pickle(data_dir + "valid_data.pkl").to_numpy()
        y_valid = pd.read_pickle(data_dir + "valid_labels.pkl").to_numpy()
        y_valid = y_valid.flatten().astype(np.int64)
        print('Validation data shape: ', X_valid.shape)
        print('Validation labels shape: ', y_valid.shape)
    if test:
        X_test = pd.read_pickle(data_dir + "test_data.pkl").to_numpy()
        y_test = pd.read_pickle(data_dir + "test_labels.pkl").to_numpy()
        y_test = y_test.flatten().astype(np.int64)
        print('Test data shape: ', X_test.shape)
        print('Test labels shape: ', y_test.shape)
    # As a sanity check, we print out the size of the data we output.
    # if didn;t load the data, it is None
    return X_train,y_train,X_valid,y_valid,X_test,y_test

def train(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    # load train and valid data
    X_train,y_train,X_valid,y_valid,_ ,_ =load_data()
    

    input_size = 3 * 128 * 128  # image size
    num_classes = 10  # number of classes
    model = FCNet(input_size, num_classes)

if __name__ == "__main__":
    