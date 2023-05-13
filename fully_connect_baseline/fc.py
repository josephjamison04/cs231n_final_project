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

from tqdm import tqdm
import time, random, numpy as np, argparse
from types import SimpleNamespace

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

def save_model(model, optimizer, args=None, config=None, filepath='fc_base.pt'):
    save_info = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict()
        # "args": args,
        # "model_config": config,
        # "system_rng": random.getstate(),
        # "numpy_rng": np.random.get_state(),
        # "torch_rng": torch.random.get_rng_state(),
        }
    
    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

def train(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    # load train and valid data
    X_train,y_train,X_valid,y_valid,_ ,_ =load_data()

    X_train = torch.from_numpy(X_train).to(device)
    X_valid = torch.from_numpy(X_valid).to(device)
    y_train = torch.from_numpy(y_train).to(device)
    y_valid = torch.from_numpy(y_valid).to(device)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_valid, y_valid)
    loader_train = DataLoader(train_dataset, batch_size=args.batch_size)
    loader_val = DataLoader(val_dataset, batch_size=args.batch_size)
    print('finished setting dataloaders')

    
    epochs =args.epochs
    input_size = 3 * 128 * 128  # image size
    num_classes = 100  # number of classes
    learning_rate = args.lr
    dtype = torch.float32
    # model = FCNet(input_size, num_classes)
    model = nn.Sequential(
        nn.Linear(input_size, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
        )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for epoch in range(epochs):
        print(f'Current, start epoch {epoch+1}')
        for batch in tqdm(loader_train):
            x ,y = batch
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            # print( f'shape of scores is {scores.shape}, while shape of y is {y.shape}')
            loss = F.cross_entropy(scores, y)/args.batch_size

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

        train_num_correct,train_num_samples =check_accuracy(loader_train,model)
        train_epoch_acc = float(train_num_correct) / train_num_samples
        
        val_num_correct,val_num_samples =check_accuracy(loader_val,model)
        val_epoch_acc = float(val_num_correct) / val_num_samples
        print('Training ACC: Got %d / %d correct (%.2f)' % (train_num_correct, train_num_samples, 100 * train_epoch_acc))
        print('Val ACC: Got %d / %d correct (%.2f)' % (val_num_correct, val_num_samples, 100 * val_epoch_acc))
        print('-'*100)

def check_accuracy(loader, model,print=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for batch in tqdm(loader):
            x,y =batch
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
    if print:
        epoch_acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * epoch_acc))
    return num_correct,num_samples


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--use_gpu", action="store_true")

    # hyper parameters
    parser.add_argument(
        "--batch_size",
        help="default 8",
        type=int,
        default=8,
    )
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument(
        "--lr",
        type=float,
        help="learning rate, default lr 1e-5",
        default=1e-5,
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    train(args)
    