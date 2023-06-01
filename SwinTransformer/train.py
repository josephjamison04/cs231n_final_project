from multiprocessing.spawn import _main
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
from torch.utils.data import sampler
from torchvision import models
import torchvision.transforms as T

from transformers import ConvNextConfig, ConvNextForImageClassification

from tqdm import tqdm
import time, random, numpy as np, argparse
from types import SimpleNamespace

# from conv_transformer import ImageTransformer, VisionTransformer


def load_data(args,train =True,valid = True,test = False):
    # if args.norm == True:
    #     data_dir = '/home/ubuntu/CS231N/data/Normalized-datasets/'
    # else:
    #     data_dir = '/home/ubuntu/CS231N/data/split-datasets/'
    data_dir = '/home/ubuntu/CS231N/data/split-datasets/'
    labels_dtype = np.int64
    # 
    # data_dir = "../../data/"
    X_train,y_train,X_valid,y_valid,X_test,y_test = None, None, None, None, None, None
    
    
    if train:
        X_train = pd.read_pickle(data_dir + "train_data.pkl").values
        y_train = pd.read_pickle(data_dir + "train_labels.pkl").values
        y_train = y_train.flatten().astype(labels_dtype)
        if args.small_data:
            X_train = X_train[:10000,:]
            y_train = y_train[:10000]
        if args.reshape:
            print(X_train.shape)
            X_train = X_train.reshape(-1, 3, 128, 128)
        print('Training data shape: ', X_train.shape)
        print('Training labels shape: ', y_train.shape)
    if valid:
        X_valid = pd.read_pickle(data_dir + "valid_data.pkl").values
        y_valid = pd.read_pickle(data_dir + "valid_labels.pkl").values
        y_valid = y_valid.flatten().astype(labels_dtype)
        if args.small_data:
            X_valid = X_valid[:4000,:]
            y_valid = y_valid[:4000]
        if args.reshape:
            X_valid = X_valid.reshape(-1,3,128,128)
        print('Validation data shape: ', X_valid.shape)
        print('Validation labels shape: ', y_valid.shape)
    if test:
        X_test = pd.read_pickle(data_dir + "test_data.pkl").values
        y_test = pd.read_pickle(data_dir + "test_labels.pkl").values
        y_test = y_test.flatten().astype(labels_dtype)
        if args.reshape:
            X_test = X_test.reshape(-1,3,128,128)
        print('Test data shape: ', X_test.shape)
        print('Test labels shape: ', y_test.shape)
    # As a sanity check, we print out the size of the data we output.
    # if didn;t load the data, it is None
    return X_train,y_train,X_valid,y_valid,X_test,y_test



class TensorDataset_transform(Dataset):
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        # Convert to float32
        x = x.float()
        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def save_model(model, optimizer, args=None, config=None,max_val_acc = None):
    filepath=args.filepath
    save_info = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "max_val_acc": max_val_acc,
        # "model_config": config,
        # "system_rng": random.getstate(),
        # "numpy_rng": np.random.get_state(),
        # "torch_rng": torch.random.get_rng_state(),
        }
    
    torch.save(save_info, filepath)
    print(f"Saving model to {filepath}...")

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
    
# We need to wrap `flatten` function in a module in order to stack it
# in nn.Sequential
class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)



def train(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    # load train and valid data
    # if args.option =='fc':
    #     args.reshape = False
    # else:
    args.reshape = True

    X_train,y_train,X_valid,y_valid,_ ,_ =load_data(args)
    X_train = torch.from_numpy(X_train).to(device)
    X_valid = torch.from_numpy(X_valid).to(device)
    y_train = torch.from_numpy(y_train).to(device)
    y_valid = torch.from_numpy(y_valid).to(device)

    # if args.norm:
    # Assuming channel_means and channel_sds are 1D tensors
    #T is transform imported from torch vision
    channel_means = torch.tensor([103.20615017604828, 111.2633871603012, 115.82018423938752]).to(device)
    channel_sds = torch.tensor([71.08110246072079, 66.65810962849511, 67.36857566774157]).to(device)

    if args.norm:
        normalize = T.Normalize(channel_means, channel_sds)

        train_dataset = TensorDataset_transform((X_train, y_train), transform=normalize)
        val_dataset = TensorDataset_transform((X_valid, y_valid), transform=normalize)
    else:
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_valid, y_valid)
    loader_train = DataLoader(train_dataset, batch_size=args.batch_size)
    loader_val = DataLoader(val_dataset, batch_size=args.batch_size)
    print('finished setting dataloaders')

    
    epochs =args.epochs
    learning_rate = args.lr
    dtype = torch.float32
    ####################################################################################################
    ####################################################################################################
    ####################################################################################################
    '''ConvNext model'''
    if args.option == 'convNext':
        
        # Initializing a ConvNext convnext-tiny-224 style configuration
        configuration = ConvNextConfig(num_labels= 100, image_size= 128, return_dict=False)

        if args.from_pretrain:
            # Initializing a model (with random weights) from the convnext-tiny-224 style configuration
            model = ConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224")
        else:
            # Initialize with random weights
            model = ConvNextForImageClassification(configuration) 
    

    ####################################################################################################
    ####################################################################################################
    ####################################################################################################
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[13, 20], gamma=0.1)


    model = model.to(device=device)  # move the model parameters to CPU/GPU
    max_val_acc = -1.0
    for epoch in range(epochs):
        print(f'Current, start epoch {epoch+1}')
        # Update the learning rate at the start of each epoch
        for batch in tqdm(loader_train):
            x ,y = batch
            model.train()  # put model to training mode
            if args.option =='fc':
                x = x.reshape(-1,128*128*3)
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
        # print(x.size())
            scores = model(x)[0]
            # print( f'shape of scores is {len(scores)}, while shape of y is {y.shape}')
            
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
        lr_scheduler.step()
        
        t1_train_num_correct,train_num_samples, t5_train_num_correct =check_accuracy(loader_train,model,args)
        t1_train_epoch_acc = float(t1_train_num_correct) / train_num_samples
        t5_train_epoch_acc = float(t5_train_num_correct) / train_num_samples
        t1_val_num_correct,val_num_samples, t5_val_num_correct =check_accuracy(loader_val,model,args)
        t1_val_epoch_acc = float(t1_val_num_correct) / val_num_samples 
        t5_val_epoch_acc = float(t5_val_num_correct) / val_num_samples 
        
        if t1_val_epoch_acc > max_val_acc:
            max_val_acc = t1_val_epoch_acc
            save_model(model,optimizer,args=args,max_val_acc=max_val_acc) # should we update this to save t5_acc too?
        print('Top-1 Training ACC: Got %d / %d correct (%.2f)' % (t1_train_num_correct, train_num_samples, 100 * t1_train_epoch_acc))
        print('Top-5 Training ACC: Got %d / %d correct (%.2f)' % (t5_train_num_correct, train_num_samples, 100 * t5_train_epoch_acc))
        print('Top-1 Val ACC: Got %d / %d correct (%.2f)' % (t1_val_num_correct, val_num_samples, 100 * t1_val_epoch_acc))
        print('Top-5 Val ACC: Got %d / %d correct (%.2f)' % (t5_val_num_correct, val_num_samples, 100 * t5_val_epoch_acc))
        print('-'*100)

def check_accuracy(loader, model, print_acc=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    TOP_K = 5

    t1_num_correct = 0
    t5_num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for batch in tqdm(loader):
            x,y =batch
            if args.option =='fc':
                x = x.reshape(-1,3*128*128)
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)[0]
            _, t1_preds = scores.max(1)
            t1_num_correct += (t1_preds == y).sum()
            # Calculate top_5 accuracy in addition to top-1
            t5_preds = torch.argsort(-scores, dim=1)[:, :TOP_K]
            t5_num_correct += (torch.any(t5_preds == y.unsqueeze(1).expand_as(t5_preds), 1)).sum()

            num_samples += t1_preds.size(0)
    # if print_acc:
    #     t1_epoch_acc = float(t1_num_correct) / num_samples
    #     print('Top-1: Got %d / %d correct (%.2f)' % (t1_num_correct, num_samples, 100 * t1_epoch_acc))
    #     t5_epoch_acc = float(t5_num_correct) / num_samples
    #     print('Top-5: Got %d / %d correct (%.2f)' % (t5_num_correct, num_samples, 100 * t5_epoch_acc))
    return t1_num_correct,num_samples, t5_num_correct


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
    parser.add_argument("--small_data", action="store_true") 
    parser.add_argument("--norm", action="store_true")
    parser.add_argument("--from_pretrain", action="store_true") 
    

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
    parser.add_argument(
        "--option",
        type=str,
        help="convNext: convNext-style model architecture",
        choices=("convNext"),
        default="fc",
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.from_pretrain:
        args.filepath = f"{args.option}-from_pretrain-{args.epochs}epochs-{args.lr}-cs231n.pt"  # save path
    else:
        args.filepath = f"{args.option}-{args.epochs}epochs-{args.lr}-cs231n.pt"  # save path
    seed_everything(args.seed)
    train(args)
    