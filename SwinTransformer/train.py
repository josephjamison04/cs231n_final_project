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

from transformers import Swinv2Config, Swinv2ForImageClassification, ViTImageProcessor

from tqdm import tqdm
import time, random, numpy as np, argparse, datetime
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
            X_train = X_train[:1000,:]
            y_train = y_train[:1000]
        if args.reshape:
            # print(X_train.shape)
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

def load_model(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

    model_path = 'swin-10-1e-05-cs231n.pt'
    model = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
    
    saved_contents = torch.load(model_path)
    
    model = model.to(device)
    print("Loaded model")
    
    model.load_state_dict(saved_contents["model"])
    # model.eval()
    return model


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
    # if args.norm:
        # normalize = T.Normalize(channel_means, channel_sds)

        # train_dataset = TensorDataset_transform((X_train, y_train), transform=normalize)
        # val_dataset = TensorDataset_transform((X_valid, y_valid), transform=normalize)
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_valid, y_valid)
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
    '''swinTransformer model'''
    if args.option == 'swin':
        
        # Initializing a ConvNext convnext-tiny-224 style configuration
        configuration = Swinv2Config(num_labels= 100, image_size= 128, return_dict=False, 
                                       drop_path_rate= args.dpr, patch_size= args.patch_size)

        if args.from_pretrain and args.norm:
            # Initializing a model (with pretrained weights and defined config) from the convnext-tiny-224 style configuration
            channel_means = torch.tensor([103.20615017604828, 111.2633871603012, 115.82018423938752])
            channel_sds = torch.tensor([71.08110246072079, 66.65810962849511, 67.36857566774157])
            image_processor = ViTImageProcessor(do_resize=True, size={"height" : 256, "width" : 256}, do_normalize=True, image_mean=channel_means, image_std=channel_sds)
            model = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        elif args.from_pretrain:
            image_processor = ViTImageProcessor(do_resize=True, size={"height" : 256, "width" : 256}, do_normalize=False)
            model = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        elif args.from_filepath:
            image_processor = ViTImageProcessor(do_resize=True, size={"height" : 256, "width" : 256}, do_normalize=False)
            model = load_model(args)
        else:
            # Initialize with random weights
            model = Swinv2ForImageClassification(configuration)
    
    ####################################################################################################
    ####################################################################################################
    ####################################################################################################
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay= args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[13, 20], gamma=0.1)

    # Write header of log file
    with open(args.logpath, "a+") as f:
        result = f"lr: {args.lr} \t batchsize: {args.batch_size} \t epochs: {args.epochs} \t option: {args.option}\n"
        result += f"\tdrop_path_rate: {args.dpr} \t patch size: {args.patch_size} \n"
        result += f"weight decay(L2 reg): {args.weight_decay}\n"
        result += "----------------- \n"
        f.write(result)

    model = model.to(device=device)  # move the model parameters to CPU/GPU
    max_val_acc = -1.0

    t1_train_accs = []
    t5_train_accs = []
    t1_val_accs = []
    t5_val_accs = []
    train_loss = []

    for epoch in range(epochs):
        print(f'Current, start epoch {epoch+1}')
        # Update the learning rate at the start of each epoch
        for batch in tqdm(loader_train):
            x ,y = batch
            
            if args.from_pretrain or args.from_filepath:
                x = image_processor(x, return_tensors="pt")
                x = x["pixel_values"]
                
            model.train()  # put model to training mode

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
        
        if args.from_pretrain or args.from_filepath:
            t1_train_num_correct,train_num_samples, t5_train_num_correct =check_accuracy(loader_train,model, image_processor=image_processor)
            t1_val_num_correct,val_num_samples, t5_val_num_correct =check_accuracy(loader_val, model, image_processor=image_processor)
        else:
            t1_train_num_correct,train_num_samples, t5_train_num_correct =check_accuracy(loader_train,model)
            t1_val_num_correct,val_num_samples, t5_val_num_correct =check_accuracy(loader_val,model)
        t1_train_epoch_acc = float(t1_train_num_correct) / train_num_samples
        t5_train_epoch_acc = float(t5_train_num_correct) / train_num_samples
        
        t1_val_epoch_acc = float(t1_val_num_correct) / val_num_samples 
        t5_val_epoch_acc = float(t5_val_num_correct) / val_num_samples 
        
        if t1_val_epoch_acc > max_val_acc:
            max_val_acc = t1_val_epoch_acc
            save_model(model,optimizer,args=args,max_val_acc=max_val_acc) # should we update this to save t5_acc too?
        print('Top-1 Training ACC: Got %d / %d correct (%.2f)' % (t1_train_num_correct, train_num_samples, 100 * t1_train_epoch_acc))
        print('Top-5 Training ACC: Got %d / %d correct (%.2f)' % (t5_train_num_correct, train_num_samples, 100 * t5_train_epoch_acc))
        print('Top-1 Val ACC: Got %d / %d correct (%.2f)' % (t1_val_num_correct, val_num_samples, 100 * t1_val_epoch_acc))
        print('Top-5 Val ACC: Got %d / %d correct (%.2f)' % (t5_val_num_correct, val_num_samples, 100 * t5_val_epoch_acc))
        

         # Append current epoch results to log file
        with open(args.logpath, "a+") as f:
            epoch_str = f"Epoch: {epoch + 1} \n"
            epoch_str += f"Top-1 Training ACC: {100*t1_train_epoch_acc} \n"
            epoch_str += f"Top-5 Training ACC: {100*t5_train_epoch_acc} \n"
            epoch_str += f"Top-1 Val ACC: {100*t1_val_epoch_acc} \n"
            epoch_str += f"Top-5 Val ACC: {100*t5_val_epoch_acc} \n"
            epoch_str += f"Training Loss: {loss.item()} \n"
            epoch_str += "----------------- \n"
            f.write(epoch_str)
            print(f"Write results to {args.logpath}")
        print('-'*100)

        t1_train_accs.append(t1_train_epoch_acc)
        t5_train_accs.append(t5_train_epoch_acc)
        t1_val_accs.append(t1_val_epoch_acc)
        t5_val_accs.append(t5_val_epoch_acc)
        train_loss.append(loss.item())
    
    return max_val_acc

def check_accuracy(loader, model, image_processor=None):
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
            
            if args.from_pretrain or args.from_filepath:
                # x = image_processor([i for i in x], return_tensors="pt")
                x = image_processor(x, return_tensors="pt")
                # print(x["pixel_values"].shape)
                x = x["pixel_values"]
            
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)[0]
            _, t1_preds = scores.max(1)
            t1_num_correct += (t1_preds == y).sum()
            # Calculate top_5 accuracy in addition to top-1
            t5_preds = torch.argsort(-scores, dim=1)[:, :TOP_K]
            t5_num_correct += (torch.any(t5_preds == y.unsqueeze(1).expand_as(t5_preds), 1)).sum()

            num_samples += t1_preds.size(0)
    
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
    parser.add_argument("--from_filepath", action="store_true") 
    

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
        help="swin: swinTransformer model architecture",
        choices=("swin"),
        default="swin",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # args = get_args()
    
    # seed_everything(args.seed)
    
    # ####################################################################################
    # # Hyperparameter grid search

    # lrs = [1e-5, 1e-4]
    # drop_path_rate = [0.0] # Drop rate for stochastic depth (i.e., randomly drops 
    #                             # entire Resblocks during training -> additional regularization)
    # weight_decay_factors = [1e-8, 0.0]
    
    # hpo_loops = len(lrs)*len(drop_path_rate)*len(weight_decay_factors)

    # print(f"HPO loop will train {hpo_loops} models for {args.epochs} epochs each.")
    # print(f"Logs will be stored in SwinTransformers/logs folder")
    # print('-'*100)
    # ####################################################################################

    # hpo_loop_counter = 1
    # best_t1_val_acc = -1.0
    # for lr in lrs:
    #     for dpr in drop_path_rate:
    #         for wd in weight_decay_factors:
    #             now = datetime.datetime.now()

    #             args.lr = lr
    #             args.dpr = dpr
    #             # args.num_stages = ns
    #             # args.hidden_sizes = [96, 192, 384] if ns == 3 else [96, 192, 384, 768]
    #             args.patch_size = 4
    #             args.weight_decay = wd
                    

    #             print(f"Now training model number {hpo_loop_counter} of {hpo_loops}...")
    #             if args.from_pretrain:
    #                 args.filepath = f"{args.option}-from_pretrain-{args.epochs}epochs-lr_{args.lr}-l2_{args.weight_decay}.pt"  # save path
    #                 args.logpath = f"logs/{args.option}-from_pretrain-{args.epochs}epochs-lr_{args.lr}-l2_{args.weight_decay}-{now.hour}_{now.minute}_{now.second}.txt"  # save path
    #             else:
    #                 args.filepath = f"{args.option}-{args.epochs}epochs-lr_{args.lr}-dpr_{args.dpr}.pt"  # save path
    #                 args.logpath = f"logs/{args.option}-{args.epochs}epochs-lr_{args.lr}_-dpr_{args.dpr}-{now.hour}_{now.minute}_{now.second}.txt"  # save path

    #             t1_val_acc = train(args)
    #             hpo_loop_counter += 1

    #             if t1_val_acc > best_t1_val_acc:
    #                 best_model_path = args.filepath
    #                 best_model_log = args.logpath
    #                 best_t1_val_acc = t1_val_acc
                
    # # Write results file
    # now2 = datetime.datetime.now()
    # result_path = f"logs/RESULT_FILE-{now2.month}m_{now2.day}d_{now2.hour}h_{now2.minute}m.txt"
    # with open(result_path, "a+") as f:
        
    #     f.write(f"Best top-1 validation accuracy out of {hpo_loops} models was {100* best_t1_val_acc}. \
    #             \nThis occurred in model {best_model_path}, \nwhich was logged in {best_model_log}")
    
    args = get_args()
    args.dpr = 0
    args.patch_size = 4
    args.weight_decay = 0
    args.filepath = f"{args.option}-{args.epochs}-{args.lr}-cs231n.pt"  # save path
    now = datetime.datetime.now()
    args.logpath = f"logs/{args.option}-{args.epochs}-{args.lr}_{now.hour}_{now.minute}_{now.second}.txt"  # save path
    seed_everything(args.seed)
    train(args)