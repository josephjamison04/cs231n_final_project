import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
from transformers import ConvNextConfig, ConvNextForImageClassification
from torchvision import models
import torch.nn as nn

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

def load_data(args, device):

    data_dir = 'test_images'
    labels_dtype = np.int64

    label_names_path = '../data_preprocessing/label_names.csv'
    label_names = pd.read_csv(label_names_path)

    id_to_label ={}
    for i in range(label_names.shape[0]):
        id_to_label[int(label_names.iloc[i][1])] = label_names.iloc[i][0]

    print('finished retreiving label ids')

    # X_test = pd.read_pickle(data_dir + "test_data.pkl").values
    # y_test = pd.read_pickle(data_dir + "test_labels.pkl").values
    try:
        print("Loading test data...")
        X_test = []
        y_test = []
        files = [i for i in os.listdir(data_dir) if not i.startswith('.')]
        for file in files:
            im = cv2.imread(os.path.join(data_dir, file)).transpose((2, 0, 1))
            X_test.append(im)
            for key in id_to_label:
                if id_to_label[key] == file[:-4]:
                    y_test.append(key)
    except:
        print("Could not read in sample images")
    
    X_test = np.array(X_test).reshape(-1, 3, 128, 128)
    y_test = np.array(y_test).flatten().astype(labels_dtype)
    print(f"Test data size: {X_test.shape}")
    print(f"Test labels size: {y_test.shape}")

    X_test = torch.from_numpy(X_test).to(device)
    y_test = torch.from_numpy(y_test).to(device)

    channel_means = torch.tensor([103.20615017604828, 111.2633871603012, 115.82018423938752]).to(device)
    channel_sds = torch.tensor([71.08110246072079, 66.65810962849511, 67.36857566774157]).to(device)

    test_dataset = TensorDataset(X_test, y_test)
    
    loader_test = DataLoader(test_dataset, batch_size=args.batch_size)
    print('finished setting dataloaders')

    return loader_test, id_to_label, X_test


def load_model(args):
    if args.option == "ConvNext": # Update this to the best/final convnext model
        model_path = '../ConvNext/convNext-from_pretrain-10epochs-lr_0.0001-l2_0.001.pt'
        model = ConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224")
    elif args.option == "ResNet":
        model_path = "../fc_conv_transformer/resnet50-10-5e-05-cs231n.pt"
        num_classes = 100
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif args.option == "SwinTransformer":
        model_path = '../SwinTransformer/swin-from_pretrain-5epochs-lr_1e-05-l2_0.0.pt'
    elif args.option == "ViT":
        model_path = '' # Update with path to top performing ViT model .pt file
        raise NotImplementedError
    
    saved_contents = torch.load(model_path)
    print("Loaded model")
    
    model.load_state_dict(saved_contents["model"])
    model.eval()
    return model
    

def get_saliency_maps(loader, model, id_to_label, X_test, device):
    dtype = torch.float32

    model.eval()  # set model to evaluation mode
    for batch in loader:
        x,y =batch
        if args.option =='fc':
            x = x.reshape(-1,3*128*128)
        x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
        y = y.to(device=device, dtype=torch.long)
        
        x.requires_grad_()
        
        if args.option == "ConvNext":
            scores = model(x)[0]
        else:
            scores = model(x)
        
        unnormalized_loss = torch.sum(scores.gather(1, y.view(-1, 1)).squeeze())
        unnormalized_loss.backward()
        saliency, _ = torch.max(torch.abs(x.grad), dim=1)

        saliency = saliency.numpy()
        print_x = torch.permute(X_test,(0, 2, 3, 1)).detach().numpy()
        print_x = np.fliplr(print_x.reshape(-1,3)).reshape(print_x.shape)

        N = x.shape[0]
        for i in range(N):
            plt.subplot(2, N, i + 1)
            plt.imshow(print_x[i])
            plt.axis('off')
            plt.title(id_to_label[y[i].item()])
            plt.subplot(2, N, N + i + 1)
            plt.imshow(saliency[i], cmap=plt.cm.hot)
            plt.axis('off')
            plt.gcf().set_size_inches(12, 5)
        plt.savefig(f"saliency_maps_{args.option}.jpg")

    
    return saliency


def analyze_class_errors(class_acc_dict):
    TOP_K = 5
    class_list = []
    for i in class_acc_dict:
        class_list.append((i, float(class_acc_dict[i][0])/class_acc_dict[i][2], 
                                float(class_acc_dict[i][1])/class_acc_dict[i][2]))
    class_list.sort(key=lambda x: x[1])
    worst_5 = class_list[:TOP_K]
    best_5 = class_list[-TOP_K:][::-1]
    print(f"The top 5 classes with best performance are {best_5}")
    print(f"The bottom 5 classes with worst performance are {worst_5}")
    return class_list

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--use_gpu", action="store_true")

    parser.add_argument(
        "--batch_size",
        help="specify batch size. default 64",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--option",
        type=str,
        help="choose which model class you want to evaluate",
        choices=("ConvNext", "fc", "conv_transformer, SwinTransformer", "ViT", "ResNet"),
        default="ConvNext",
    )

    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

    loader_test, id_to_label, X_test = load_data(args, device)
    model = load_model(args)
    saliency = get_saliency_maps(loader_test, model, id_to_label, X_test, device)
    