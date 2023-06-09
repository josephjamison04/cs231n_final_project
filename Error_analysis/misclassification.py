import argparse
import numpy as np
import pandas as pd

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
from transformers import ConvNextConfig, ConvNextForImageClassification, Swinv2ForImageClassification, ViTImageProcessor
from torchvision import models
import torch.nn as nn

from tqdm import tqdm

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

    data_dir = '/home/ubuntu/CS231N/data/split-datasets/'
    labels_dtype = np.int64
    print("Loading test data...")
    X_test = pd.read_pickle(data_dir + "test_data.pkl").values
    y_test = pd.read_pickle(data_dir + "test_labels.pkl").values

    X_test = X_test.reshape(-1, 3, 128, 128)
    y_test = y_test.flatten().astype(labels_dtype)
    print(f"Test data size: {X_test.shape}")
    print(f"Test labels size: {y_test.shape}")

    X_test = torch.from_numpy(X_test).to(device)
    y_test = torch.from_numpy(y_test).to(device)

    channel_means = torch.tensor([103.20615017604828, 111.2633871603012, 115.82018423938752]).to(device)
    channel_sds = torch.tensor([71.08110246072079, 66.65810962849511, 67.36857566774157]).to(device)

    if args.norm:
        normalize = T.Normalize(channel_means, channel_sds)
        test_dataset = TensorDataset_transform((X_test, y_test), transform=normalize)
    elif args.option == 'ViT':
        transform = T.Compose([
                    T.Resize((224, 224),antialias =True),  # Resize to 224x224
                    # Add any other transforms you need
                ])
        test_dataset = TensorDataset_transform((X_test, y_test), transform=transform)
    else:
        test_dataset = TensorDataset(X_test, y_test)
    
    loader_test = DataLoader(test_dataset, batch_size=args.batch_size)
    print('finished setting dataloaders')

    label_names_path = '../data_preprocessing/label_names.csv'
    label_names = pd.read_csv(label_names_path)

    id_to_label ={}
    for i in range(label_names.shape[0]):
        id_to_label[int(label_names.iloc[i][1])] = label_names.iloc[i][0]

    print('finished retreiving label ids')
    return loader_test, id_to_label


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
        # model_path = '../SwinTransformer/swin-from_pretrain-5epochs-lr_1e-05-l2_0.0.pt'
        model_path = '../SwinTransformer/swin-12-1e-05-cs231n.pt'
        model = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
    elif args.option == "ViT":
        model_path = "../fc_conv_transformer/vit_b16-from_pretrain-5epochs-lr_3e-05-l2_1e-08.pt"
        num_classes = 100
        model = models.vit_b_16(weights='IMAGENET1K_V1')
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, num_classes)
    
    saved_contents = torch.load(model_path)
    
    model = model.to(device)
    print("Loaded model")
    
    model.load_state_dict(saved_contents["model"])
    model.eval()
    return model
    

def check_class_accuracy(loader, model, id_to_label, device):
    dtype = torch.float32

    TOP_K = 5
    # Initialize class_acc_dict with class_label:[0, 0, 0] for [t1_acc, t5_acc, num_examples]
    class_acc_dict = {}
    for i in id_to_label:
        class_acc_dict[id_to_label[i]] = np.array([0,0,0])

    t1_num_correct = 0
    t5_num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for batch in tqdm(loader):
            x,y =batch
            if args.option == 'SwinTransformer':
                image_processor = ViTImageProcessor(do_resize=True, size={"height" : 256, "width" : 256}, do_normalize=False)
                x = image_processor(x, return_tensors="pt")
                x = x["pixel_values"]
            if args.option =='fc':
                x = x.reshape(-1,3*128*128)
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            if args.option == "ConvNext" or args.option == "SwinTransformer":
                scores = model(x)[0]
            else:
                scores = model(x)
            _, t1_preds = scores.max(1)
            t1_num_correct += (t1_preds == y).sum()
            # Calculate top_5 accuracy in addition to top-1
            t5_preds = torch.argsort(-scores, dim=1)[:, :TOP_K]
            t5_correct = (torch.any(t5_preds == y.unsqueeze(1).expand_as(t5_preds), 1))
            
            t5_num_correct += t5_correct.sum()
            num_samples += t1_preds.size(0)

            # Add class accuracies to class accuaracy dictionary
            for i in range(len(y)):
                if t1_preds[i] == y[i]:
                    class_acc_dict[id_to_label[y[i].item()]] += np.array([1, 1, 1])
                elif t5_correct[i].item():
                    class_acc_dict[id_to_label[y[i].item()]] += np.array([0, 1, 1])
                else:
                    class_acc_dict[id_to_label[y[i].item()]] += np.array([0, 0, 1])

    
    t1_acc = t1_num_correct / num_samples
    t5_acc = t5_num_correct / num_samples
    print('Top-1 Test ACC: Got %d / %d correct (%.2f)' % (t1_num_correct, num_samples, 100 * t1_acc))
    print('Top-5 Test ACC: Got %d / %d correct (%.2f)' % (t5_num_correct, num_samples, 100 * t5_acc))
    
    return class_acc_dict, t1_acc.item(), t5_acc.item()


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
    parser.add_argument("--norm", action="store_true")

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
        choices=("ConvNext", "fc", "conv_transformer", "SwinTransformer", "ViT", "ResNet"),
        default="ConvNext",
    )

    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

    loader_test, id_to_label = load_data(args, device)
    model = load_model(args)
    class_acc_dict, t1_acc, t5_acc = check_class_accuracy(loader_test, model, id_to_label, device)
    sorted_class_list = analyze_class_errors(class_acc_dict)
    sorted_class_list.append(("Total", t1_acc, t5_acc))
    sorted_class_df = pd.DataFrame(np.array(sorted_class_list))
    
    save_path = f"sorted_class_accuracies_{args.option}.csv"
    sorted_class_df.to_csv(save_path)
