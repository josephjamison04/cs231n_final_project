import argparse
import numpy as np
import pandas as pd

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
from transformers import ConvNextConfig, ConvNextForImageClassification

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

    normalize = T.Normalize(channel_means, channel_sds)
    test_dataset = TensorDataset_transform((X_test, y_test), transform=normalize)
    
    loader_test = DataLoader(test_dataset, batch_size=args.batch_size)
    print('finished setting dataloaders')

    label_names_path = '../data_preprocessing/label_names.csv'
    label_names = pd.read_csv(label_names_path)

    id_to_label ={}
    for i in range(label_names.shape[0]):
        id_to_label[int(label_names.iloc[i][1])] = label_names.iloc[i][0]

    print("id_to_label dict:", id_to_label)
    raise ValueError

    print('finished retreiving label ids')
    return loader_test, id_to_label


def load_model(args):
    print("Loading model ...")
    if args.option == "ConvNext": # Update this to the best/final convnext model
        model_path = '../ConvNext/convNext-from_pretrain-10epochs-lr_0.0001-l2_0.0.pt'
    # model = ConvNextForImageClassification()
    saved_contents = torch.load(model_path)
    model = ConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224")
    model.load_state_dict(saved_contents["model"])
    model.eval()
    return model
    

def check_class_accuracy(loader, model, id_to_label, device):
    dtype = torch.float32

    TOP_K = 5
    # Initialize class_acc_dict with class_label:[0, 0, 0] for [t1_acc, t5_acc, num_examples]
    class_acc_dict = {}
    for i in id_to_label:
        class_acc_dict[id_to_label[i]] = [0,0,0]

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
            t5_correct = (torch.any(t5_preds == y.unsqueeze(1).expand_as(t5_preds), 1))
            t5_num_correct += t5_correct.sum()
            num_samples += t1_preds.size(0)

            # Add class accuracies to class accuaracy dictionary
            for i in range(len(y)):
                class_acc_dict[id_to_label[y[i]]] += [0, 0, 1]
                if t1_preds[i] == y[i]:
                    class_acc_dict[id_to_label[y[i]]] += [1, 1, 0]
                elif t5_correct:
                    class_acc_dict[id_to_label[y[i]]] += [0, 1, 0]

    
    t1_acc = t1_num_correct / num_samples
    t5_acc = t5_num_correct / num_samples
    print('Top-1 Val ACC: Got %d / %d correct (%.2f)' % (t1_num_correct, num_samples, 100 * t1_acc))
    print('Top-5 Val ACC: Got %d / %d correct (%.2f)' % (t5_num_correct, num_samples, 100 * t5_acc))
    
    return class_acc_dict


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
        choices=("ConvNext", "fc", "conv_transformer"),
        default="ConvNext",
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

    loader_test, id_to_label = load_data(args, device)
    model = load_model(args)
    class_acc_dict = check_class_accuracy(loader_test, model, id_to_label, device)
    print("Class_acc_dict:", class_acc_dict)