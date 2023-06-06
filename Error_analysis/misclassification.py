import argparse
import numpy as np
import os
import pandas as pd
import torch

from transformers import ConvNextConfig, ConvNextForImageClassification

def load_data():

    data_dir = '/home/ubuntu/CS231N/data/split-datasets/'
    labels_dtype = np.int64
    
    X_test = pd.read_pickle(data_dir + "test_data.pkl").values
    y_test = pd.read_pickle(data_dir + "test_labels.pkl").values

    X_test = X_test.reshape(-1, 3, 128, 128)
    y_test = y_test.flatten().astype(labels_dtype)

    return X_test, y_test

def load_model():
    if args.option == "ConvNext":
        model_dir = '/ConvNext/'
    model = ConvNextForImageClassification()
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    

def get_args():
    parser = argparse.ArgumentParser()
    
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
    X_test, y_test = load_data()
    model = load_model()