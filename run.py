import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
from PIL import Image
import datetime
from tqdm import tqdm
import sys

import torch
from torch import nn
import torch.nn.functional as func
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet152
#from torchvision.models import ResNet152_Weights
import torchvision
from torchvision.io import read_image
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim

from models import VGG, ResNet
#from train import train1Epoch, test1Epoch
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
import seaborn as sns


import os
import cv2
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from scipy.stats import pearsonr
import random
from src.features.build_features import files2df, PreprocessedImageDataset, Loader
from src.models.train_model import train1Epoch, trainAndSave
from src.models.predict_model import test1Epoch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LR = 0.0001
EPOCHS = 5




#####TODO fix paths 
#WORKDIR = os.getcwd()
#DATADIR = f'/home/{username}/teams/dsc-180a---a14-[88137]/'

WORKING_DIR = os.getcwd()
test_path = os.path.join(WORKING_DIR, 'test', 'test_data.csv')
data = pd.read_csv(test_path)

valid_path = os.path.join(WORKING_DIR, 'test', 'val_data.csv')
val_data = pd.read_csv(valid_path)



def run_all(df_val, df_train=None):
    ### 1) if train != None: use train set (on train mode)
    # 2) save model
    # 3) use model on val set (on eval mode)
    # 4) output predictions/visualizations
    # Otherwise, use pretrained model on the test set (on eval mode, ~100 rows), output predictions/visualizations

    valid_set = PreprocessedImageDataset(df=df_val)
    valid_loader = Loader(valid_set, mode="eval")   
    resnet = resnet152(pretrained=True)
    #resnet = resnet152(weights=ResNet152_Weights.DEFAULT, progress=True)
    resnet.fc = nn.Linear(in_features=2048, out_features=1, bias=True)
    resnet.to(DEVICE)
    optimizer = optim.Adam(resnet.parameters(), lr=LR)

    checkpoint = torch.load("resnet152.pt")
    resnet.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    loss_fn = nn.L1Loss().to(DEVICE)
    print("loaded pretrained model")
    checkpoint = torch.load("resnet152.pt")
    
    resnet.eval()
    for param in resnet.parameters():
        param.requires_grad = False
    with torch.no_grad():
        test_loss = test1Epoch(0, resnet, loss_fn, valid_loader)
        print(f"Overall Test Loss: {test_loss}")

    return None



def main(targets):

    if targets[0] == "test":
        print("Testing pretrained model on test set...")
        ### if test, use pretrained model on the test set (on eval mode, ~100 rows), output predictions/visualizations
        df_test = pd.read_csv(test_path, index_col=0)
        data = df_test
        data.drop(columns=['cardio_edema','bmi','cr','PNA','AcuteHF'], inplace=True)
        data['bnpp_log'] = data.bnpp.apply(lambda x: np.log10(x))
        data['edema'] = data['bnpp']>=400
        data.bnpp_log = data.bnpp_log.astype('float32')
        data = data.to_numpy()
        data = data.head(5)
        print(data.head())
        run_all(data)

   

if __name__ == "__main__":
    ### Run with `python run.py test` or `python run.py train`
    ### target: train or test
    print("Running run.py...")
    print(DEVICE)
    targets = sys.argv[1:]
    main(targets)

