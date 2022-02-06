
"""
Python Script for Generating images for BERT-representations
of the IMDB Dataset using pyDeepInsight from the paper:
     DeepInsight: A methodology to transform a non-image data to an image 
     for convolution neural network architecture 
    Paper : https://www.nature.com/articles/s41598-019-47765-6
    GitHub Repository : https://github.com/alok-ai-lab/DeepInsight
"""
from pyDeepInsight import ImageTransformer, LogScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import torch
import os
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
os.environ['KMP_DUPLICATE_LIB_OK']='True'

data_path = Path('../')
data = torch.load(data_path / "IMDB_cls_last6layers.pt")
data = data[:,:-1,:]
data = np.array(data).reshape(50000,-1)
#Data normalization
ln = LogScaler()
X_train = ln.fit_transform(data)

del data
tsne = TSNE(
    n_components=2,
    random_state=1701,
    n_jobs=-1)

it = ImageTransformer(
    feature_extractor=tsne,
    pixels=50)

X_train_img = it.fit_transform(X_train)

dInsightImages = torch.from_numpy(X_train_img)
torch.save(dInsightImages, "Ready_images-six2elev.pt") 
