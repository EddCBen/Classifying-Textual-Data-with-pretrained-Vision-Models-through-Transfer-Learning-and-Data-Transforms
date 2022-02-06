import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from torch.utils.data import TensorDataset, random_split, DataLoader
from torchvision import models
from torch import LongTensor
from torch.autograd import Variable
from torch.nn import functional
from torch.nn.functional import interpolate

dtype = torch.cuda.FloatTensor
batch_size = 32

def set_parameter_requires_grad(model, train_early=False):

    feature_extractor_early = model.features[0:5]
    if train_early == True:  # No Fine FineTuning for Early Layers
        for param in feature_extractor_early.parameters():
            param.requires_grad = True
    else:
        for param in feature_extractor_early.parameters():
            param.requires_grad = False

    return feature_extractor_early

def create_feature_extractor(CNNmodel):
    model = CNNmodel
    model_features = set_parameter_requires_grad(model, train_early=False)
    return model_features

pretrained_early = create_feature_extractor(models.alexnet(pretrained=True))

class alexnet(nn.Module):
    def __init__(self):
        global batch_size
        super().__init__()
        self.name = "alexnet"
        self.feature_extractor = pretrained_early  # Early Layers PRetrained

        self.conv_auto_encoder = nn.Sequential(
        nn.Conv2d(in_channels=192, out_channels=192, kernel_size=2),
        nn.ReLU(),
        nn.BatchNorm2d(192),
        nn.Conv2d(in_channels=192, out_channels=192, kernel_size=2),
        nn.ReLU(),
	nn.BatchNorm2d(192),
        nn.Conv2d(in_channels=192, out_channels=192, kernel_size=2),
        nn.ReLU(),
	nn.BatchNorm2d(192),
        nn.Conv2d(in_channels=192, out_channels=64, kernel_size=2),
        nn.ReLU(),
        nn.BatchNorm2d(64)
	)
        self.Adaptiveavgpool = nn.AdaptiveAvgPool2d(5)
        self.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=False),
        nn.Linear(1600, 700),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3, inplace=False),
        nn.Linear(700, 50),
        nn.ReLU(inplace=True),
        nn.Linear(50, 2)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_embedding):
        preTrained_features = self.feature_extractor(input_embedding)
        found_features = self.conv_auto_encoder(preTrained_features)
        found_features = self.Adaptiveavgpool(found_features)
        conv_shape = found_features.shape

        try:
            found_features = found_features.contiguous().view(batch_size, conv_shape[1] * conv_shape[2]*conv_shape[3])
        except Exception as e:
            found_features = found_features.contiguous().view(16, conv_shape[1] * conv_shape[2]*conv_shape[3])

        logits = self.classifier(found_features)

        return logits
