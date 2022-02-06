
import torch
import torch.nn as nn
from torch.nn import functional
from torch.nn.functional import interpolate
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import TensorDataset, random_split, DataLoader
from torchvision import models
from torch import LongTensor
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import numpy as np

dtype = torch.cuda.FloatTensor
batch_size = 32
def shuffleNet_frozen():
    shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
    feature_extractor = nn.Sequential(shufflenet.conv1,
                                      shufflenet.maxpool,
                                      shufflenet.stage2)

    #Freezing the Gradients NOT REALLY HERE We'll See ...
    for param in feature_extractor.parameters():
        param.requires_grad = True

    del shufflenet
    return feature_extractor

class shufflenet(nn.Module):
    def __init__(self):
        global batch_size
        super().__init__()
        self.name = "shufflenet"
        self.feature_extractor = shuffleNet_frozen()
        self.conv_auto_encoder = nn.Sequential(
                        nn.Conv2d(in_channels=116, out_channels=192, kernel_size=4),
                        nn.ReLU(),
                        nn.BatchNorm2d(192),
                        nn.Dropout(p=0.05),
                        nn.Conv2d(in_channels=192, out_channels=128, kernel_size=4),
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.Dropout(p=0.05),
                        nn.Conv2d(in_channels=128, out_channels=64, kernel_size=4),
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.Dropout(p=0.05),
                        nn.Conv2d(in_channels=64, out_channels=32, kernel_size=4),
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                                        )
        self.adaptavgpool = nn.AdaptiveAvgPool2d(10)

        self.classifier = nn.Sequential(
                        nn.Linear(3200,1600),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.3, inplace = False),
                        nn.Linear(1600,700),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.3, inplace = False),
                        nn.Linear(700,50),
                        nn.ReLU(inplace=True),
                        nn.Linear(50,2)
                        )

    def forward(self, input_embedding):
        from_pretrained = self.feature_extractor(input_embedding)
        from_init = self.conv_auto_encoder(from_pretrained)
        pooled_features = self.adaptavgpool(from_init)

        pooled_features = pooled_features.contiguous().view(pooled_features.shape[0],-1)

        logits = self.classifier(pooled_features)

        return logits
