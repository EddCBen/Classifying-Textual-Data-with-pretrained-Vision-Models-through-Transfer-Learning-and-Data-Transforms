""""
Training Script for Models in ./models
import model according to : from models.model_name import model_name
initialize model in create_model() : line 56

"""
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from torch.nn import functional
from torch.nn.functional import interpolate
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
from models.alexnet import alexnet 
import sys

dtype = torch.cuda.FloatTensor
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda:0")
data_path = Path("./data/imdb_image")
labels_path = Path("./data")


data = torch.load(data_path / "imdb_images-6to12.pt",map_location='cuda:0')
labels = torch.load(labels_path / "labels2D.pt")

batch_size = 32

def create_dataset(input_embedding, input_labels):
	global batch_size
	dataset = TensorDataset(input_embedding.type(dtype).cuda(),
			            input_labels.type(dtype).cuda())
	#Splits
	train_size = int(0.8 * len(dataset))
	val_size = int(0.2 * len(dataset))

	train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
	train_loader = DataLoader(dataset=train_dataset, batch_size = batch_size,shuffle=True)
	val_loader = DataLoader(dataset = val_dataset, batch_size = batch_size,shuffle=True)
	return train_loader, val_loader

train_loader, val_loader = create_dataset(data, labels)

del data, labels 

def initialize_parameters(m) -> None:
	if isinstance(m, nn.Linear):
		m.weight.data = init.xavier_uniform_(m.weight.data,
						    gain=nn.init.calculate_gain('relu'))
	if isinstance(m, nn.Conv2d):
		m.weight.data = init.xavier_normal_(m.weight.data)

def create_model():
    model = alexnet()
    model.conv_auto_encoder.apply(initialize_parameters)
    model.classifier.apply(initialize_parameters)
    model = model.cuda()
    return model
model = create_model()

learning_rates = {'alexnet': {"CAE LR": 0.00001, "LC LR":0.0005},
                    'resnet': {"CAE LR": 0.00005, "LC LR":0.0001},
                    'resnext': {"CAE LR": 0.00005, "LC LR":0.001 },
                    'shufflenet': {"CAE LR": 0.0005, "LC LR":0.001},
                    'vgg16': {"CAE LR": 0.00005, "LC LR":0.001}
                }

optimizer = optim.Adam([{'params': model.feature_extractor.parameters()},
    {'params': model.conv_auto_encoder.parameters(), 'lr' : learning_rates[str(model.name)]['CAE LR']},
    {'params': model.classifier.parameters(), 'lr' : learning_rates[str(model.name)]['LC LR']}],
    lr=0.0)

criterion = nn.CrossEntropyLoss()

def opt_model()-> None:
    optimizer.zero_grad()
    loss.backward() #Global Var 
    optimizer.step()

upsample = nn.Upsample(scale_factor = 3, mode = "nearest")              

"""
scale_image_batch : A Function to resize the input images to fit
                    The input layer of the pretrained model, and adjust
                    the shape according to : [B,W,H,C] ----> [B,C,W,H]
"""
def scale_image_batch(image_batch) -> torch.Tensor:
    a = torch.movedim(image_batch, -1,1)
    scaled_batch = upsample(a)
    return scaled_batch.cuda()

"""
z_normalize : applies Z-Normalization (Described in the paper) to adjust 
                image contrast by moving it to a clearer pixel space.

"""

def z_normalize(input_tensor) -> torch.Tensor:
    mean = input_tensor.mean()
    std = input_tensor.std()
    up = torch.sub(input_tensor, mean)
    down = torch.add(std**2, 0.0001**2)
    return torch.div(up,torch.sqrt(down))


#Relevant Training settings and results
train_total = 0
train_correct = 0
val_total = 0
val_correct = 0
train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []
Epochs = 15
iteration  = 0
step = 0
val_step = 0

for epoch in range(Epochs):
    for i, (input_batch, label) in enumerate(train_loader):
        model.train()
        input_batch = scale_image_batch(input_batch)
        input_batch = z_normalize(input_batch)
        label = label.contiguous().view(batch_size, 2)
        label = torch.max(label.long().to(device),1)[1]
        output = model(input_batch)
        _, predicted = torch.max(output.data, 1)
        train_total += label.size(0)
        train_correct += (predicted == label).sum().item()
        train_accuracy = train_correct/train_total
        train_accuracies.append(train_accuracy)
        loss = criterion(output, label)
        train_losses.append(loss.item())
        step += 1
        opt_model()
        iteration += 1

        if iteration %50 == 0:
            model.eval()
            for j,(val_input_batch, val_label) in enumerate(val_loader):
                val_input_batch = scale_image_batch(val_input_batch)
                val_input_batch = z_normalize(val_input_batch)
                try:
                    val_label = val_label.contiguous().view(batch_size,2)
                except Exception as e:
                    val_label = val_label.contiguous().view(16,2)
                val_label = torch.max(val_label.long().to(device),1)[1]
                val_output = model(val_input_batch)
                _, val_predicted = torch.max(val_output.data, 1)
                val_total += val_label.size(0)
                val_correct += (val_predicted == val_label).sum().item()
                val_accuracy = val_correct/val_total
                val_accuracies.append(val_accuracy)
                val_loss = criterion(val_output, val_label)
                val_step += 1
                val_losses.append(val_loss.item())

            try:
                print(f"""    epoch: {epoch + 1}
                \t     Train Loss : {np.mean(train_losses)}
                \t     Validation Loss : {np.mean(val_losses)}
                \t     Training Accuracy : {train_accuracy}
                \t     Validation Accuracy : {val_accuracy}
                """)

            except Exception as e:
                print(e)
                continue

#Saving Validation Losses and Accuracies for Polotting
torch.save(val_losses, str(model.name)+"_val_losses.pt")
torch.save(val_accuracies , str(model.name)+"_val_accs.pt")

