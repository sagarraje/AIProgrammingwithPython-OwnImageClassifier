# Imports here
import PIL
import pandas as pd
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.utils.data 
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import time
import json
import argparse
from pathlib import Path

args_val = argparse.ArgumentParser (description = "Train")
args_val.add_argument ('data_dir', help = 'Provide data directory. Mandatory argument', type = str)
args_val.add_argument ('--save_dir', help = 'Provide saving directory. Optional argument', type = str)
args_val.add_argument('--gpu', action="store",default='cpu', help='Use GPU + Cuda for calculations')
args_val.add_argument ('--pre_train', default='vgg16', help = 'Provide pre-trained models', type = str)
args_val.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
args_val.add_argument('--hidden_units', default=4096, type=int, help='number of neurons in hidden layer')
args_val.add_argument('--epochs', default=5, type=int, help='number of epochs for training')
args = args_val.parse_args ()

pre_train = args.pre_train
data_dir = args.data_dir
lr = args.learning_rate
hid = args.hidden_units
gpu = args.gpu
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# TODO: Define your transforms for the training, validation, and testing sets

#define variables to calculated from the ImageNet images.
means_val = [0.485, 0.456, 0.406]
standard_deviations = [0.229, 0.224, 0.225]
shapeval = 224 
rotationval=30
batch=64


transforms_datatrain = transforms.Compose ([transforms.RandomRotation (rotationval),
                                             transforms.RandomResizedCrop (shapeval),
                                             transforms.RandomHorizontalFlip (),
                                             transforms.ToTensor (),
                                             transforms.Normalize (means_val,standard_deviations)
                                            ])

transforms_datavalid = transforms.Compose ([transforms.Resize (255),
                                             transforms.CenterCrop (shapeval),
                                             transforms.ToTensor (),
                                             transforms.Normalize (means_val,standard_deviations)
                                            ])

transforms_datatest = transforms.Compose ([transforms.Resize (255),
                                             transforms.CenterCrop (shapeval),
                                             transforms.ToTensor (),
                                             transforms.Normalize (means_val,standard_deviations)
                                            ])

# TODO: Load the datasets with ImageFolder
datasets_imgtrain = datasets.ImageFolder (train_dir, transform = transforms_datatrain)
datasets_imgvalid = datasets.ImageFolder (valid_dir, transform = transforms_datavalid)
datasets_imgtest = datasets.ImageFolder (test_dir, transform = transforms_datatest)


# TODO: Using the image datasets and the trainforms, define the dataloaders
#torch.utils.data.DataLoader. It represents a Python iterable over a dataset, with support for
dataloader_train = torch.utils.data.DataLoader(datasets_imgtrain, batch_size = batch, shuffle = True)
dataloader_valid = torch.utils.data.DataLoader(datasets_imgvalid, batch_size = batch, shuffle = True)
dataloader_test = torch.utils.data.DataLoader(datasets_imgtest, batch_size = batch, shuffle = True)



inputs, labels = next(iter(dataloader_valid))
inputs [0,:]
inputs.size ()
#plt.imshow (inputs) #helper.imshow(image[0,:]);

datasets_imgtrain.class_to_idx

#open cat_to_name.json and load in variable 
with open('cat_to_name.json', 'r') as f:
    loadname_cat = json.load(f) 
        
#loadname_cat

# TODO: Build and train your network
#model = models.vgg16 (pretrained = True)
if pre_train.lower() == 'alexnet':
    model = models.alexnet(pretrained=True)
    model.name = "alexnet"
else:
    model = models.vgg16(pretrained=True)
    model.name = "vgg16"
#The models subpackage contains definitions of models for addressing different tasks, including: image classification, pixelwise semantic segmentation, object detection, instance segmentation, person keypoint detection and video classification.
#model.name = "vgg16"
model

for param in model.parameters(): 
    param.requires_grad = False

if pre_train.lower() == 'alexnet':
    classifier = nn.Sequential  (
                        nn.Linear (9216, 4096),
                        nn.ReLU (),
                        nn.Dropout (p = 0.3),
                        nn.Linear (4096, 2048),
                        nn.ReLU (),
                        nn.Dropout (p = 0.3),
                        nn.Linear (2048, 102),
                        nn.LogSoftmax (dim =1)
                        )
else:
    classifier = nn.Sequential(
        nn.Linear(25088, hid, bias=True),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(hid, 102, bias=True),
        nn.LogSoftmax(dim=1)
    )

model.classifier = classifier

if gpu == 'cuda':
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#tensor to routed to the CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


model.to(device);
print(model)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
print_every = 30 
steps = 0
epochs = args.epochs

# Implement a function for the validation pass
def validation(model, dataloader_test, criterion):
    cal_loss = 0
    accuracy = 0
    
    for count, (inputs, labels) in enumerate(dataloader_test):
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        output_val = model.forward(inputs)
        cal_loss += criterion(output_val, labels).item()
        
        texp_ps = torch.exp(output_val)
        equality = (labels.data == texp_ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return cal_loss, accuracy


for index in range(epochs):
    training_run_loss = 0
    model.train() 
    
    for count, (inputs, labels) in enumerate(dataloader_train):
        steps += 1
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward and backward passes
        outputs = model.forward(inputs)
        def_loss = criterion(outputs, labels)
        def_loss.backward()
        optimizer.step()
        
        training_run_loss += def_loss.item()
        
        if steps % print_every == 0:
            model.eval()

            with torch.no_grad():
                validation_loss, accuracy = validation(model, dataloader_valid, criterion)
            
            print("Epoch: {}/{} | ".format(index+1, epochs),
                  "Loss in Trainning : {:.4f} | ".format(training_run_loss/print_every),
                  "Loss in Validation: {:.4f} | ".format(validation_loss/len(dataloader_test)),
                  " Accuracy: {:.4f}".format(accuracy/len(dataloader_test)))
            
            running_loss = 0
            model.train()
  
# TODO: Do validation on the test set
correct_img = 0
total_count = 0
with torch.no_grad():
    model.eval()
    for data in dataloader_train:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total_count += labels.size(0)
        correct_img += (predicted == labels).sum().item()

print('with network test image accuricy is : %d%%' % (100 * correct_img / total_count))


# TODO: Save the checkpoint # TODO: Save the checkpoint 
# Create this `class_to_idx` attribute quickly
model.class_to_idx = datasets_imgtrain.class_to_idx


# In[92]:


checkpoint = {'architecture': model.name,
             'classifier': model.classifier,
             'class_to_idx': model.class_to_idx,
             'state_dict': model.state_dict()}


torch.save (checkpoint, args.save_dir)
