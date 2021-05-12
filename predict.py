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
import os
import argparse

args_val = argparse.ArgumentParser (description = "Train")
args_val.add_argument ('--image_path', help = 'Provide image path. Mandatory argument', type = str)
args_val.add_argument ('--checkpoint', help = 'Provide checkpoint path. Optional argument', type = str)
args_val.add_argument('--gpu', action="store", help='Use GPU + Cuda for calculations')
args_val.add_argument ('--pre_train', default='vgg16', help = 'Provide pre-trained models', type = str)
args_val.add_argument('--top_k', default=5, type=int, help='Choose top K matches as integer value ')


args = args_val.parse_args ()
pre_train = args.pre_train
gpu = args.gpu 
top_k= args.top_k


def load_checkpoint():
# get checkpoint of  deep learning model .
    
    
    # Load the saved file my_checkpoint.pth
    file = args.checkpoint
    checkpoint = torch.load(file)
    
    if pre_train == 'alexnet':
        model = models.alexnet(pretrained=True); #get the pretrained model
    else:
        model = models.vgg16(pretrained=True);
    for param in model.parameters(): param.requires_grad = False
    
    model.class_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.state_dict(checkpoint['state_dict'])
    return model

def process_image(image):
     #clssify the images with  PyTorch model and return numpy array 
        
    # Normalize each color channel
    normalisemeans = [0.485, 0.456, 0.406]
    normalisestd = [0.229, 0.224, 0.225]

    eval_image = PIL.Image.open(image)
    orig_width, orig_height = eval_image.size #width
    if orig_width < orig_height: resize_size=[256, 256**600]
    else: resize_size=[256**600, 256]
        
    eval_image.thumbnail(size=resize_size)
    center = orig_width/4, orig_height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    eval_image = eval_image.crop((left, top, right, bottom))
    nump_image = np.array(eval_image)/255 

    
    nump_image = (nump_image-normalisemeans)/normalisestd
        
    # Set the color to the first channel
    nump_image = nump_image.transpose(2, 0, 1)
    
    return nump_image

def predict(image_path, model, top_k):
    #predict the class of Image
    #cuda start
    """if gpu == 'cuda':
        device = 'cuda'
    else:
        device = 'cpu'
    device

    model.to(device);"""
    #cuda end
    model.to("cpu")
    model.eval();

    new_image_torch = torch.from_numpy(np.expand_dims(process_image(image_path), 
                                                  axis=0)).type(torch.FloatTensor).to("cpu")
    probs = model.forward(new_image_torch)

    # Convert to linear scale
    linear_scale = torch.exp(probs)
    #top results
    top_results, top_labels = linear_scale.topk(top_k)
    
    top_results = np.array(top_results.detach())[0]
    top_labels = np.array(top_labels.detach())[0]
    
    # Convert to classes
    """class_idx = {val: key for key, val in    
                                      model.class_to_idx.items()}"""
    #top_labels = [class_idx[lab] for lab in top_labels]
    
    #top_flowers = [loadname_cat[lab] for lab in top_labels]
    
    return top_results, top_labels

image_path =  args.image_path
model = load_checkpoint()
probs, labs = predict(image_path, model, top_k)
print(probs)
print(labs)