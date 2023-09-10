#To train transformer:
#python autoencoder_training.py -use_model transformer -save_model_name trans_encoder_model -training_log_save transformer_log
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Autoencoder training', add_help=False)
    parser.add_argument('-lr', default=4e-4, type=float, help='initial learning rate')
    parser.add_argument('-weight_decay', default=0.1, type=float, help='weight decay for training')
    parser.add_argument('-batch_size', default=64, type=int, help='batch size for training')
    parser.add_argument('-num_epochs', default=1000, type=int, help='number of epochs for training')
    parser.add_argument('-load_model_name', default=None , help='name of the model to load before training')
    parser.add_argument('-save_model_name', default=None , help='name of the model to save after training')
    parser.add_argument('-training_log_save', default='transformer_log', help='save name for loss log of training')
    parser.add_argument('-use_model', default=None, help='the model to use (CNN or transformer)')
    return parser

args = get_args_parser()
args = args.parse_args()
lr = args.lr
weight_decay = args.weight_decay
batch_size = args.batch_size
num_epochs = args.num_epochs
load_model_name = args.load_model_name
save_model_name = args.save_model_name
training_log_save = args.training_log_save
use_model = args.use_model

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import random
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
from PIL import Image
from transformer_model import Transformer_Autoencoder
import timm.optim.optim_factory as optim_factory

# Define the transformation to apply to the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 64x64
    transforms.Grayscale(num_output_channels=1),  # Convert to greyscale
    transforms.RandomRotation(degrees=(-10, 10)), # data augmentation
    transforms.RandomVerticalFlip(p=0.5),  #  # Flip horizontally with a 50% probability
    transforms.RandomHorizontalFlip(p=0.5),  # Flip horizontally with a 50% probability
    transforms.RandomPerspective(p=0.5),
    transforms.RandomApply([
            transforms.RandomResizedCrop(size=(224, 224) , scale=(0.8, 1.0), ratio=(0.9, 1.1))
        ], p=0.3),
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
])

# Open the saved image using PIL
image_list = []
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
folder_path = dir_path + "/OptiState/data_collection/trajectories/saved_images/saved_images_combined"

idx = 0
image_iteration = 1
for i in range(1):
    for filename in os.listdir(folder_path):
        print(f'Converting images to tensor format, current image number: {image_iteration}')
        if filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                img = transform(img).unsqueeze(0)
                image_list.append(img)
                image_iteration += 1

training_list = image_list[:len(image_list) - 50]
val_list = image_list[len(image_list) - 50:]


model = Transformer_Autoencoder()

if not load_model_name is None:
    model.load_state_dict(torch.load(load_model_name))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

param_groups = optim_factory.add_weight_decay(model, weight_decay)
optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))


# Initialize lists to store the loss values
train_losses = []
val_losses = []

# Turn on interactive mode
plt.ion()

# Create a figure and axis object to plot the loss
fig, ax = plt.subplots()
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training and Validation Loss')
line1, = ax.plot(train_losses, label='Training Loss')
line2, = ax.plot(val_losses, label='Validation Loss')
ax.legend()

print("NOW TRAINING THE TRANSFORMER")
for epoch in range(num_epochs):
    # training
    train_loss = 0.0
    random.shuffle(training_list)
    for j in range(0, len(training_list), batch_size):
        batch = training_list[j:j+batch_size]
        image = torch.cat(batch, dim=0).to(device)
        loss, pred = model.forward(image)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss = train_loss / (len(training_list) / batch_size)
    train_losses.append(train_loss)

    # validation
    with torch.no_grad():
        val_loss = 0.0
        for k in range(len(val_list)):
            inp_img = val_list[k].to(device)
            loss, pred = model.forward(inp_img)
            val_loss += loss.item()
        val_loss = val_loss / len(val_list)
        val_losses.append(val_loss)

    # Update the plot
    line1.set_data(range(len(train_losses)), train_losses)
    line2.set_data(range(len(val_losses)), val_losses)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()

    if epoch % 1 == 0:
        print(f'Epoch:{epoch + 1}/{num_epochs}, Loss:{train_loss:.6f}, Val loss:{val_loss:.6f}')

    if epoch % 100 == 0:
        torch.save(model.state_dict(), save_model_name)

torch.save(model.state_dict(), save_model_name)
import numpy as np
np.savez(training_log_save, train=np.array(train_losses), val=np.array(val_losses))





