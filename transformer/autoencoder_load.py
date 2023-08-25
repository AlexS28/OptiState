#To load transformer:
#python autoencoder_load.py -use_model transformer -load_model_name trans_encoder_model -test_case 1000

#To load CNN:
#python autoencoder_load.py -use_model CNN -load_model_name CNN_encoder_model -test_case 1000


import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Autoencoder training', add_help=False)
    parser.add_argument('-test_case', default=1000, type=int, help='the case to test')
    parser.add_argument('-load_model_name', default=None , help='name of the model to load before training')
    parser.add_argument('-use_model', default=None, help='the model to use (CNN or transformer)')
    return parser

args = get_args_parser()
args = args.parse_args()
load_model_name = args.load_model_name
use_model = args.use_model
test_case = args.test_case

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib as mpl
mpl.use('tkagg')

import matplotlib.pyplot as plt
from PIL import Image

# Define the transformation to apply to the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 64x64
    transforms.Grayscale(num_output_channels=1),  # Convert to greyscale
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
])
# Open the saved image using PIL
image_list = []
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
folder_path = dir_path + "/OptiState/data_collection/trajectories/saved_images/saved_images_traj_1"

import re

def numerical_sort(value):
    # Extract the numerical part of the filename
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else -1

for filename in sorted(os.listdir(folder_path), key=numerical_sort):
    if filename.endswith(".png"):
        image_path = os.path.join(folder_path, filename)
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            img = transform(img).unsqueeze(0)
            image_list.append(img)

from transformer_model import Transformer_Autoencoder



if use_model == 'transformer':
    model = Transformer_Autoencoder()
else:
    print("Not valid model, please use CNN or transformer")

model.load_state_dict(torch.load(load_model_name))
model.eval()

input_img = image_list[test_case]


with torch.no_grad():
    loss, pred = model.forward(input_img)

if use_model == 'transformer':
    output_img = model.unpatchify(pred)
else:
    output_img = pred


# Generate and display the input and output images
plt.subplot(1, 2, 1)
plt.imshow(input_img.squeeze(), cmap='gray')
plt.title('Input Image')
plt.subplot(1, 2, 2)
plt.imshow(output_img.squeeze().detach(), cmap='gray')
plt.title('Output Image')
plt.show()

model.eval()
# Pass the input image through the encoder to get the latent representation
with torch.no_grad():
    if use_model == 'transformer':
        encoded = model.forward_encoder(input_img)
    else:
        encoded = model.encoder(input_img)

# Print the shape of the encoded tensor
print("Encoded tensor shape:", encoded.shape)

# Reshape the encoded tensor into a 1D array
encoded_1d = encoded.view(-1)

# Print the shape of the 1D encoded tensor
print("Encoded 1D tensor shape:", encoded_1d.shape)

encoded_1d_np = encoded_1d.numpy()

print(encoded_1d_np)

