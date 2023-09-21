#To load transformer:
#python autoencoder_load.py -load_model_name trans_encoder_model
from transformer_model import Transformer_Autoencoder
import argparse
import time


def get_args_parser():
    parser = argparse.ArgumentParser('Autoencoder training', add_help=False)
    parser.add_argument('-load_model_name', default=None , help='name of the model to load before training')
    return parser

args = get_args_parser()
args = args.parse_args()
load_model_name = args.load_model_name

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
folder_path = dir_path + "/OptiState/data_collection/trajectories/saved_images/saved_images_combined"
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

model = Transformer_Autoencoder()
model.load_state_dict(torch.load(load_model_name))
model.eval()

import cv2
import numpy as np
for i in range(len(image_list)):
    input_img = image_list[i]
    with torch.no_grad():
        _, pred = model.forward(input_img)
        output_img = model.unpatchify(pred)

        # Convert the input and output images to numpy arrays
        input_img = input_img.squeeze().detach().cpu().numpy()
        output_img = output_img.squeeze().detach().cpu().numpy()

        # Concatenate the input and output images horizontally
        side_by_side = np.hstack((input_img, output_img))

        # Create a named window
        cv2.namedWindow('Input and Output Images', cv2.WINDOW_AUTOSIZE)

        # Show the concatenated image in the window
        cv2.imshow('Input and Output Images', side_by_side)
        if i == 1:
            time.sleep(1)
        # Wait for a key press and close the window if 'q' is pressed
        key = cv2.waitKey(1)
        if key == ord('q'):
            break