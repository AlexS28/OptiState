import pickle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from gru_model import RNN
import os
from torchvision import transforms
from PIL import Image
from transformer.transformer_model import Transformer_Autoencoder

# specify number of models to train
num_models = 1

# we train both the model for state output, and afterwards, model for the covariances
training_percentage = 0.8
# Hyper-parameters for state estimate
num_outputs = 12
input_size = 13+128
sequence_length = 10
hidden_size = 128+64
num_layers = 4
num_epochs = 5000
batch_size = 64
learning_rate = 0.00001

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using Device: {device}')

with open(dir_path + '/OptiState/data_collection/rnn_data.pkl', 'rb') as f:
    data_collection = pickle.load(f)

# get data from dictionary
state_KF_init = data_collection['state_KF']
state_VICON_init = data_collection['state_VICON']

# get images for autoencoder

# Define the transformation to apply to the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 64x64
    transforms.Grayscale(num_output_channels=1),  # Convert to greyscale
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
])

image_list = []

folder_path = dir_path + "/OptiState/transformer/imagesReal"
import re

def numerical_sort(value):
    # Extract the numerical part of the filename
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else -1


load_model_name = dir_path  + "/OptiState/transformer/trans_encoder_model"
model = Transformer_Autoencoder()
model.load_state_dict(torch.load(load_model_name))
model.eval()

# load the autoencoder and output encoder values
encoder_output = []
for filename in sorted(os.listdir(folder_path), key=numerical_sort):
    if filename.endswith(".png"):
        image_path = os.path.join(folder_path, filename)
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            img = transform(img).unsqueeze(0)
            with torch.no_grad():
                encoded = model.forward_encoder(img)
                encoded_1d = encoded.view(-1)
                encoded_1d_list = encoded_1d.numpy().tolist()
                encoder_output.append(encoded_1d_list)

# add the encoder output to the Kalman filter states for input to the GRU
for i in range(len(state_KF_init)):
    state_KF_init[i].extend(encoder_output[i])


# reshape data to have sequences of length sequence_length
#state_KF = [state_KF_init[i:i+sequence_length] for i in range(len(state_KF_init)-sequence_length+1)]
#state_VICON = [state_VICON[i+sequence_length-1] for i in range(len(state_KF))]
state_KF = []
state_IMU = []
for i in range(len(state_KF_init) - sequence_length + 1):
    state_KF.append(state_KF_init[i:i + sequence_length])

state_VICON = []
for i in range(len(state_KF)):
    state_VICON.append(state_VICON_init[i + sequence_length - 1])

# convert to tensor format
state_KF_tensor = torch.tensor(state_KF, dtype=torch.float32)
state_VICON_tensor = torch.tensor(state_VICON, dtype=torch.float32)

# Create TensorDataset object
dataset = TensorDataset(state_KF_tensor, state_VICON_tensor)

# Create DataLoader objects for training and testing
num_samples = len(dataset)
num_train = int(training_percentage * num_samples)  # 80% for training
num_test = num_samples - num_train  # 20% for testing

cur_model = 1
# set random seeds
import random
for i in range(num_models):
    seed = i + 1  # Use any logic to generate unique seed values
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # TODO: BELOW IS RANDOM SPLIT
    #train_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train, num_test])
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # TODO: BELOW IS NO RANDOM SPLIT
    # specify the training and testing percentages
    testing_percentage = 1 - training_percentage

    # calculate the number of samples for training and testing
    num_samples = len(dataset)
    num_train = int(training_percentage * num_samples)
    num_test = num_samples - num_train

    # create indices for training and testing samples
    train_indices = list(range(num_train))
    test_indices = list(range(num_train, num_samples))

    # create Subset objects using the indices
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # create DataLoader objects for training and testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = RNN(input_size, hidden_size, num_layers, num_outputs, device).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    train_loss = []
    test_loss = []
    loss_list_training = []

    for epoch in range(num_epochs):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list_training.append(loss.cpu().detach().numpy())

            # Print progress every 1000 iterations
            #if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.16f}')

    # save your model
    torch.save(model.state_dict(), dir_path + f'/OptiState/gru/gru_models/model{cur_model}.pth')
    cur_model += 1


plt.figure(1)
plt.plot(loss_list_training)
plt.title('Loss on training set: State')
plt.xlabel('Data points')
plt.ylabel('Loss')
plt.show()