import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from gru_model import RNN
import os
from PIL import Image
from transformer.transformer_model import Transformer_Autoencoder
from torchvision import transforms
import torch.nn as nn
import shutil

# specify the dataset number to train on
dataset_train_number = [1,2,3,4,5,6]
# Train GRU2
training_percentage_2 = 1.0
batch_size_2 = 64
num_epochs_2 = 100
learning_rate_2 = 0.00001
hidden_size_2 = 128+64
num_layers_2 = 4

# Hyper-parameters (same values as GRU 1)
num_outputs = 12
input_size = 30+128
sequence_length = 10
hidden_size = 128+64
num_layers = 4
print("EVALUATING THE FIRST GRU, PLEASE WAIT...")
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open(dir_path+'/OptiState/data_collection/trajectories/rnn_data.pkl', 'rb') as f:
    data_collection = pickle.load(f)

# Define the transformation to apply to the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 64x64
    transforms.Grayscale(num_output_channels=1),  # Convert to greyscale
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
])

def numerical_sort(value):
    # Extract the numerical part of the filename
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else -1

image_list = []
import re

state_KF_init = []
state_VICON_init = []

image_iterate = 1
for i in range(len(dataset_train_number)):
    cur_dataset = dataset_train_number[i]
    state_KF_init.extend(data_collection[cur_dataset]['state_INPUT'])
    state_VICON_init.extend(data_collection[cur_dataset]['state_MOCAP'])

load_model_name = dir_path  + "/OptiState/transformer/trans_encoder_model"
model = Transformer_Autoencoder()
model.load_state_dict(torch.load(load_model_name))
model.eval()

file_encoder_path = dir_path + "/OptiState/data_collection/trajectories/encoder_output_training.pkl"  # Replace with the actual file path
# Load the data from the pickle file
with open(file_encoder_path, 'rb') as file:
    encoder_output = pickle.load(file)

# add the encoder output to the Kalman filter states for input to the GRU
for i in range(len(state_KF_init)):
    state_KF_init[i].extend(encoder_output[i])

state_KF = []
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

model = RNN(input_size, hidden_size, num_layers, num_outputs, device).to(device)

# load your saved model
model.load_state_dict(torch.load(dir_path + '/OptiState/gru/gru_models/model1.pth'))

# Put model in eval mode
model.eval()

# Create DataLoader object for entire dataset
data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

# Create lists to store predicted and ground truth values
preds = []
ground_truth = []
with torch.no_grad():
    for inputs, labels in data_loader:
        # Move inputs and labels to device
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(inputs)
        # Append predicted and ground truth values to respective lists
        preds_total = outputs.cpu().numpy()
        preds.append(preds_total)
        ground_truth.append(labels.cpu().numpy())


print("EVALUATION COMPLETE, NOW INITIATING TRAINING OF THE SECOND GRU")
# Convert predicted and ground truth lists to numpy arrays
preds = np.array(preds)
ground_truth = np.array(ground_truth)

# Reshape predicted and ground truth arrays to have the same shape
preds = preds.reshape(-1, num_outputs)
ground_truth = ground_truth.reshape(-1, num_outputs)
input_KF = np.array(state_KF_init)

output_error = []
for i in range(preds.shape[0]):
    thx_error = (preds[i, 0] - ground_truth[i, 0])
    thy_error = (preds[i, 1] - ground_truth[i, 1])
    thz_error = (preds[i, 2] - ground_truth[i, 2])
    x_error = (preds[i, 3] - ground_truth[i, 3])
    y_error = (preds[i, 4] - ground_truth[i, 4])
    z_error = (preds[i, 5] - ground_truth[i, 5])
    dthx_error = (preds[i, 6] - ground_truth[i, 6])
    dthy_error = (preds[i, 7] - ground_truth[i, 7])
    dthz_error = (preds[i, 8] - ground_truth[i, 8])
    dx_error = (preds[i, 9] - ground_truth[i, 9])
    dy_error = (preds[i, 10] - ground_truth[i, 10])
    dz_error = (preds[i, 11] - ground_truth[i, 11])
    output_error.append([thx_error, thy_error, thz_error, x_error, y_error, z_error, dthx_error, dthy_error, dthz_error, dx_error, dy_error, dz_error])

state_KF_2 = []
state_KF_init_2 = state_KF_init[:preds.shape[0]]
for i in range(len(state_KF_init_2) - sequence_length + 1):
    state_KF_2.append(state_KF_init_2[i:i + sequence_length])

state_output_error = []
for i in range(len(state_KF_2)):
    state_output_error.append(output_error[i + sequence_length - 1])

# convert to tensor format
state_KF_tensor = torch.tensor(state_KF_2, dtype=torch.float32)
state_output_error_tensor = torch.tensor(state_output_error, dtype=torch.float32)

# Create TensorDataset object
dataset = TensorDataset(state_KF_tensor, state_output_error_tensor)

# Create DataLoader objects for training and testing
num_samples = len(dataset)
num_train = int(training_percentage_2 * num_samples)  # 80% for training
num_test = num_samples - num_train  # 20% for testing

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train, num_test])

# create DataLoader objects for training and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size_2, shuffle=True)
model = RNN(input_size, hidden_size_2, num_layers_2, num_outputs, device).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_2, weight_decay=1e-5)

train_loss = []
test_loss = []
loss_list_training = []

for epoch in range(num_epochs_2):
    train_loader = DataLoader(train_dataset, batch_size=batch_size_2, shuffle=True)
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

    print(f'Epoch [{epoch + 1}/{num_epochs_2}], Loss: {loss.item():.16f}')

# save your model
torch.save(model.state_dict(), dir_path + f'/OptiState/gru/gru_models/model_error.pth')

plt.figure(1)
plt.plot(loss_list_training)
plt.title('Loss on training set: State')
plt.xlabel('Data points')
plt.ylabel('Loss')
plt.show()
