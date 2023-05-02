import pickle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from gru_model import RNN
import os

# specify if we want to train both state and covariance
train_state = True
train_cov = True

# we train both the model for state output, and afterwards, model for the covariances
training_percentage = 0.8
# Hyper-parameters for state estimate
num_outputs = 12
input_size = 12
sequence_length = 20
hidden_size = 128
num_layers = 2
num_epochs = 5000
batch_size = 10
learning_rate = 0.0001

# Hyper-parameters for COV estimate
num_outputs2 = 23
input_size2 = 12
sequence_length2 = 20
hidden_size2 = 128
num_layers2 = 2
num_epochs2 = 5000
batch_size2 = 10
learning_rate2 = 0.0001

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open(dir_path + '/OptiState/data_collection/rnn_data.pkl', 'rb') as f:
    data_collection = pickle.load(f)

# get data from dictionary
state_KF_init = data_collection['state_KF']
state_VICON_init = data_collection['state_VICON']

# reshape data to have sequences of length sequence_length
#state_KF = [state_KF_init[i:i+sequence_length] for i in range(len(state_KF_init)-sequence_length+1)]
#state_VICON = [state_VICON[i+sequence_length-1] for i in range(len(state_KF))]
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

# Create DataLoader objects for training and testing
num_samples = len(dataset)
num_train = int(training_percentage * num_samples)  # 80% for training
num_test = num_samples - num_train  # 20% for testing

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train, num_test])
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = RNN(input_size, hidden_size, num_layers, num_outputs, device).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.16f}')

# save your model
torch.save(model.state_dict(), dir_path + '/OptiState/gru/model.pth')

# test the model on unseen data
# Put model in eval mode
model.eval()

loss_list = []

for i, (inputs, labels) in enumerate(test_loader):
    inputs = inputs.to(device)
    labels = labels.to(device)
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels).cpu().detach().numpy()
    loss_list.append(loss)

plt.figure(1)
plt.plot(loss_list)
plt.title('Loss on testing set: State')
plt.xlabel('Data points')
plt.ylabel('Loss')

plt.figure(2)
plt.plot(loss_list_training)
plt.title('Loss on training set: State')
plt.xlabel('Data points')
plt.ylabel('Loss')



dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open(dir_path + '/OptiState/data_collection/rnn_data.pkl', 'rb') as f:
    data_collection = pickle.load(f)

# get data from dictionary
state_KF_init = data_collection['state_KF']
state_COV_init = data_collection['COV']

# reshape data to have sequences of length sequence_length
#state_KF = [state_KF_init[i:i+sequence_length] for i in range(len(state_KF_init)-sequence_length+1)]
#state_VICON = [state_VICON[i+sequence_length-1] for i in range(len(state_KF))]
state_KF = []
for i in range(len(state_KF_init) - sequence_length2 + 1):
    state_KF.append(state_KF_init[i:i + sequence_length2])

state_COV = []
for i in range(len(state_KF)):
    state_COV.append(state_COV_init[i + sequence_length2 - 1])

# convert to tensor format
state_KF_tensor = torch.tensor(state_KF, dtype=torch.float32)
state_COV_tensor = torch.tensor(state_COV, dtype=torch.float32)

# Create TensorDataset object
dataset = TensorDataset(state_KF_tensor, state_COV_tensor)

# Create DataLoader objects for training and testing
num_samples = len(dataset)
num_train = int(training_percentage * num_samples)  # 80% for training
num_test = num_samples - num_train  # 20% for testing

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train, num_test])
test_loader = DataLoader(test_dataset, batch_size=batch_size2, shuffle=True)

model2 = RNN(input_size2, hidden_size2, num_layers2, num_outputs2, device, True).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model2.parameters(), lr=learning_rate2)

train_loss = []
test_loss = []
loss_list_training = []
for epoch in range(num_epochs2):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model2(inputs)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list_training.append(loss.cpu().detach().numpy())

        # Print progress every 1000 iterations
        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs2}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.16f}')

# save your model
torch.save(model.state_dict(), dir_path + '/OptiState/gru/model_cov.pth')

# test the model on unseen data
# Put model in eval mode
model.eval()

loss_list = []

for i, (inputs, labels) in enumerate(test_loader):
    inputs = inputs.to(device)
    labels = labels.to(device)
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels).cpu().detach().numpy()
    loss_list.append(loss)

plt.figure(3)
plt.plot(loss_list)
plt.title('Loss on testing set: Cov')
plt.xlabel('Data points')
plt.ylabel('Loss')

plt.figure(4)
plt.plot(loss_list_training)
plt.title('Loss on training set: Cov')
plt.xlabel('Data points')
plt.ylabel('Loss')