import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from gru_model import RNN

# Hyper-parameters
num_outputs = 12
input_size = 12
sequence_length = 20
hidden_size = 128
num_layers = 2

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('/home/alexander/PycharmProjects/OptiState/data_collection/rnn_data.pkl', 'rb') as f:
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

model = RNN(input_size, hidden_size, num_layers, num_outputs, device).to(device)

# load your saved model
model.load_state_dict(torch.load('model.pth'))

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
        preds.append(outputs.cpu().numpy())
        ground_truth.append(labels.cpu().numpy())

# Convert predicted and ground truth lists to numpy arrays
preds = np.array(preds)
ground_truth = np.array(ground_truth)

# Reshape predicted and ground truth arrays to have the same shape
preds = preds.reshape(-1, num_outputs)
ground_truth = ground_truth.reshape(-1, num_outputs)
input_KF = np.array(state_KF_init)

# Plot predicted, ground truth, and input or Kalman filter output
plt.figure(1)
plt.plot(preds[:, 0], label="GRU")
plt.plot(ground_truth[:, 0], label="Vicon")
plt.plot(input_KF[:, 0], label="Kalman Filter")
plt.title('theta x')
plt.legend(['thx GRU','thx Vicon','thx Kalman Filter'])

plt.figure(2)
plt.plot(preds[:, 1], label="GRU")
plt.plot(ground_truth[:, 1], label="Vicon")
plt.plot(input_KF[:, 1], label="Kalman Filter")
plt.title('theta y')
plt.legend(['thy GRU','thy Vicon','thy Kalman Filter'])

plt.figure(3)
plt.plot(preds[:, 2], label="GRU")
plt.plot(ground_truth[:, 2], label="Vicon")
plt.plot(input_KF[:, 2], label="Kalman Filter")
plt.title('theta z')
plt.legend(['thz GRU','thz Vicon','thz Kalman Filter'])

plt.figure(4)
plt.plot(preds[:, 3], label="GRU")
plt.plot(ground_truth[:, 3], label="Vicon")
plt.plot(input_KF[:, 3], label="Kalman Filter")
plt.title('x')
plt.legend(['x GRU','x Vicon','x Kalman Filter'])

plt.figure(5)
plt.plot(preds[:, 4], label="GRU")
plt.plot(ground_truth[:, 4], label="Vicon")
plt.plot(input_KF[:, 4], label="Kalman Filter")
plt.title('y')
plt.legend(['y GRU','y Vicon','y Kalman Filter'])

plt.figure(6)
plt.plot(preds[:, 5], label="GRU")
plt.plot(ground_truth[:, 5], label="Vicon")
plt.plot(input_KF[:, 5], label="Kalman Filter")
plt.title('z')
plt.legend(['z GRU','z Vicon','z Kalman Filter'])

plt.figure(7)
plt.plot(preds[:, 6], label="GRU")
plt.plot(ground_truth[:, 6], label="Vicon")
plt.plot(input_KF[:, 6], label="Kalman Filter")
plt.title('dthx')
plt.legend(['dthx GRU','dthx Vicon','dthx Kalman Filter'])

plt.figure(8)
plt.plot(preds[:, 7], label="GRU")
plt.plot(ground_truth[:, 7], label="Vicon")
plt.plot(input_KF[:, 7], label="Kalman Filter")
plt.title('dthy')
plt.legend(['dthy GRU','dthy Vicon','dthy Kalman Filter'])

plt.figure(9)
plt.plot(preds[:, 8], label="GRU")
plt.plot(ground_truth[:, 8], label="Vicon")
plt.plot(input_KF[:, 8], label="Kalman Filter")
plt.title('dthz')
plt.legend(['dthz GRU','dthz Vicon','dthz Kalman Filter'])

plt.figure(10)
plt.plot(preds[:, 9], label="GRU")
plt.plot(ground_truth[:, 9], label="Vicon")
plt.plot(input_KF[:, 9], label="Kalman Filter")
plt.title('dx')
plt.legend(['dx GRU','dx Vicon','dx Kalman Filter'])

plt.figure(11)
plt.plot(preds[:, 10], label="GRU")
plt.plot(ground_truth[:, 10], label="Vicon")
plt.plot(input_KF[:, 10], label="Kalman Filter")
plt.title('dy')
plt.legend(['dy GRU','dy Vicon','dy Kalman Filter'])

plt.figure(12)
plt.plot(preds[:, 11], label="GRU")
plt.plot(ground_truth[:, 11], label="Vicon")
plt.plot(input_KF[:, 11], label="Kalman Filter")
plt.title('dz')
plt.legend(['dz GRU','dz Vicon','dz Kalman Filter'])

plt.show()