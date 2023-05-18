import pickle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from gru_model import RNN
import os

# Hyper-parameters
num_models = 2 # TODO: Ensure this number is the same as the number of models in the gru_models directory!!!
num_outputs = 12
input_size = 13
sequence_length = 20
hidden_size = 128
num_layers = 2

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open(dir_path+'/OptiState/data_collection/rnn_data.pkl', 'rb') as f:
    data_collection = pickle.load(f)

# get data from dictionary
state_KF_init = data_collection['state_KF']
state_VICON_init = data_collection['state_VICON']

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

COV00 = []
COV11 = []
COV22 = []
COV33 = []
COV44 = []
COV55 = []
COV66 = []
COV77 = []
COV88 = []
COV99 = []
COV1010 = []
COV1111 = []

COV_list = []

# Create lists to store predicted and ground truth values
preds_list = []
ground_truth_list = []

# Create DataLoader object for entire dataset
data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

cov_dict_models = {}
preds_dict_models = {}
preds = []

import random
for i in range(1,num_models+1):
    seed = i + 1  # Use any logic to generate unique seed values
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = RNN(input_size, hidden_size, num_layers, num_outputs, device).to(device)
    model.load_state_dict(torch.load(dir_path + f'/OptiState/gru/gru_models/model{i}.pth'))
    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            # Move inputs and labels to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Append predicted and ground truth values to respective lists
            preds_total = outputs.cpu().numpy().reshape(12,1)
            ground_truth = labels.cpu().numpy().reshape(12,1)

            preds_list.append(preds_total)
            ground_truth_list.append(labels.cpu().numpy())

            COV00 = (preds_total[0] - ground_truth[0])**2
            COV11 = (preds_total[1] - ground_truth[1])**2
            COV22 = (preds_total[2] - ground_truth[2])**2
            COV33 = (preds_total[3] - ground_truth[3])**2
            COV44 = (preds_total[4] - ground_truth[4])**2
            COV55 = (preds_total[5] - ground_truth[5])**2
            COV66 = (preds_total[6] - ground_truth[6])**2
            COV77 = (preds_total[7] - ground_truth[7])**2
            COV88 = (preds_total[8] - ground_truth[8])**2
            COV99 = (preds_total[9] - ground_truth[9])**2
            COV1010 = (preds_total[10] - ground_truth[10])**2
            COV1111 = (preds_total[11] - ground_truth[11])**2

            COV_list.append([COV00, COV11, COV22, COV33, COV44, COV55, COV66, COV77, COV88, COV99, COV1010, COV1111])
            preds.append([preds_total[0],preds_total[1],preds_total[2],preds_total[3],preds_total[4],preds_total[5],
                     preds_total[6],preds_total[7],preds_total[8],preds_total[9],preds_total[10],preds_total[11]])

    cov_dict_models.update({i: {'cov_list': COV_list}})
    preds_dict_models.update({i: {'pred_list': preds}})
    COV_list = []
    ground_truth_list = []
    preds_list = []

trajectory_length  = len(cov_dict_models[1]['cov_list'])
cov_00 = []
cov_11 = []
cov_22 = []
cov_33 = []
cov_44 = []
cov_55 = []
cov_66 = []
cov_77 = []
cov_88 = []
cov_99 = []
cov_1010 = []
cov_1111 = []

pred_00 = []
pred_11 = []
pred_22 = []
pred_33 = []
pred_44 = []
pred_55 = []
pred_66 = []
pred_77 = []
pred_88 = []
pred_99 = []
pred_1010 = []
pred_1111 = []

for i in range(trajectory_length):
    error_00 = 0
    error_11 = 0
    error_22 = 0
    error_33 = 0
    error_44 = 0
    error_55 = 0
    error_66 = 0
    error_77 = 0
    error_88 = 0
    error_99 = 0
    error_1010 = 0
    error_1111 = 0

    errorPred_00 = 0
    errorPred_11 = 0
    errorPred_22 = 0
    errorPred_33 = 0
    errorPred_44 = 0
    errorPred_55 = 0
    errorPred_66 = 0
    errorPred_77 = 0
    errorPred_88 = 0
    errorPred_99 = 0
    errorPred_1010 = 0
    errorPred_1111 = 0

    for j in range(1,num_models+1):
        cov_list_cur = cov_dict_models[j]['cov_list']
        pred_list_cur = preds_dict_models[j]['pred_list']
        cov_state = cov_list_cur[i]
        pred_state = pred_list_cur[i]
        error_00 += cov_state[0]
        error_11 += cov_state[1]
        error_22 += cov_state[2]
        error_33 += cov_state[3]
        error_44 += cov_state[4]
        error_55 += cov_state[5]
        error_66 += cov_state[6]
        error_77 += cov_state[7]
        error_88 += cov_state[8]
        error_99 += cov_state[9]
        error_1010 += cov_state[10]
        error_1111 += cov_state[11]

        errorPred_00 += pred_state[0]
        errorPred_11 += pred_state[1]
        errorPred_22 += pred_state[2]
        errorPred_33 += pred_state[3]
        errorPred_44 += pred_state[4]
        errorPred_55 += pred_state[5]
        errorPred_66 += pred_state[6]
        errorPred_77 += pred_state[7]
        errorPred_88 += pred_state[8]
        errorPred_99 += pred_state[9]
        errorPred_1010 += pred_state[10]
        errorPred_1111 += pred_state[11]

    cov_00.append(error_00 / (num_models - 1))
    cov_11.append(error_11 / (num_models - 1))
    cov_22.append(error_22 / (num_models - 1))
    cov_33.append(error_33 / (num_models - 1))
    cov_44.append(error_44 / (num_models - 1))
    cov_55.append(error_55 / (num_models - 1))
    cov_66.append(error_66 / (num_models - 1))
    cov_77.append(error_77 / (num_models - 1))
    cov_88.append(error_88 / (num_models - 1))
    cov_99.append(error_99 / (num_models - 1))
    cov_1010.append(error_1010 / (num_models - 1))
    cov_1111.append(error_1111 / (num_models - 1))

    pred_00.append(errorPred_00 / (num_models))
    pred_11.append(errorPred_11 / (num_models))
    pred_22.append(errorPred_22 / (num_models))
    pred_33.append(errorPred_33 / (num_models))
    pred_44.append(errorPred_44 / (num_models))
    pred_55.append(errorPred_55 / (num_models))
    pred_66.append(errorPred_66 / (num_models))
    pred_77.append(errorPred_77 / (num_models))
    pred_88.append(errorPred_88 / (num_models))
    pred_99.append(errorPred_99 / (num_models))
    pred_1010.append(errorPred_1010 / (num_models))
    pred_1111.append(errorPred_1111 / (num_models))

input_KF = np.array(state_KF_init)
ground_truth = np.array(state_VICON_init)

pred_00_high = []
pred_00_low = []

pred_11_high = []
pred_11_low = []

pred_22_high = []
pred_22_low = []

pred_33_high = []
pred_33_low = []

pred_44_high = []
pred_44_low = []

pred_55_high = []
pred_55_low = []

pred_66_high = []
pred_66_low = []

pred_77_high = []
pred_77_low = []

pred_88_high = []
pred_88_low = []

pred_99_high = []
pred_99_low = []

pred_1010_high = []
pred_1010_low = []

pred_1111_high = []
pred_1111_low = []

for i in range(len(pred_00)):
    pred_00_low.append(pred_00[i] - np.sqrt(cov_00[i]))
    pred_00_high.append(pred_00[i] + np.sqrt(cov_00[i]))

    pred_11_low.append(pred_11[i] - np.sqrt(cov_11[i]))
    pred_11_high.append(pred_11[i] + np.sqrt(cov_11[i]))

    pred_22_low.append(pred_22[i] - np.sqrt(cov_22[i]))
    pred_22_high.append(pred_22[i] + np.sqrt(cov_22[i]))

    pred_33_low.append(pred_33[i] - np.sqrt(cov_33[i]))
    pred_33_high.append(pred_33[i] + np.sqrt(cov_33[i]))

    pred_44_low.append(pred_44[i] - np.sqrt(cov_44[i]))
    pred_44_high.append(pred_44[i] + np.sqrt(cov_44[i]))

    pred_55_low.append(pred_55[i] - np.sqrt(cov_55[i]))
    pred_55_high.append(pred_55[i] + np.sqrt(cov_55[i]))

    pred_66_low.append(pred_66[i] - np.sqrt(cov_66[i]))
    pred_66_high.append(pred_66[i] + np.sqrt(cov_66[i]))

    pred_77_low.append(pred_77[i] - np.sqrt(cov_77[i]))
    pred_77_high.append(pred_77[i] + np.sqrt(cov_77[i]))

    pred_88_low.append(pred_88[i] - np.sqrt(cov_88[i]))
    pred_88_high.append(pred_88[i] + np.sqrt(cov_88[i]))

    pred_99_low.append(pred_99[i] - np.sqrt(cov_99[i]))
    pred_99_high.append(pred_99[i] + np.sqrt(cov_99[i]))

    pred_1010_low.append(pred_1010[i] - np.sqrt(cov_1010[i]))
    pred_1010_high.append(pred_1010[i] + np.sqrt(cov_1010[i]))

    pred_1111_low.append(pred_1111[i] - np.sqrt(cov_1111[i]))
    pred_1111_high.append(pred_1111[i] + np.sqrt(cov_1111[i]))

# Convert the lists to arrays
pred_00 = np.array(pred_00)
pred_00_low = np.array(pred_00_low)
pred_00_high = np.array(pred_00_high)

# Convert the lists to arrays
pred_11 = np.array(pred_11)
pred_11_low = np.array(pred_11_low)
pred_11_high = np.array(pred_11_high)

# Convert the lists to arrays
pred_22 = np.array(pred_22)
pred_22_low = np.array(pred_22_low)
pred_22_high = np.array(pred_22_high)

# Convert the lists to arrays
pred_33 = np.array(pred_33)
pred_33_low = np.array(pred_33_low)
pred_33_high = np.array(pred_33_high)

# Convert the lists to arrays
pred_44 = np.array(pred_44)
pred_44_low = np.array(pred_44_low)
pred_44_high = np.array(pred_44_high)

# Convert the lists to arrays
pred_55 = np.array(pred_55)
pred_55_low = np.array(pred_55_low)
pred_55_high = np.array(pred_55_high)

# Convert the lists to arrays
pred_66 = np.array(pred_66)
pred_66_low = np.array(pred_66_low)
pred_66_high = np.array(pred_66_high)

# Convert the lists to arrays
pred_77 = np.array(pred_77)
pred_77_low = np.array(pred_77_low)
pred_77_high = np.array(pred_77_high)

# Convert the lists to arrays
pred_88 = np.array(pred_88)
pred_88_low = np.array(pred_88_low)
pred_88_high = np.array(pred_88_high)

# Convert the lists to arrays
pred_99 = np.array(pred_99)
pred_99_low = np.array(pred_99_low)
pred_99_high = np.array(pred_99_high)

# Convert the lists to arrays
pred_1010 = np.array(pred_1010)
pred_1010_low = np.array(pred_1010_low)
pred_1010_high = np.array(pred_1010_high)

# Convert the lists to arrays
pred_1111 = np.array(pred_1111)
pred_1111_low = np.array(pred_1111_low)
pred_1111_high = np.array(pred_1111_high)

# Create x-values
x0 = np.arange(len(pred_00))
x1 = np.arange(len(pred_11))
x2 = np.arange(len(pred_22))
x3 = np.arange(len(pred_33))
x4 = np.arange(len(pred_44))
x5 = np.arange(len(pred_55))
x6 = np.arange(len(pred_66))
x7 = np.arange(len(pred_77))
x8 = np.arange(len(pred_88))
x9 = np.arange(len(pred_99))
x10 = np.arange(len(pred_1010))
x11 = np.arange(len(pred_1111))


# Plot data
plt.figure(1)
plt.plot(ground_truth[sequence_length:,0])
plt.plot(input_KF[sequence_length:,0])
plt.plot(pred_00, label='Pred')
# Customize plot appearance
plt.title('theta x')
plt.legend(['thx Vicon', 'thx Kalman Filter', 'thx GRU'])
plt.fill_between(x0, pred_00_low.flatten(), pred_00_high.flatten(), color='grey', alpha=0.6, linewidth=0)  # Fill area between bounds

plt.figure(2)
plt.plot(ground_truth[sequence_length:,1])
plt.plot(input_KF[sequence_length:,1])
plt.plot(pred_11, label='Pred')
# Customize plot appearance
plt.title('theta y')
plt.legend(['thy Vicon', 'thy Kalman Filter', 'thy GRU'])
plt.fill_between(x1, pred_11_low.flatten(), pred_11_high.flatten(), color='grey', alpha=0.6, linewidth=0)  # Fill area between bounds

plt.figure(3)
plt.plot(ground_truth[sequence_length:,2])
plt.plot(input_KF[sequence_length:,2])
plt.plot(pred_22, label='Pred')
# Customize plot appearance
plt.title('theta z')
plt.legend(['thz Vicon', 'thz Kalman Filter', 'thz GRU'])
plt.fill_between(x2, pred_22_low.flatten(), pred_22_high.flatten(), color='grey', alpha=0.6, linewidth=0)  # Fill area between bounds

plt.figure(4)
plt.plot(ground_truth[sequence_length:,3])
plt.plot(input_KF[sequence_length:,3])
plt.plot(pred_33, label='Pred')
# Customize plot appearance
plt.title('x pos')
plt.legend(['x Vicon', 'x Kalman Filter', 'x GRU'])
plt.fill_between(x3, pred_33_low.flatten(), pred_33_high.flatten(), color='grey', alpha=0.6, linewidth=0)  # Fill area between bounds

plt.figure(5)
plt.plot(ground_truth[sequence_length:,4])
plt.plot(input_KF[sequence_length:,4])
plt.plot(pred_44, label='Pred')
# Customize plot appearance
plt.title('y pos')
plt.legend(['y Vicon', 'y Kalman Filter', 'y GRU'])
plt.fill_between(x4, pred_44_low.flatten(), pred_44_high.flatten(), color='grey', alpha=0.6, linewidth=0)  # Fill area between bounds

plt.figure(6)
plt.plot(ground_truth[sequence_length:,5])
plt.plot(input_KF[sequence_length:,5])
plt.plot(pred_55, label='Pred')
# Customize plot appearance
plt.title('z pos')
plt.legend(['z Vicon', 'z Kalman Filter', 'z GRU'])
plt.fill_between(x5, pred_55_low.flatten(), pred_55_high.flatten(), color='grey', alpha=0.6, linewidth=0)  # Fill area between bounds


plt.figure(7)
plt.plot(ground_truth[sequence_length:,6])
plt.plot(input_KF[sequence_length:,6])
plt.plot(pred_66, label='Pred')
# Customize plot appearance
plt.title('dthx')
plt.legend(['dthx Vicon', 'dthx Kalman Filter', 'dthx GRU'])
plt.fill_between(x6, pred_66_low.flatten(), pred_66_high.flatten(), color='grey', alpha=0.6, linewidth=0)  # Fill area between bounds


plt.figure(8)
plt.plot(ground_truth[sequence_length:,7])
plt.plot(input_KF[sequence_length:,7])
plt.plot(pred_77, label='Pred')
# Customize plot appearance
plt.title('dthy')
plt.legend(['dthy Vicon', 'dthy Kalman Filter', 'dthy GRU'])
plt.fill_between(x7, pred_77_low.flatten(), pred_77_high.flatten(), color='grey', alpha=0.6, linewidth=0)  # Fill area between bounds

plt.figure(9)
plt.plot(ground_truth[sequence_length:,8])
plt.plot(input_KF[sequence_length:,8])
plt.plot(pred_88, label='Pred')
# Customize plot appearance
plt.title('dthz')
plt.legend(['dthz Vicon', 'dthz Kalman Filter', 'dthz GRU'])
plt.fill_between(x8, pred_88_low.flatten(), pred_88_high.flatten(), color='grey', alpha=0.6, linewidth=0)  # Fill area between bounds


plt.figure(10)
plt.plot(ground_truth[sequence_length:,9])
plt.plot(input_KF[sequence_length:,9])
plt.plot(pred_99)
# Customize plot appearance
plt.title('dx')
plt.legend(['dx Vicon', 'dx Kalman Filter', 'dx GRU'])
plt.fill_between(x9, pred_99_low.flatten(), pred_99_high.flatten(), color='grey', alpha=0.6, linewidth=0)  # Fill area between bounds


plt.figure(11)
plt.plot(ground_truth[sequence_length:,10])
plt.plot(input_KF[sequence_length:,10])
plt.plot(pred_1010)
# Customize plot appearance
plt.title('dy')
plt.legend(['dy Vicon', 'dy Kalman Filter', 'dy GRU'])
plt.fill_between(x10, pred_1010_low.flatten(), pred_1010_high.flatten(), color='grey', alpha=0.6, linewidth=0)  # Fill area between bounds


plt.figure(12)
plt.plot(ground_truth[sequence_length:,11])
plt.plot(input_KF[sequence_length:,11])
plt.plot(pred_1111)
# Customize plot appearance
plt.title('dz')
plt.legend(['dz Vicon', 'dz Kalman Filter', 'dz GRU'])
plt.fill_between(x11, pred_1111_low.flatten(), pred_1111_high.flatten(), color='grey', alpha=0.6, linewidth=0)  # Fill area between bounds

# Show the plot
plt.show()




