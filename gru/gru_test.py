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
from scipy import io
from settings import INITIAL_PARAMS
# specify the dataset number to test on
dataset_test_number = [1,2]

hidden_size_2 = 128+64
num_layers_2 = 4

# Hyper-parameters (same values as GRU 1)
num_outputs = 24
if INITIAL_PARAMS.USE_VISION:
    input_size = 76-24+128
else:
    input_size = 76-24
sequence_length = 10
hidden_size = 128+64
num_layers = 4

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open(dir_path+'/OptiState/data_collection/trajectories/rnn_data.pkl', 'rb') as f:
    data_collection = pickle.load(f)

# Specify the file path to save the pickle file
pickle_file_path =  dir_path + '/OptiState/data_collection/trajectories/scaling_params.pkl'
# Load the dictionary from the pickle file
with open(pickle_file_path, 'rb') as pickle_file:
    loaded_scaling_params = pickle.load(pickle_file)

# Extract min_vals and max_vals from the loaded dictionary
min_vals = loaded_scaling_params['min_vals_KF']
max_vals = loaded_scaling_params['max_vals_KF']
min_vals_VIC = loaded_scaling_params['min_vals_VIC']
max_vals_VIC = loaded_scaling_params['max_vals_VIC']


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

import re


# get data from dictionary
state_KF_init = []
state_VICON_init = []
state_t265_init = []
images_folder_test = dir_path + '/OptiState/data_collection/trajectories/saved_images/saved_images_traj_test'
# Check if the folder exists
if os.path.exists(images_folder_test):
    # Remove the folder and its contents
    shutil.rmtree(images_folder_test)

if not os.path.exists(images_folder_test):
    os.makedirs(images_folder_test)

image_iterate = 1
for i in range(len(dataset_test_number)):
    cur_dataset = dataset_test_number[i]
    state_KF_init.extend(data_collection[cur_dataset]['state_INPUT'])
    state_VICON_init.extend(data_collection[cur_dataset]['state_MOCAP'])
    state_t265_init.extend(data_collection[cur_dataset]['state_T265'])

    if INITIAL_PARAMS.USE_VISION:
        cur_image_directory = dir_path + f'/OptiState/data_collection/trajectories/saved_images/saved_images_traj_{cur_dataset}'
        image_files = [file for file in os.listdir(cur_image_directory) if file.lower().endswith('.png')]
        # Rename and copy images to the training folder
        for index, image_file in enumerate(image_files):
            new_filename = f"image_{cur_dataset}_{image_iterate}.png"
            source_path = os.path.join(cur_image_directory, image_file)
            destination_path = os.path.join(images_folder_test, new_filename)
            shutil.copyfile(source_path, destination_path)
            image_iterate+=1

load_model_name = dir_path  + "/OptiState/transformer/trans_encoder_model"
model = Transformer_Autoencoder()
model.load_state_dict(torch.load(load_model_name))
model.eval()


# Convert your list of lists to a numpy array for easier processing
state_KF_init_array = np.array(state_KF_init)
# Perform Min-Max scaling for each component
normalized_state_KF_init = (state_KF_init_array - min_vals) / (max_vals - min_vals)
# The 'normalized_state_KF_init' array now contains the normalized values
# Each row corresponds to a list in the original 'state_KF_init'
# If you want to convert the normalized numpy array back to a list of lists
state_KF_init = normalized_state_KF_init.tolist()

# Convert your list of lists to a numpy array for easier processing
state_VICON_init_array = np.array(state_VICON_init)
# Perform Min-Max scaling for each component
normalized_state_VICON_init = (state_VICON_init_array - min_vals_VIC) / (max_vals_VIC - min_vals_VIC)
# The 'normalized_state_KF_init' array now contains the normalized values
# Each row corresponds to a list in the original 'state_KF_init'
# If you want to convert the normalized numpy array back to a list of lists
state_VICON_init = normalized_state_VICON_init.tolist()

if INITIAL_PARAMS.USE_VISION:
    # load the autoencoder and output encoder values
    encoder_output = []
    counter = 0
    for filename in sorted(os.listdir(images_folder_test), key=numerical_sort):
        if filename.endswith(".png"):
            image_path = os.path.join(images_folder_test, filename)
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                img = transform(img).unsqueeze(0)
                with torch.no_grad():
                    encoded = model.forward_encoder(img)
                    encoded_1d = encoded.view(-1)
                    encoded_1d_list = encoded_1d.numpy().tolist()
                    encoder_output.append(encoded_1d_list)
                    counter += 1
                    if counter == len(state_KF_init):
                        break

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
# Create DataLoader object for entire dataset
data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

model = RNN(input_size, hidden_size, num_layers, num_outputs, device, evaluate=True).to(device)

if INITIAL_PARAMS.USE_VISION:
    # load your saved model
    model.load_state_dict(torch.load(dir_path + '/OptiState/gru/gru_models/model1_vision.pth'))
else:
    # load your saved model
    model.load_state_dict(torch.load(dir_path + '/OptiState/gru/gru_models/model1.pth'))
# Put model in eval mode
model.eval()


# Create lists to store predicted and ground truth values
preds = []
ground_truth = []
preds_error = []
with torch.no_grad():
    for inputs, labels in data_loader:
        # Move inputs and labels to device
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(inputs)
        # Append predicted and ground truth values to respective lists
        preds_np = outputs[0,0:12].cpu().numpy().reshape(12,1)
        preds_error_np = outputs[0,12:].cpu().numpy().reshape(12,1)
        preds_np = preds_np - preds_error_np

        preds.append(preds_np)
        preds_error.append(preds_error_np)
        ground_truth_np = labels.cpu().numpy()
        ground_truth.append(ground_truth_np)

# Convert predicted and ground truth lists to numpy arrays
preds = np.array(preds)
ground_truth = np.array(ground_truth)
preds_error = np.array(preds_error)

# Reshape predicted and ground truth arrays to have the same shape
preds = preds.reshape(-1, 12)
ground_truth = ground_truth.reshape(-1, 12)
input_KF = np.array(state_KF_init)
preds = preds * (max_vals_VIC - min_vals_VIC) + min_vals_VIC
ground_truth = ground_truth * (max_vals_VIC - min_vals_VIC) + min_vals_VIC

state_t265 = []
for i in range(sequence_length-1,len(state_t265_init)):
    state_t265.append(state_t265_init[i])

state_t265 = np.array(state_t265)

from scipy.signal import butter, lfilter
# Moving average window size
#window_size = 10  # Adjust this according to your needs
window_size_GRU = 1
# Apply the moving average filter to all components of the data
filtered_preds = np.zeros_like(preds)
for i in range(preds.shape[1]):
    preds[:, i] = np.convolve(preds[:, i], np.ones(window_size_GRU) / window_size_GRU, mode='same')
    #ground_truth[:, i] = np.convolve(ground_truth[:, i], np.ones(200) / 200, mode='same')
    #input_KF[:, i] = np.convolve(input_KF[:, i], np.ones(100) / 100, mode='same')


end_plot = preds.shape[0]

# Plot predicted, ground truth, and input or Kalman filter output
plt.figure(1)
plt.plot(preds[:end_plot, 0], label="GRU")
#plt.fill_between(range(end_plot), preds[:end_plot, 0] - error_GRU[:end_plot,0], preds[:end_plot, 0] + error_GRU[:end_plot,0], alpha=0.3)
plt.plot(ground_truth[:, 0], label="Vicon")
plt.plot(state_t265[:,0], label='Baseline')
plt.title('theta x')
plt.legend(['thx GRU','thx Vicon','thx Kalman Filter','Baseline'])

plt.figure(2)
plt.plot(preds[:, 1], label="GRU")
#plt.fill_between(range(end_plot), preds[:end_plot, 1] - error_GRU[:end_plot,1], preds[:end_plot, 1] + error_GRU[:end_plot,1], alpha=0.3)
plt.plot(ground_truth[:, 1], label="Vicon")
plt.plot(state_t265[:,1], label='Baseline')
plt.title('theta y')
plt.legend(['thy GRU','thy Vicon','thy Kalman Filter','Basline'])

plt.figure(3)
plt.plot(preds[:, 2], label="GRU")
#plt.fill_between(range(end_plot), preds[:end_plot, 2] - error_GRU[:end_plot,2], preds[:end_plot, 2] + error_GRU[:end_plot,2], alpha=0.3)
plt.plot(ground_truth[:, 2], label="Vicon")
plt.plot(state_t265[:,2], label='Baseline')
plt.title('theta z')
plt.legend(['thz GRU','thz Vicon','thz Kalman Filter','Basline'])

plt.figure(4)
plt.plot(preds[:, 3], label="GRU")
#plt.fill_between(range(end_plot), preds[:end_plot, 3] - error_GRU[:end_plot,3], preds[:end_plot, 3] + error_GRU[:end_plot,3], alpha=0.3)
plt.plot(ground_truth[:, 3], label="Vicon")
plt.plot(state_t265[:,3], label='Baseline')
plt.title('x')
plt.legend(['x GRU', 'x Vicon','x Kalman Filter','Baseline'])

plt.figure(5)
plt.plot(preds[:, 4], label="GRU")
#plt.fill_between(range(end_plot), preds[:end_plot, 4] - error_GRU[:end_plot, 4], preds[:end_plot, 4] + error_GRU[:end_plot,4], alpha=0.3)
plt.plot(ground_truth[:, 4], label="Vicon")
plt.plot(state_t265[:,4], label='Baseline')
plt.title('y')
plt.legend(['y GRU','y Vicon','y Kalman Filter','y Baseline'])

plt.figure(6)
plt.plot(preds[:, 5], label="GRU")
#plt.fill_between(range(end_plot), preds[:end_plot, 5] - error_GRU[:end_plot, 5], preds[:end_plot, 5] + error_GRU[:end_plot,5], alpha=0.3)
plt.plot(ground_truth[:, 5], label="Vicon")
plt.title('z')
plt.legend(['z GRU','z Vicon','z Kalman Filter'])

plt.figure(7)
plt.plot(preds[:, 6], label="GRU")
#plt.fill_between(range(end_plot), preds[:end_plot, 6] - error_GRU[:end_plot, 6], preds[:end_plot, 6] + error_GRU[:end_plot,6], alpha=0.3)
plt.plot(ground_truth[:, 6], label="Vicon")
plt.title('dthx')
plt.legend(['dthx GRU','dthx Vicon','dthx Kalman Filter'])

plt.figure(8)
plt.plot(preds[:, 7], label="GRU")
#plt.fill_between(range(end_plot), preds[:end_plot, 7] - error_GRU[:end_plot, 7], preds[:end_plot, 7] + error_GRU[:end_plot,7], alpha=0.3)
plt.plot(ground_truth[:, 7], label="Vicon")
plt.title('dthy')
plt.legend(['dthy GRU','dthy Vicon','dthy Kalman Filter'])

plt.figure(9)
plt.plot(preds[:, 8], label="GRU")
#plt.fill_between(range(end_plot), preds[:end_plot, 8] - error_GRU[:end_plot, 8], preds[:end_plot, 8] + error_GRU[:end_plot,8], alpha=0.3)
plt.plot(ground_truth[:, 8], label="Vicon")
plt.title('dthz')
plt.legend(['dthz GRU','dthz Vicon','dthz Kalman Filter'])

plt.figure(10)
plt.plot(preds[:, 9], label="GRU")
#plt.fill_between(range(end_plot), preds[:end_plot, 9] - error_GRU[:end_plot, 9], preds[:end_plot, 9] + error_GRU[:end_plot,9], alpha=0.3)
plt.plot(ground_truth[:, 9], label="Vicon")
plt.title('dx')
plt.legend(['dx GRU','dx Vicon','dx Kalman Filter'])

plt.figure(11)
plt.plot(preds[:, 10], label="GRU")
#plt.fill_between(range(end_plot), preds[:end_plot, 10] - error_GRU[:end_plot, 10], preds[:end_plot, 10] + error_GRU[:end_plot,10], alpha=0.3)
plt.plot(ground_truth[:, 10], label="Vicon")
plt.plot(state_t265[:,10], label='Baseline')

plt.title('dy')
plt.legend(['dy GRU','dy Vicon','dy Kalman Filter','Baseline'])

plt.figure(12)
plt.plot(preds[:, 11], label="GRU")
#plt.fill_between(range(end_plot), preds[:end_plot, 11] - error_GRU[:end_plot, 11], preds[:end_plot, 11] + error_GRU[:end_plot,11], alpha=0.3)
plt.plot(ground_truth[:, 11], label="Vicon")
plt.title('dz')
plt.legend(['dz GRU','dz Vicon','dz Kalman Filter'])
plt.show()

dataset_save = {
    'gru': preds,
    'gru_error': [],
    'mocap': ground_truth,
    't265': state_t265
}

# Assuming you have two datasets as NumPy arrays: 'predictions' and 'ground_truth'
# Each dataset has a shape of (4440, 12)

# Calculate absolute errors for each component
absolute_errors = np.abs(preds[:end_plot,0:12] - ground_truth[:end_plot,0:12])
# Calculate the mean absolute error (MAE) for each column
mae_gru = np.mean(absolute_errors, axis=0)
# Print or use the mae and rmse arrays
print("Mean Absolute Error for each column (GRU):", mae_gru)


# Calculate absolute errors for each component
absolute_errors = np.abs(state_t265[:end_plot,0:12] - ground_truth[:end_plot,0:12])
# Calculate the mean absolute error (MAE) for each column
mae_t265 = np.mean(absolute_errors, axis=0)
# Print or use the mae and rmse arrays
print("Mean Absolute Error for each column (t265):", mae_t265)

mat_file_path = dir_path + '/OptiState/data_results/test_results_flat.mat'
# Save the datasets to the .mat file
io.savemat(mat_file_path, dataset_save)

dataset_save = {
    'error_gru': mae_gru,
    'error_t265': mae_t265
}
mat_file_path = dir_path + '/OptiState/data_results/test_results_flat_error.mat'

io.savemat(mat_file_path, dataset_save)