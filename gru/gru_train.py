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
import shutil
from scipy import io
from settings import INITIAL_PARAMS

# specify the dataset number to train on
dataset_train_number = [1]
total_number_datasets = [1,2]
# specify number of models to train
num_models = 1
# we train both the model for state output, and afterwards, model for the covariances
training_percentage = 0.8
# Hyper-parameters for state estimate
num_outputs = 12
if INITIAL_PARAMS.USE_VISION:
    input_size = 30+128
else:
    input_size = 30
sequence_length = 10
hidden_size = 128+64
num_layers = 4
num_epochs = 3000
batch_size = 64
learning_rate = 0.0001

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using Device: {device}')

with open(dir_path + '/OptiState/data_collection/trajectories/rnn_data.pkl', 'rb') as f:
    data_collection = pickle.load(f)

state_KF_init = []
state_VICON_init = []

image_iterate = 1
for i in range(len(total_number_datasets)):
    cur_dataset = total_number_datasets[i]
    state_KF_init.extend(data_collection[cur_dataset]['state_INPUT'])
    state_VICON_init.extend(data_collection[cur_dataset]['state_MOCAP'])

# Convert your list of lists to a numpy array for easier processing
state_KF_init_array = np.array(state_KF_init)
# Calculate the minimum and maximum values for each component
min_vals = np.min(state_KF_init_array, axis=0)
max_vals = np.max(state_KF_init_array, axis=0)
# Perform Min-Max scaling for each component
normalized_state_KF_init = (state_KF_init_array - min_vals) / (max_vals - min_vals)
# The 'normalized_state_KF_init' array now contains the normalized values
# Each row corresponds to a list in the original 'state_KF_init'
# Convert your list of lists to a numpy array for easier processing
state_VICON_init_array = np.array(state_VICON_init)
# Calculate the minimum and maximum values for each component
min_vals_VIC = np.min(state_VICON_init_array, axis=0)
max_vals_VIC = np.max(state_VICON_init_array, axis=0)
# Perform Min-Max scaling for each component
normalized_state_VICON_init = (state_VICON_init_array - min_vals_VIC) / (max_vals_VIC - min_vals_VIC)

# Combine min_vals and max_vals into a dictionary
scaling_params = {
    'min_vals_KF': min_vals,
    'max_vals_KF': max_vals,
    'min_vals_VIC': min_vals_VIC,
    'max_vals_VIC': max_vals_VIC,
}

state_KF_init = []
state_VICON_init = []
images_folder_train = dir_path + '/OptiState/data_collection/trajectories/saved_images/saved_images_traj_train'

# Check if the folder exists
if os.path.exists(images_folder_train):
    # Remove the folder and its contents
    shutil.rmtree(images_folder_train)

if not os.path.exists(images_folder_train):
    os.makedirs(images_folder_train)

for i in range(len(dataset_train_number)):
    cur_dataset = dataset_train_number[i]
    state_KF_init.extend(data_collection[cur_dataset]['state_INPUT'])
    state_VICON_init.extend(data_collection[cur_dataset]['state_MOCAP'])


    if INITIAL_PARAMS.USE_VISION:
        cur_image_directory = dir_path + f'/OptiState/data_collection/trajectories/saved_images/saved_images_traj_{cur_dataset}'
        image_files = [file for file in os.listdir(cur_image_directory) if file.lower().endswith('.png')]
        # Rename and copy images to the training folder
        for index, image_file in enumerate(image_files):
            new_filename = f"image_{cur_dataset}_{image_iterate}.png"
            source_path = os.path.join(cur_image_directory, image_file)
            destination_path = os.path.join(images_folder_train, new_filename)
            shutil.copyfile(source_path, destination_path)
            image_iterate+=1


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


# Specify the file path to save the pickle file
pickle_file_path = dir_path  + '/OptiState/data_collection/trajectories/scaling_params.pkl'

# Save the dictionary as a pickle file
with open(pickle_file_path, 'wb') as pickle_file:
    pickle.dump(scaling_params, pickle_file)

print("Scaling parameters saved as a pickle file:", pickle_file_path)

# Define the transformation to apply to the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 64x64
    transforms.Grayscale(num_output_channels=1),  # Convert to greyscale
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
])

image_list = []
import re

def numerical_sort(value):
    # Extract the numerical part of the filename
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else -1

if INITIAL_PARAMS.USE_VISION:
    load_model_name = dir_path  + "/OptiState/transformer/trans_encoder_model"
    model = Transformer_Autoencoder()
    model.load_state_dict(torch.load(load_model_name))
    model.eval()

    # load the autoencoder and output encoder values
    encoder_output = []
    cur_image_iterate = 1
    for filename in sorted(os.listdir(images_folder_train), key=numerical_sort):
        if filename.endswith(".png"):
            image_path = os.path.join(images_folder_train, filename)
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                img = transform(img).unsqueeze(0)
                with torch.no_grad():
                    encoded = model.forward_encoder(img)
                    encoded_1d = encoded.view(-1)
                    encoded_1d_list = encoded_1d.numpy().tolist()
                    encoder_output.append(encoded_1d_list)
                    print(f'Calculating encoder values on image number: {cur_image_iterate}/{image_iterate}')
                    cur_image_iterate += 1

    # save encoder output as pickle file so we do not need to the above operation every time
    file_encoder_path = dir_path + "/OptiState/data_collection/trajectories/encoder_output_training.pkl"
    # Serialize and save the data to the file
    with open(file_encoder_path, 'wb') as file:
        pickle.dump(encoder_output, file)
    print("Encoder output saved to", file_encoder_path)

    # add the encoder output to the Kalman filter states for input to the GRU
    for i in range(len(state_KF_init)):
        state_KF_init[i].extend(encoder_output[i])

# reshape data to have sequences of length sequence_length
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
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train, num_test])
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # TODO: BELOW IS NO RANDOM SPLIT
    # specify the training and testing percentages
    #testing_percentage = 1 - training_percentage

    # calculate the number of samples for training and testing
    #num_samples = len(dataset)
    #num_train = int(training_percentage * num_samples)
    #num_test = num_samples - num_train

    # create indices for training and testing samples
    #train_indices = list(range(num_train))
    #test_indices = list(range(num_train, num_samples))

    # create Subset objects using the indices
    #train_dataset = torch.utils.data.Subset(dataset, train_indices)
    #test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # create DataLoader objects for training and testing
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

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
            #if (i + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.16f}')

    if INITIAL_PARAMS.USE_VISION:
        # save your model
        torch.save(model.state_dict(), dir_path + f'/OptiState/gru/gru_models/model{cur_model}_vision.pth')
    else:
        torch.save(model.state_dict(), dir_path + f'/OptiState/gru/gru_models/model{cur_model}.pth')
    cur_model += 1


plt.figure(1)
plt.plot(loss_list_training)
plt.title('Loss on training set: State')
plt.xlabel('Data points')
plt.ylabel('Loss')
plt.show()

dataset_save = {
    'loss': loss_list_training,
}
mat_file_path = dir_path + '/OptiState/data_results/gru_1_loss_results.mat'
# Save the datasets to the .mat file
io.savemat(mat_file_path, dataset_save)