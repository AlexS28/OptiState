import torch
import torch.nn as nn

class CustomNeuralNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(CustomNeuralNetwork, self).__init__()
        self.input_layer = nn.Linear(num_inputs, 256)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 32)
        ])
        self.output_layer = nn.Linear(32, num_outputs)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        outputs = self.output_layer(x)
        return outputs

# Specify the number of inputs and outputs
num_inputs = 10
num_outputs = 5

# Create an instance of the custom neural network
model = CustomNeuralNetwork(num_inputs, num_outputs)

# Print model architecture
print(model)