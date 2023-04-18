import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(GRU, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.gru1 = nn.GRUCell(input_size, hidden_sizes[0])
        self.gru2 = nn.GRUCell(hidden_sizes[0], hidden_sizes[1])
        self.gru3 = nn.GRUCell(hidden_sizes[1], hidden_sizes[2])
        self.fc = nn.Linear(hidden_sizes[2], output_size)
        self.activation = nn.LeakyReLU()

    def forward(self, x, h):
        # x is input of shape (batch_size, input_size)
        # h is hidden state of shape (batch_size, hidden_size)
        h1 = self.activation(self.gru1(x, h[0]))
        h2 = self.activation(self.gru2(h1, h[1]))
        h3 = self.activation(self.gru3(h2, h[2]))
        output = self.fc(h3)
        return output, (h1, h2, h3)