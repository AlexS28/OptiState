import torch
import torch.nn as nn
import torch.nn as nn
import torch

# Fully connected neural network with one hidden layer
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device, evaluate=False, use_sigmoid=True):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        #self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)
        self.device = device
        # or:
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.6)  # Add a dropout layer
        self.evaluate = evaluate
        #self.softplus = nn.Softplus().to(self.device)
        #self.use_softplus = use_softplus
        self.use_sigmoid = use_sigmoid
        self.sigmoid = nn.Sigmoid()  # Add the sigmoid activation
    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # x: (n, 28, 28), h0: (2, n, 128)

        # Forward propagate RNN
        out, _ = self.gru(x, h0)
        if not self.evaluate:
            out = self.dropout(out)
        # or:
        # out, _ = self.lstm(x, (h0,c0))

        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n, 28, 128)

        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (n, 128)
        out = self.fc(out)
        if self.use_sigmoid:
            #out = self.softplus(out)
            out = self.sigmoid(out)  # Apply the sigmoid activation
        return out
