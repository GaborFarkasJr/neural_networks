import torch

class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers):
        super(LSTM, self).__init__()
        
        # Setting attributes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        
        # Creating LSTM
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.output_layer = torch.nn.Linear(hidden_size, num_classes) # Probability vector of the classes
        
    def forward(self, x):
        batch_size = x.size(0)
        hidden_states = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        cell_states = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        out, _ = self.lstm(x, (hidden_states, cell_states))
        out = self.output_layer(out[:, -1, :])
        return out