from torch import nn

class FrimageNet(nn.Module):
    def __init__(self, feature_size, num_layers=2, num_hidden_nodes=512, device='cuda'):
        super(FrimageNet, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.num_hidden_nodes = num_hidden_nodes

        # input dim is 167, output 200
        self.lstm = nn.LSTM(feature_size, num_hidden_nodes,
                            batch_first=True, num_layers=num_layers)
        # fully connected
        self.fc1 = nn.Linear(num_hidden_nodes, num_hidden_nodes)
        self.act = nn.Sigmoid()
        self.fc2 = nn.Linear(num_hidden_nodes, 2)
        self.softmax = nn.Softmax()

    def forward(self, x, hidden):
#         print(x.device, hidden[0].device)
        y, hidden = self.lstm(x, hidden)    # returns the two outputs
        y = y[:, -1, :]  # get only the last output
        y = self.fc1(y)
        y = self.fc2(y)
        y = F.softmax(y, dim=1)

        return y, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).datac
        hidden = (weight.new(self.num_layers, batch_size, self.num_hidden_nodes).zero_().to(self.device),
                  weight.new(self.num_layers, batch_size, self.num_hidden_nodes).zero_().to(self.device))
        return hidden