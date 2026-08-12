class MLP(nn.Module):
    def __init__(self, in_features=2, hidden_dim=16, out_features=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)   # W1: (16, 2)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)     # W2: (16, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, out_features)   # W3: (1, 16)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
      #  x = self.fc1(x)
      # x = self.fc2(x)
        x = self.sigmoid(self.fc3(x))   # squashes output to a (0, 1) probability
        return x

model = MLP()
