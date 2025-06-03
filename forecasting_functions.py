import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, xs, ys):
        """
        Args:
            xs (array-like): The input data (features).
            ys (array-like): The target data (labels).
        """
        self.xs = xs
        self.ys = ys

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.xs)

    def __getitem__(self, idx):
        """Generates one sample of data."""
        x = self.xs[idx]
        y = self.ys[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class torch_simple_MLP(torch.nn.Module):
    def __init__(self, hidden_size, epochs = 1000):
        super(torch_simple_MLP, self).__init__()
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.relu = torch.nn.ReLU()
        self.name = 'MLP'
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
    def predict(self, x):
        x = torch.Tensor(x)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == torch.device("cpu"):
            if torch.backends.mps.is_available():
                # self.device = torch.device("mps")
                print("using cpu because mps is slow")
                self.device = torch.device("cpu")
            
            
        self = self.to(self.device)
        x = x.to(self.device)
        prediction = self.forward(x)

        prediction = prediction.detach().cpu().numpy()
        self.device = torch.device("cpu")
        self = self.to(self.device)
        return prediction
    
    def fit(self, x, y, verbose = False, init_only = False, batch_size = 1024):
        if batch_size is None:
            batch_size = x.shape[0]
        x, y = torch.Tensor(x), torch.Tensor(y)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        self.fc1 = torch.nn.Linear(x.shape[1], self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, y.shape[1])
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        if init_only:
            return self
        
        # x, y = x.to(self.device), y.to(self.device)
        dataset = CustomDataset(x, y)
        # Create a DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == torch.device("cpu"):
            if torch.backends.mps.is_available():
                # self.device = torch.device("mps")
                print("using cpu because mps is slow")
                self.device = torch.device("cpu")

        self = self.to(self.device)
        print(f'using {self.device}')
        for epoch in tqdm(range(self.epochs)):
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                outputs = self.forward(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
            if epoch % 100 == 0 and verbose:
                print(f"Epoch: {epoch} Loss: {loss.item()}") 
        self.device = torch.device("cpu")
        self = self.to(self.device)
        return self
    

class torch_simple_RNN(torch.nn.Module):
    def __init__(self, hidden_size, num_layers=1, epochs=1000):
        super(torch_simple_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.name = 'RNN'

    def forward(self, x):
        # Handle the batch size dynamically
        batch_size = x.size(0)  # N: Number of instances
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)

        # RNN forward pass
        out, _ = self.rnn(x, h0)  # x is 3D now after reshaping in .fit()
        
        # Only take the output from the last time step
        out = self.fc(out[:, -1, :])  # Output shape: (N, H)
        return out

    def predict(self, x):
        x = torch.Tensor(x)
        # Reshape the 2D input (N, W) to 3D (N, W, 1) since it's univariate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == torch.device("cpu"):
            if torch.backends.mps.is_available():
                # self.device = torch.device("mps")        
                print("using cpu because mps is slow")
                self.device = torch.device("cpu")
        self = self.to(self.device)
        x = x.unsqueeze(2).to(self.device)
        dataset = CustomDataset(x, x)
        # Create a DataLoader
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
        
        prediction = []
        with torch.no_grad():
            for x, _ in dataloader:
                x  = x.to(self.device)
                outputs = self.forward(x) 
                prediction.append(outputs)
        prediction = torch.cat(prediction, axis = 0)
        prediction = prediction.cpu().numpy()
        self.device = torch.device("cpu")
        self = self.to(self.device)
        return prediction
    
    def fit(self, x, y, verbose=False, init_only = False, batch_size=1024):
        x, y = torch.Tensor(x), torch.Tensor(y)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        
        # Ensure the entire dataset is used as a single batch
        N = x.shape[0]  # N: Number of instances
        W = x.shape[1]  # W: Number of lagged features (sequence length)
        H = y.shape[1]  # H: Number of ahead forecast values (output size)

        # Reshape x to 3D: (N, W, 1) for univariate time series
        x = x.unsqueeze(2)

        # Define the RNN and FC layers based on input size
        input_size = 1  # Univariate input, so 1 feature per time step
        output_size = H  # Output size is H, the number of forecast steps

        # Define the RNN layer and FC layer here based on the input data
        self.rnn = torch.nn.RNN(input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = torch.nn.Linear(self.hidden_size, output_size)
        if init_only:
            return self
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

        # x, y = x.to(self.device), y.to(self.device)
        dataset = CustomDataset(x, y)
        # Create a DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == torch.device("cpu"):
            if torch.backends.mps.is_available():
                # self.device = torch.device("mps")
                print("using cpu because mps is slow")
                self.device = torch.device("cpu")
        self = self.to(self.device)
        print(f'using {self.device}')
        for epoch in tqdm(range(self.epochs)):
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
            
                outputs = self.forward(x) 
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
            
            if epoch % 100 == 0 and verbose:
                print(f"Epoch: {epoch} Loss: {loss.item()}")
        self.device = torch.device("cpu")
        self = self.to(self.device)
        return self
    
    
    
class torch_simple_LSTM(torch.nn.Module):
    def __init__(self, hidden_size, num_layers=1, epochs=1000):
        super(torch_simple_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == torch.device("cpu"):
            if torch.backends.mps.is_available():
                # self.device = torch.device("mps")
                print("using cpu because mps is slow")
                self.device = torch.device("cpu")
        self.name = 'LSTM'

    def forward(self, x):
        # Handle the batch size dynamically
        batch_size = x.size(0)  # N: Number of instances
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)  # LSTM also needs c0 for the cell state

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))  # x is 3D now after reshaping in .fit()
        
        # Only take the output from the last time step
        out = self.fc(out[:, -1, :])  # Output shape: (N, H)
        return out

    def predict(self, x):
        x = torch.Tensor(x)
        # Reshape the 2D input (N, W) to 3D (N, W, 1) since it's univariate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == torch.device("cpu"):
            if torch.backends.mps.is_available():
                 # self.device = torch.device("mps")
                print("using cpu because mps is slow")
                self.device = torch.device("cpu")
        self = self.to(self.device)
        x = x.unsqueeze(2).to(self.device)
        dataset = CustomDataset(x, x)
        # Create a DataLoader
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
        
        prediction = []
        with torch.no_grad():
            for x, _ in dataloader:
                x  = x.to(self.device)
                outputs = self.forward(x) 
                prediction.append(outputs)
        prediction = torch.cat(prediction, axis = 0)
        prediction = prediction.cpu().numpy()
        self.device = torch.device("cpu")
        self = self.to(self.device)
        return prediction
    
    def fit(self, x, y, verbose=False, init_only = False, batch_size=1024):
        x, y = torch.Tensor(x), torch.Tensor(y)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        
        # Ensure the entire dataset is used as a single batch
        N = x.shape[0]  # N: Number of instances
        W = x.shape[1]  # W: Number of lagged features (sequence length)
        H = y.shape[1]  # H: Number of ahead forecast values (output size)

        # Reshape x to 3D: (N, W, 1) for univariate time series
        x = x.unsqueeze(2)

        # Define the LSTM and FC layers based on input size
        input_size = 1  # Univariate input, so 1 feature per time step
        output_size = H  # Output size is H, the number of forecast steps

        # Define the LSTM layer and FC layer here based on the input data
        self.lstm = torch.nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = torch.nn.Linear(self.hidden_size, output_size)
        if init_only:
            return self
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

        self = self.to(self.device)
        # x, y = x.to(self.device), y.to(self.device)
        dataset = CustomDataset(x, y)
        # Create a DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        print(f'using {self.device}')
        for epoch in tqdm(range(self.epochs)):
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
            
                outputs = self.forward(x)  
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
            
            if epoch % 100 == 0 and verbose:
                print(f"Epoch: {epoch} Loss: {loss.item()}")
                
        return self
    
    
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class torch_simple_Transformer(nn.Module):
    def __init__(self, feature_size, num_layers=1, epochs=1000, nhead=4):
        super(torch_simple_Transformer, self).__init__()
        self.feature_size = feature_size
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == torch.device("cpu"):
            if torch.backends.mps.is_available():
                # self.device = torch.device("mps")
                print("using cpu because mps is slow")
                self.device = torch.device("cpu")
        self.name = 'Transformer'

        self.pos_encoder = PositionalEncoding(feature_size)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.activation = nn.ReLU()

    def forward(self, x):
        # Positional encoding
        x = self.pos_encoder(x)
        # Transformer forward pass
        x = x.to(self.device)
        self = self.to(self.device)
        out = self.transformer_encoder(x)
        # Fully connected output layer
        out = self.fc(out[:, -1, :])  # Take the last time step output
        return out

    def predict(self, x):
        x = torch.Tensor(x)
        # Reshape the 2D input (N, W) to 3D (N, W, 1) since it's univariate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == torch.device("cpu"):
            if torch.backends.mps.is_available():
                # self.device = torch.device("mps")
                print("using cpu because mps is slow")
                self.device = torch.device("cpu")
        self = self.to(self.device)
        x = x.unsqueeze(2).to(self.device)
        dataset = CustomDataset(x, x)
        # Create a DataLoader
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
        
        prediction = []
        with torch.no_grad():
            for x, _ in dataloader:
                x  = x.to(self.device)
                outputs = self.forward(x) 
                prediction.append(outputs)
        prediction = torch.cat(prediction, axis = 0)
        prediction = prediction.cpu().numpy()
        self.device = torch.device("cpu")
        self = self.to(self.device)
        return prediction

    def fit(self, x, y, verbose=False, batch_size = 1024, init_only = False):
        x, y = torch.Tensor(x), torch.Tensor(y)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        H = y.shape[1]  # H: Number of ahead forecast values (output size)
        # Reshape x to 3D: (N, W, 1) for univariate time series
        x = x.unsqueeze(2)

        self = self.to(self.device)
        x, y = x.to(self.device), y.to(self.device)
        self.fc = nn.Linear(self.feature_size, H)  # Output size is 1 for univariate
        
        if init_only:
            return self
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self = self.to(self.device)
        # x, y = x.to(self.device), y.to(self.device)
        dataset = CustomDataset(x, y)
        # Create a DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        print(f'using {self.device}')
        for epoch in tqdm(range(self.epochs)):
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = self.forward(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

            if epoch % 100 == 0 and verbose:
                print(f"Epoch: {epoch} Loss: {loss.item()}")

        return self
    
# from sklearn.ensemble import RandomForestRegressor
# class sklearn_RF(RandomForestRegressor):
#     def __init__(self, name='RF', **kwargs):
#         super().__init__(**kwargs)
#         self.name = name
        
# import xgboost as xgb

# class sklearn_XGB(xgb.XGBRegressor):
#     def __init__(self, name='XGB', **kwargs):
#         super().__init__(**kwargs)
#         self.name = name