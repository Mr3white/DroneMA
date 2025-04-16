import torch.nn as nn

class R2DGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model_type='gru'):
        super().__init__()
        
        if model_type == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        elif model_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        elif model_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
            
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out)
