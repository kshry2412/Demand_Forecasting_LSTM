import torch

import torch.nn as nn



class LSTMModel(nn.Module):

    def __init__(self, input_size, hidden_layer_size, output_size):

        super(LSTMModel, self).__init__()



        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=2, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(hidden_layer_size * 2, output_size)  # *2 because of bidirectional  

        self.dropout = nn.Dropout(0.1)



    def forward(self, x):

        out, (hn, cn) = self.lstm(x)

        out = self.dropout(out[:, -1, :])  # Get the last time-step output

        out = self.fc(out)  # Final output layer

        return out