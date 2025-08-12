import torch
import torch.nn as nn


def get_encoder(in_channels=1, base_channels=16):
        return nn.Sequential(
            # Input: (B, C, 180, 360)
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1, padding_mode='circular'), # (B, bc, 90, 180)
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1, padding_mode='circular'), # (B, 2*bc, 45, 90)
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1, padding_mode='circular'), # (B, 4*bc, 23, 45)
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(),
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1, padding_mode='circular'), # (B, 8*bc, 12, 23)
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(),
            nn.Conv2d(base_channels * 8, base_channels * 16, 3, stride=2, padding=1, padding_mode='circular'), # (B, 16*bc, 6, 12)
            nn.BatchNorm2d(base_channels * 16),
            nn.LeakyReLU(),
            nn.Flatten()
            )

def get_decoder(out_channels=1, base_channels=16):
    return nn.Sequential(
        nn.Unflatten(1, (base_channels * 16, 6, 12)),

        nn.Upsample(size=(12, 23), mode='bilinear', align_corners=False),
        nn.Conv2d(base_channels * 16, base_channels * 8, 3, stride=1, padding=1, padding_mode='circular'),
        nn.BatchNorm2d(base_channels * 8),
        nn.LeakyReLU(),

        nn.Upsample(size=(23, 45), mode='bilinear', align_corners=False),
        nn.Conv2d(base_channels * 8, base_channels * 4, 3, stride=1, padding=1, padding_mode='circular'),
        nn.BatchNorm2d(base_channels * 4),
        nn.LeakyReLU(),

        nn.Upsample(size=(45, 90), mode='bilinear', align_corners=False),
        nn.Conv2d(base_channels * 4, base_channels * 2, 3, stride=1, padding=1, padding_mode='circular'),
        nn.BatchNorm2d(base_channels * 2),
        nn.LeakyReLU(),

        nn.Upsample(size=(90, 180), mode='bilinear', align_corners=False),
        nn.Conv2d(base_channels * 2, base_channels, 3, stride=1, padding=1, padding_mode='circular'),
        nn.BatchNorm2d(base_channels),
        nn.LeakyReLU(),

        nn.Upsample(size=(180, 360), mode='bilinear', align_corners=False),
        nn.Conv2d(base_channels, out_channels, 3, stride=1, padding=1, padding_mode='circular'),
    )


class IonCastLSTM(nn.Module):
    def __init__(self, input_channels=17, output_channels=17, base_channels=8, lstm_dim=1024, num_layers=2, context_window=4, dropout=0.25):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.base_channels = base_channels
        self.lstm_dim = lstm_dim
        self.num_layers = num_layers
        self.context_window = context_window
        self.dropout = dropout

        self.encoder = get_encoder(input_channels, base_channels)
        self.decoder = get_decoder(output_channels, base_channels)

        image_size = (180, 360)
        dummy_input = torch.zeros((1, input_channels, *image_size))
        embedding_dim = self.encoder(dummy_input).size(1)

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(lstm_dim, embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.hidden = None

        print('IonCastLSTM')
        print('  input_channels:', input_channels)
        print('  output_channels:', output_channels)
        print('  base_channels:', base_channels)
        print('  embedding_dim:', embedding_dim)
        print('  lstm_dim:', lstm_dim)

    def init(self, batch_size=1):
        h = torch.zeros(self.num_layers, batch_size, self.lstm_dim)
        c = torch.zeros(self.num_layers, batch_size, self.lstm_dim)
        device = next(self.parameters()).device
        h = h.to(device)
        c = c.to(device)
        self.hidden = (h, c)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.reshape(B * T, C, H, W)
        x = self.encoder(x)
        x = x.view(B, T, -1)
        x = self.dropout1(x)
        x = torch.relu(x)
        x, self.hidden = self.lstm(x, self.hidden)
        x = self.dropout1(x)
        x = torch.relu(x)
        x = self.fc1(x)
        x = x.view(B * T, -1)
        x = self.dropout1(x)
        x = torch.relu(x)
        x = self.decoder(x)
        x = x.view(B, T, self.output_channels, H, W)
        return x

    def loss(self, data, jpld_channel_index=0, jpld_weight=1.0):
        # data shape (B, T, C, H, W)
        (B, T, C, H, W) = data.size()

        # For seq-to-seq, input is steps 0 to T-1, target is steps 1 to T
        data_input = data[:, :-1, :, :, :]
        data_target = data[:, 1:, :, :, :]

        self.init(batch_size=B)
        data_predict = self.forward(data_input)
        data_predict = data_predict.view(B, T - 1, C, H, W)
        
        elementwise_loss = nn.functional.mse_loss(data_predict, data_target, reduction='none')

        weights = torch.ones_like(data_target[0, 0, :, :, :]).unsqueeze(0).unsqueeze(0) # Shape (1, 1, C, H, W)
        weights[:, :, jpld_channel_index, :, :] = jpld_weight

        loss = torch.mean(elementwise_loss * weights)

        with torch.no_grad():
            rmse = torch.sqrt(nn.functional.mse_loss(data_predict, data_target, reduction='mean'))
            jpld_rmse = torch.sqrt(nn.functional.mse_loss(data_predict[:, :, jpld_channel_index, :, :], 
                                                          data_target[:, :, jpld_channel_index, :, :], reduction='mean'))
            
        return loss, rmse, jpld_rmse
    
    def predict(self, data_context, prediction_window=4):
        """ Forecasts the next time steps given the context window. """
        # data_context shape: (B, T_context, C, H, W)
        B, T_context, C, H, W = data_context.size()
        
        self.init(batch_size=B)
        context_input = data_context
        context_output = self.forward(context_input)
        x = context_output[:, -1, :, :, :].unsqueeze(1)  # Last time step output
        prediction = []
        for _ in range(prediction_window):
            prediction.append(x)
            x = self.forward(x)
        prediction = torch.cat(prediction, dim=1)  # Concatenate along time dimension
        return prediction