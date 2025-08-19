import torch
import torch.nn as nn


class IonCastPersistence(nn.Module):
    """
    Persistence model for ionosphere forecasting.
    
    This is the ultimate baseline model that simply repeats the last frame
    as the prediction for the next frame. No learnable parameters, no training
    required - just copies the input.
    """
    
    def __init__(self, input_channels=17, output_channels=17, context_window=4, name=None):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.context_window = context_window
        
        # Add a dummy parameter that's never actually used
        # This allows optimizers to work without code changes
        self.dummy_param = nn.Parameter(torch.tensor(0.0))
        
        if name is None:
            name = 'IonCastPersistence'
        self.name = name

        print(self.name)
        print('  input_channels:', input_channels)
        print('  output_channels:', output_channels)
        print('  context_window:', context_window, '(only uses last frame)')
        print('  parameters: 1 (dummy parameter for optimizer compatibility)')

    def forward(self, x):
        """
        Forward pass using persistence (repeat last frame).
        
        Args:
            x: Input tensor of shape (B, T, C, H, W) - context window
            
        Returns:
            Last frame repeated as prediction of shape (B, 1, C, H, W)
        """
        B, T, C, H, W = x.size()
        
        # Simply take the last frame from the context
        last_frame = x[:, -1, :, :, :]  # (B, C, H, W)
        
        # Add the dummy parameter multiplied by zero to maintain gradient flow
        # This doesn't change the result but allows gradients to flow
        last_frame = last_frame + self.dummy_param * 0.0
        
        # Add time dimension
        pred = last_frame.unsqueeze(1)  # (B, 1, C, H, W)
        
        return pred

    def loss(self, data, jpld_channel_index=0, jpld_weight=1.0):
        """
        Compute loss for sequence prediction using persistence.
        
        Args:
            data: Input data of shape (B, T, C, H, W)
            jpld_channel_index: Index of JPLD channel for weighted loss
            jpld_weight: Weight for JPLD channel
            
        Returns:
            tuple: (loss, rmse, jpld_rmse)
        """
        B, T, C, H, W = data.size()

        # For seq-to-seq, input is steps 0 to T-1, target is steps 1 to T
        data_input = data[:, :-1, :, :, :]
        data_target = data[:, -1, :, :, :].unsqueeze(1)

        # For persistence, prediction is the forward pass output
        data_predict = self.forward(data_input)
        
        # Compute weighted loss
        elementwise_loss = nn.functional.mse_loss(data_predict, data_target, reduction='none')

        weights = torch.ones_like(data_target[0, 0, :, :, :]).unsqueeze(0).unsqueeze(0)  # Shape (1, 1, C, H, W)
        weights[:, :, jpld_channel_index, :, :] = jpld_weight

        loss = torch.mean(elementwise_loss * weights)

        # Compute metrics
        with torch.no_grad():
            rmse = torch.sqrt(nn.functional.mse_loss(data_predict, data_target, reduction='mean'))
            jpld_rmse = torch.sqrt(nn.functional.mse_loss(
                data_predict[:, :, jpld_channel_index, :, :], 
                data_target[:, :, jpld_channel_index, :, :], 
                reduction='mean'
            ))
            
        return loss, rmse, jpld_rmse
    
    def predict(self, data_context, prediction_window=4):
        """
        Forecast future timesteps given context using persistence.
        
        Args:
            data_context: Context data of shape (B, T_context, C, H, W)
            prediction_window: Number of future steps to predict
            
        Returns:
            Predictions of shape (B, prediction_window, C, H, W)
        """
        B, T_context, C, H, W = data_context.size()
        
        # Start prediction from the last frame of the context
        current_frame = data_context[:, -1, :, :, :].unsqueeze(1)  # (B, 1, C, H, W)
        predictions = []

        for _ in range(prediction_window):
            # Use forward method to get the next frame prediction
            next_frame = self.forward(current_frame)  # (B, 1, C, H, W)
            predictions.append(next_frame)
            # The predicted frame becomes the input for the next step
            current_frame = next_frame

        prediction = torch.cat(predictions, dim=1)  # (B, prediction_window, C, H, W)
        return prediction
