import torch
import torch.nn as nn


class IonCastLinear(nn.Module):
    """
    Ultra-simple baseline model for ionosphere forecasting.
    
    This model just takes the last frame and applies a simple linear transformation
    to predict the next frame. No context window, minimal parameters.
    """
    
    def __init__(self, input_channels=17, output_channels=17, context_window=4, name=None):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.context_window = context_window  # Keep for API compatibility, but we only use last frame
        
        # Ultra-simple: just a channel-wise linear transformation
        self.linear = nn.Linear(input_channels, output_channels)

        if name is None:
            name = 'IonCastLinear'
        self.name = name

        print(self.name)
        print('  input_channels:', input_channels)
        print('  output_channels:', output_channels)
        print('  context_window:', context_window, '(only uses last frame)')
        print('  parameters:', input_channels * output_channels + output_channels)

    def forward(self, x):
        """
        Forward pass using only the last frame.
        
        Args:
            x: Input tensor of shape (B, T, C, H, W) - context window
            
        Returns:
            Predicted next frame of shape (B, 1, C, H, W)
        """
        B, T, C, H, W = x.size()
        
        # Use only the last frame from the context
        last_frame = x[:, -1, :, :, :]  # (B, C, H, W)
        
        # Apply channel-wise linear transformation across spatial locations
        # Reshape to apply linear layer channel-wise
        last_frame_flat = last_frame.permute(0, 2, 3, 1)  # (B, H, W, C)
        pred_flat = self.linear(last_frame_flat)  # (B, H, W, output_C)
        pred = pred_flat.permute(0, 3, 1, 2)  # (B, output_C, H, W)
        
        # Add time dimension
        pred = pred.unsqueeze(1)  # (B, 1, C, H, W)
        
        return pred

    def loss(self, data, jpld_channel_index=0, jpld_weight=1.0):
        """
        Compute loss for sequence prediction.
        
        Args:
            data: Input data of shape (B, T, C, H, W)
            jpld_channel_index: Index of JPLD channel for weighted loss
            jpld_weight: Weight for JPLD channel
            
        Returns:
            tuple: (loss, rmse, jpld_rmse)
        """
        B, T, C, H, W = data.size()

        # For training, use each frame to predict the next frame
        all_predictions = []
        all_targets = []
        
        # Iterate through the sequence
        for t in range(T - 1):
            # Use frame t to predict frame t+1
            context = data[:, t:t+1, :, :, :]  # (B, 1, C, H, W)
            target = data[:, t+1:t+2, :, :, :]  # (B, 1, C, H, W)
            
            # Predict the next frame (residual)
            residual_pred = self.forward(context)  # (B, 1, C, H, W)
            
            # Add residual to current frame to get absolute prediction
            frame_pred = context + residual_pred
            
            all_predictions.append(frame_pred)
            all_targets.append(target)
        
        if len(all_predictions) == 0:
            # Not enough frames, return zero loss
            loss = torch.tensor(0.0, device=data.device, requires_grad=True)
            rmse = torch.tensor(0.0, device=data.device)
            jpld_rmse = torch.tensor(0.0, device=data.device)
            return loss, rmse, jpld_rmse
        
        # Concatenate all predictions and targets
        data_predict = torch.cat(all_predictions, dim=1)  # (B, T-1, C, H, W)
        data_target = torch.cat(all_targets, dim=1)  # (B, T-1, C, H, W)
        
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
        Forecast future timesteps given context using autoregressive prediction.
        
        Args:
            data_context: Context data of shape (B, T_context, C, H, W)
            prediction_window: Number of future steps to predict
            
        Returns:
            Predictions of shape (B, prediction_window, C, H, W)
        """
        B, T_context, C, H, W = data_context.size()
        
        # Start with the context
        current_sequence = data_context
        predictions = []
        
        for _ in range(prediction_window):
            # Use only the last frame to predict the next frame
            last_frame = current_sequence[:, -1:, :, :, :]  # (B, 1, C, H, W)
            
            # Predict residual for the next frame
            residual = self.forward(last_frame)  # (B, 1, C, H, W)
            
            # Add residual to the last frame to get the next frame
            next_frame = last_frame + residual
            
            predictions.append(next_frame)
            
            # Update the sequence for the next iteration
            current_sequence = torch.cat([current_sequence, next_frame], dim=1)
        
        # Concatenate all predictions
        prediction = torch.cat(predictions, dim=1)  # (B, prediction_window, C, H, W)
        return prediction
