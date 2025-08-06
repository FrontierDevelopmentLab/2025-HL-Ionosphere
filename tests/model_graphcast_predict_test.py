from ioncast import *
import torch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, T, C, H, W = 1, 4, 3, 3, 6
    input_grid = torch.ones(B, T, C, H, W).to(device)  # Example input grid with ones

    print(f"input_grid:\n{input_grid}\n\n")
    context_window = 2
    # prediction_window = T - context_window
    forcing_channels = None  # Assume the first channel is a forcing channel
    # T * (X + F) + F
    # [0 0 0 1 1 1 0 0 0 0] -> len(c)
    # [3,4,5]    -> len(n_forcing)
    model = IonCastGNN(
        input_dim_grid_nodes=context_window * C + (len(forcing_channels) if not forcing_channels is None else 0),  # Flattened context window + forcing channels at next time step
        output_dim_grid_nodes=C,  # Predict all channels at next time step
        mesh_level=1, 
        input_res=(3, 6), # Expects W = H*2
        device=device
    )

    # Halil it ran! Needed to update input_dim_grid_nodes

    output = model.predict(
        input_grid=input_grid,
        context_window=context_window,
        forcing_channels=forcing_channels,
        train=True
    )

    print(f"output:\n{output}\n\n")

    print("Input shape:", input_grid.shape)
    print("Output shape:", output.shape)  # Should be [B, T+1, C, H, W]


if __name__ == "__main__":
    main()