import argparse
import datetime
import pprint
import os
import sys
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

from util import Tee
from util import set_random_seed
from models import VAE1
from datasets import JPLDGIMDataset


matplotlib.use('Agg')




def plot_gims(gims, file_name):
    # fig, axs = plt.subplots(2, 2, figsize=(16, 8))
    # axs = axs.flatten()
    # for j in range(4):
    #     axs[j].imshow(samples[j, 0], cmap='gist_ncar')
    #     axs[j].axis('off')
    
    # print(f'Saving sample plot to {file_name}')
    # # colorbar
    # # fig.colorbar(axs[0].imshow(sample[0, 0], cmap='gist_ncar'), ax=axs, orientation='horizontal', fraction=0.02, pad=0.04)
    
    # plt.tight_layout()
    # plt.savefig(file_name)
    # plt.close()

    num_samples = gims.shape[0]
    # find the best grid size
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(16 * grid_size, 8 * grid_size))
    axs = axs.flatten()

    for i in range(num_samples):
        axs[i].imshow(gims[i, 0], cmap='gist_ncar')
        axs[i].axis('off')

    print(f'Saving GIM plot to {file_name}')
    # colorbar
    # fig.colorbar(axs[0].imshow(sample[0, 0], cmap='gist_ncar'), ax=axs, orientation='horizontal', fraction=0.02, pad=0.04)

    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def main():
    description = 'NASA Heliolab 2025 - Ionosphere-Thermosphere Twin, ML experiments'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory for the datasets')
    parser.add_argument('--target_dir', type=str, help='Directory to save the statistics', required=True)
    parser.add_argument('--jpld_gim_dir', type=str, default='jpld_gim_20100513-20240731', help='JPLD GIM dataset directory')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help='Mode of operation: train or test')
    parser.add_argument('--valid_proportion', type=float, default=0.15, help='Proportion of data to use for validation')
    parser.add_argument('--valid_interval', type=int, default=10, help='Interval for validation during training')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cpu', help='Device')

    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)
    log_file = os.path.join(args.target_dir, 'log.txt')

    set_random_seed(args.seed)
    device = torch.device(args.device)

    with Tee(log_file):
        print(description)
        print('Log file:', log_file)
        print('Arguments:\n{}'.format(' '.join(sys.argv[1:])))
        print('Config:')
        pprint.pprint(vars(args), depth=2, width=50)

        start_time = datetime.datetime.now()
        print('Start time: {}'.format(start_time))

        if args.mode == 'train':
            print('Training mode selected.')

            data_dir_jpld_gim = os.path.join(args.data_dir, args.jpld_gim_dir)
            dataset_jpld_gim = JPLDGIMDataset(data_dir_jpld_gim, normalize=True)
            
            valid_size = int(args.valid_proportion * len(dataset_jpld_gim))
            train_size = len(dataset_jpld_gim) - valid_size
            dataset_train, dataset_valid = random_split(dataset_jpld_gim, [train_size, valid_size])

            print('\nTrain size: {:,}'.format(len(dataset_train)))
            print('Valid size: {:,}'.format(len(dataset_valid)))

            train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            valid_loader = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


            model = VAE1(z_dim=512, sigma_vae=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

            iteration = 0
            train_losses = []
            valid_losses = []
            model = model.to(device)

            model.train()

            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print('\nNumber of parameters: {:,}\n'.format(num_params))
            
            for epoch in range(args.epochs):
                with tqdm(total=len(train_loader)) as pbar:
                    for i, batch in enumerate(train_loader):

                        jpld_gim, _ = batch
                        jpld_gim = jpld_gim.to(device)

                        optimizer.zero_grad()
                        loss = model.loss(jpld_gim)
                        loss.backward()

                        optimizer.step()

                        iteration += 1

                        train_losses.append((iteration, float(loss)))
                        pbar.set_description(f'Epoch {epoch + 1}/{args.epochs}, Loss: {loss.item():.4f}')
                        pbar.update(1)

                        # if iteration % args.valid_interval == 0:
                        #     print(f'\nValidating')
                        #     plot_file = os.path.join(args.target_dir, f'loss-{iteration}.pdf')
                        #     print(f'Saving plot to {plot_file}')
                        #     plt.figure(figsize=(10, 5))
                        #     plt.plot(*zip(*train_losses), label='Train Loss')
                        #     plt.plot(*zip(*valid_losses), label='Valid Loss')
                        #     plt.xlabel('Iteration')
                        #     plt.ylabel('Loss')
                        #     plt.title('Loss over iterations')
                        #     plt.yscale('log')
                        #     plt.grid(True)
                        #     plt.legend()
                        #     plt.savefig(plot_file)
                        #     plt.close()

                        #     # sample a batch from the VAE
                        #     model.eval()
                        #     with torch.no_grad():
                        #         sample = model.sample(n=4)
                        #         sample = JPLDGIMDataset.unnormalize(sample)
                        #         sample = sample.cpu().numpy()

                        #         sample_file = os.path.join(args.target_dir, f'sample-{iteration}.pdf')
                        #         plot_gims(sample, sample_file)

                        #         recon_original_file = os.path.join(args.target_dir, f'reconstruction-{iteration}-original.pdf')
                        #         plot_gims(jpld_gim.cpu().numpy(), recon_original_file)


                        #         # plot last minibatch of data and its reconstruction
                        #         recon, _, _ = model.forward(jpld_gim)
                        #         recon = JPLDGIMDataset.unnormalize(recon)
                        #         recon = recon.cpu().numpy()

                        #         recon_file = os.path.join(args.target_dir, f'reconstruction-{iteration}.pdf')
                        #         plot_gims(recon, recon_file)

            

        elif args.mode == 'test':
            raise NotImplementedError("Testing mode is not implemented yet.")

        end_time = datetime.datetime.now()
        print('End time: {}'.format(end_time))
        print('Total duration: {}'.format(end_time - start_time))


if __name__ == '__main__':
    main()


# Example
# python run.py --data_dir /disk2-ssd-8tb/data/2025-hl-ionosphere --mode train --target_dir ./train-1 --num_workers 4 --batch_size 4 
