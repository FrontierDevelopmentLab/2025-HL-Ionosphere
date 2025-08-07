import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from tqdm import tqdm
import imageio
from matplotlib import animation

def plot_global_ionosphere_map(ax, image, cmap='jet', vmin=None, vmax=None, title=None):
    """
    Plots a 180x360 global ionosphere image on a given Cartopy axes.
    
    Parameters:
        ax (matplotlib.axes._subplots.AxesSubplot): Axes with a Cartopy projection.
        image (np.ndarray): 2D numpy array with shape (180, 360), representing lat [-90, 90], lon [-180, 180].
        cmap (str): Colormap to use for imshow.
        vmin (float): Minimum value for colormap normalization.
        vmax (float): Maximum value for colormap normalization.
        title (str): Title for the plot.
    """
    if image.shape != (180, 360):
        raise ValueError("Input image must have shape (180, 360), but got shape {}.".format(image.shape))

    im = ax.imshow(
        image,
        extent=[-180, 180, -90, 90],
        origin='upper',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree()
    )
    
    ax.coastlines()
    if title is not None:
        ax.set_title(title, fontsize=12, loc='left')

    return im


def save_gim_plot(gim, file_name, cmap='jet', vmin=None, vmax=None, title=None):
    """
    Plots a single 180x360 global ionosphere image using GridSpec,
    with a colorbar aligned to the full height of the imshow map.
    """
    print(f'Plotting GIM to {file_name}')
    
    if gim.shape != (180, 360):
        raise ValueError("Input image must have shape (180, 360) corresponding to lat [-90, 90], lon [-180, 180].")
    
    fig = plt.figure(figsize=(10, 5))
    
    # GridSpec: one row, two columns
    gs = fig.add_gridspec(
        1, 2, width_ratios=[20, 1], wspace=0.05,
        left=0.05, right=0.98, top=0.9, bottom=0.1
    )
    
    # Main plot
    ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    im = plot_global_ionosphere_map(ax, gim, cmap=cmap, vmin=vmin, vmax=vmax, title=title)
    
    # Colorbar axis — NOT a projection axis
    cbar_ax = fig.add_subplot(gs[0, 1])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("TEC (TECU)")
    
    plt.savefig(file_name, dpi=150, bbox_inches='tight')
    plt.close()

# Save a sequence of GIM images as a video, exactly the same as save_gim_plot but for a sequence of images
def save_gim_video(gim_sequence, file_name, cmap='jet', vmin=None, vmax=None, titles=None, fps=2):
    # gim_sequence has shape (num_frames, 180, 360)
    print(f'Saving GIM video to {file_name}')

    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[20, 1], wspace=0.05, left=0.05, right=0.92, top=0.9, bottom=0.1)
    ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    cbar_ax = fig.add_subplot(gs[0, 1])
    
    # Initialize with first frame
    im = plot_global_ionosphere_map(ax, gim_sequence[0], cmap=cmap, vmin=vmin, vmax=vmax, 
                                   title=titles[0] if titles else None)
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("TEC (TECU)")

    def update(frame):
        # Update the image data instead of clearing
        new_im = plot_global_ionosphere_map(ax, gim_sequence[frame], cmap=cmap, vmin=vmin, vmax=vmax, 
                                           title=titles[frame] if titles else None)
        return [new_im]

    ani = animation.FuncAnimation(fig, update, frames=len(gim_sequence), blit=False, 
                                 interval=1000/fps, repeat=False)
    ani.save(file_name, dpi=150, writer='ffmpeg', extra_args=['-pix_fmt', 'yuv420p'])
    plt.close()


def save_gim_video_comparison(gim_sequence_top, gim_sequence_bottom, file_name, cmap='jet', vmin=None, vmax=None, 
                                       titles_top=None, titles_bottom=None, fps=2, max_frames=None, cbar_label='TEC (TECU)', fig_title=None):
    """
    Pre-render all frames to avoid memory accumulation during animation.
    Now includes colorbars in each frame.
    """
    # Ensure both sequences have the same length
    if len(gim_sequence_top) != len(gim_sequence_bottom):
        raise ValueError(f"Sequences must have same length: {len(gim_sequence_top)} vs {len(gim_sequence_bottom)}")
    
    if max_frames is not None:
        if max_frames <= 0 or max_frames > len(gim_sequence_top):
            raise ValueError(f"max_frames must be between 1 and {len(gim_sequence_top)}")
        gim_sequence_top = gim_sequence_top[:max_frames]
        gim_sequence_bottom = gim_sequence_bottom[:max_frames]
        if titles_top:
            titles_top = titles_top[:max_frames]
        if titles_bottom:
            titles_bottom = titles_bottom[:max_frames]

    print(f'Saving GIM video to {file_name}')
    
    # Pre-render all frames as numpy arrays
    frames = []
    for i in tqdm(range(len(gim_sequence_top)), desc="Rendering frames"):
        # Create temporary figure for this frame with colorbars
        fig_temp = plt.figure(figsize=(10.88, 10.88))
        if fig_title:
            fig_temp.suptitle(fig_title, fontsize=12)
        gs = fig_temp.add_gridspec(2, 2, width_ratios=[20, 1], height_ratios=[1, 1], 
                                  wspace=0.05, hspace=0.15, left=0.05, right=0.92, top=0.92, bottom=0.05)
        
        # Plot frame - maps
        ax_top = fig_temp.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        ax_bottom = fig_temp.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
        
        # Plot frame - colorbar axes
        cbar_ax_top = fig_temp.add_subplot(gs[0, 1])
        cbar_ax_bottom = fig_temp.add_subplot(gs[1, 1])
        
        # Create the top map and get its image object. This will use the function's vmin/vmax or auto-scale.
        im_top = plot_global_ionosphere_map(ax_top, gim_sequence_top[i], cmap=cmap, vmin=vmin, vmax=vmax,
                                           title=titles_top[i] if titles_top else None)
        
        # Get the effective color limits from the top plot to ensure the bottom plot uses the exact same scale.
        vmin_actual, vmax_actual = im_top.get_clim()
        
        # Create the bottom map using the same color limits as the top map.
        im_bottom = plot_global_ionosphere_map(ax_bottom, gim_sequence_bottom[i], cmap=cmap, vmin=vmin_actual, vmax=vmax_actual,
                                              title=titles_bottom[i] if titles_bottom else None)
        
        # Add colorbars. They will now be identical.
        cbar_top = fig_temp.colorbar(im_top, cax=cbar_ax_top)
        cbar_top.set_label(cbar_label)

        cbar_bottom = fig_temp.colorbar(im_bottom, cax=cbar_ax_bottom)
        cbar_bottom.set_label(cbar_label)

        # Convert to array - fix deprecation warning
        fig_temp.canvas.draw()
        frame_array = np.frombuffer(fig_temp.canvas.buffer_rgba(), dtype=np.uint8)
        frame_array = frame_array.reshape(fig_temp.canvas.get_width_height()[::-1] + (4,))
        # Convert RGBA to RGB
        frame_array = frame_array[:, :, :3]
        frames.append(frame_array)
        
        plt.close(fig_temp)
    
    with imageio.get_writer(file_name, format='mp4', fps=fps, codec='libx264', 
                       output_params=['-pix_fmt', 'yuv420p', '-loglevel', 'error']) as writer:
        for frame in tqdm(frames, desc="Writing video   "):
            writer.append_data(frame)