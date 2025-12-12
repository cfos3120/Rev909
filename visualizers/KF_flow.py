import numpy as np
import matplotlib.pyplot as plt
import imageio
import random

def plot_evaluation_gif(out,y, n=10):
    if len(out.shape) == 3: out = out.unsqueeze(-1)
    if len(y.shape) == 3: y = y.unsqueeze(-1)

    # Choose 10 random frames
    #N = out.shape[0]
    #random_indices = random.sample(range(N), n)

    frames = []

    for idx in range(out.shape[0]):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Squeeze C if it's 1
        b_img = out[idx, :, :, 0]
        a_img = y[idx, :, :, 0]
        error_img = a_img - b_img
        
        # Heatmaps
        im0 = axes[0].imshow(a_img, cmap='viridis', aspect='equal')
        im1 = axes[1].imshow(b_img, cmap='viridis', aspect='equal')
        im2 = axes[2].imshow(error_img, cmap='plasma', aspect='equal')
        
        # Titles
        axes[0].set_title('Dataset')
        axes[1].set_title('Prediction')
        axes[2].set_title('Error')
        
        # Remove axes ticks/labels
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Add horizontal colorbars below each subplot
        cbar0 = fig.colorbar(im0, ax=axes[0], orientation='horizontal', fraction=0.046, pad=0.1)
        cbar1 = fig.colorbar(im1, ax=axes[1], orientation='horizontal', fraction=0.046, pad=0.1)
        cbar2 = fig.colorbar(im2, ax=axes[2], orientation='horizontal', fraction=0.046, pad=0.1)
        
        # Render figure to numpy array
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        
        plt.close(fig)

    return frames

if __name__ == '__main__':
    import os
    import yaml
    import sys
    parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(parent)
    from utils.dataloader import KF_flow_data

    with open(r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\Rev909\configs\KF_flow.yaml', 'r') as file:
        config = yaml.safe_load(file)

    dataset_name    = config['dataset_params']['dataset_name']
    dataset_split   = config['dataset_params']['split']
    dataset_sub     = config['dataset_params']['sub']
    dataset_T_in    = config['dataset_params']['T_in']
    dataset_T_out   = config['dataset_params']['T_out']
    
    data_path = r"C:\Users\Noahc\Documents\USYD\PHD\0 - Work Space\Markov Studies"
    dataset_path = f'{data_path}/{dataset_name}'

    train_loader, test_loader, S, max_norm = KF_flow_data(dataset_path, 
                                                          dataset_split, 
                                                          batch_size=10, 
                                                          sub=dataset_sub, 
                                                          T_in=dataset_T_in, 
                                                          T_out=dataset_T_out
                                                          )
    
    batch = next(iter(test_loader))

    x,y = batch
    print(y.shape)

    frames = plot_evaluation_gif(y+np.random.uniform(-0.1, 0.1, size=y.shape),y, n=10)
    imageio.mimsave('dataset_comparison_test.gif', frames, duration=0.5)
