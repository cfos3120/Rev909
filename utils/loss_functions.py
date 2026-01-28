import torch
import numpy as np
import socket
from dissipative_utils import sample_uniform_spherical_shell, linear_scale_dissipative_target


# HPC or Local
if socket.gethostname() == 'DESKTOP-157DQSC':
    device = torch.device('cpu')
    home_path =  "C:/Users/Noahc/Documents/USYD"
    git_path = f'{home_path}/PHD/8 - Github'
    DEBUG=True
else:
    device = torch.device('cuda')
    home_path =  "/home/n.foster"
    git_path = home_path



class RegulatorSampler():
    def __init__(self, 
                 radii, 
                 shape, 
                 scale_down_factor = 0.5, 
                 weight = 0.01, 
                 sampling_fn = sample_uniform_spherical_shell,
                 target_fn = linear_scale_dissipative_target,
                 loss_fn=torch.nn.MSELoss(reduction='mean')):
        
        self.sampling_fn = sampling_fn
        self.target_fn = target_fn
        self.loss_function = loss_fn
        self.weight = weight
        self.scale_down_factor = scale_down_factor
        
        if all(isinstance(x, np.ndarray) for x in radii):
            assert len(radii[0]) == shape[-1]
            print('Regularizer enforces per channel')
            self.split_channels = True 
            radii = [radii[0][None,...], radii[1][None,...]]
            radii = np.concatenate(radii, axis = 0)
        else:
            self.split_channels = False 

        # get shape((S, S, 1))
        self.shape = shape
        # get radii
        self.radii = radii

    def get_input(self, batch_s):
        if self.split_channels:
            input_sample = []
            for i in range(self.shape[-1]):
                input_sample.append(torch.tensor(self.sampling_fn(batch_s, self.radii[...,i], self.shape[:-1]+ (1,)), dtype=torch.float))
            input_sample = torch.concatenate(input_sample, dim=-1)
        else:
            input_sample = torch.tensor(self.sampling_fn(batch_s, self.radii, self.shape), dtype=torch.float)
        return input_sample
    
    def get_target(self, x_diss):
        return self.target_fn(x_diss, self.scale_down_factor)
    
    def loss_fn(self, out, y):
        return self.weight * self.loss_function(out, y)