import torch
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser("TRAIN THE FLASH GNOT TRANSFORMER")
    parser.add_argument('--config', type=str, help="json configuration file")
    parser.add_argument('--wandb', type=str, help="wandb mode", choices=['online','offline','disabled'], default='offline')
    parser.add_argument('--sweep', type=str, help="sweep json file")
    args = parser.parse_args()
    return args

class HsLoss_real(object):
    def __init__(self,
                 grad_calculator = periodic_derivatives,
                 d=2, 
                 p=2, 
                 k=1, 
                 a=None, 
                 group=False, 
                 size_average=True, 
                 reduction=True
                 ):
        super(HsLoss_real, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a == None:
            a = [1,] * k
        self.a = a

        self.grad_function = grad_calculator

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms
    

    def __call__(self, x, y, a=None):
        
        dh = 2*np.pi/x.shape[-2]
        x_dx1, x_dy1, x_dx2, x_dy2 = self.grad_function(x, dx=dh, dy=dh)
        y_dx1, y_dy1, y_dx2, y_dy2 = self.grad_function(y, dx=dh, dy=dh)

        if self.balanced==False:
            weight_x = x**2
            weight_y = y**2
            if self.k >= 1:
                weight_x += a[0]**2 * (x_dx1[...,0]**2 + x_dy1[...,1]**2) 
                weight_y += a[0]**2 * (y_dx1[...,0]**2 + y_dy1[...,1]**2) 
            if self.k >= 2:
                weight_x += a[1]**2 * (x_dx2[...,0]**2 + x_dy2[...,1]**2) 
                weight_y += a[1]**2 * (y_dx2[...,0]**2 + y_dy2[...,1]**2) 
            weight_x = torch.sqrt(weight_x)
            weight_y = torch.sqrt(weight_y)
            loss = self.rel(weight_x, weight_y)
        
        else:
            loss = self.rel(x, y)
            if self.k >= 1:
                weight_x = a[0] * torch.sqrt(x_dx1[...,0]**2 + x_dy1[...,1]**2) 
                weight_y = a[0] * torch.sqrt(y_dx1[...,0]**2 + y_dy1[...,1]**2)
                loss += self.rel(weight_x, weight_y)
            if self.k >= 2:
                weight_x = a[1] * torch.sqrt(x_dx2[...,0]**2 + x_dy2[...,1]**2) 
                weight_y = a[1] * torch.sqrt(y_dx2[...,0]**2 + y_dy2[...,1]**2)
                loss += self.rel(weight_x, weight_y)
            loss = loss / (self.k+1)

        return loss
    
def periodic_derivatives(x, dx=1.0, dy=1.0):
    """
    Compute first and second derivatives with 2nd-order central differences
    and periodic BCs.
    
    x: tensor of shape (B, N, N) or (B, N, N, C)
    dx, dy: grid spacing
    
    Returns:
        dx1, dy1 : first derivatives
        dx2, dy2 : second derivatives
    """

    # shift along x-direction (dim=1)
    x_ip = torch.roll(x, shifts=-1, dims=1)
    x_im = torch.roll(x, shifts=+1, dims=1)

    # shift along y-direction (dim=2)
    y_ip = torch.roll(x, shifts=-1, dims=2)
    y_im = torch.roll(x, shifts=+1, dims=2)

    # First derivatives: (f[i+1] - f[i-1]) / (2*dx)
    dx1 = (x_ip - x_im) / (2.0 * dx)
    dy1 = (y_ip - y_im) / (2.0 * dy)

    # Second derivatives: (f[i+1] - 2f[i] + f[i-1]) / dx^2
    dx2 = (x_ip - 2.0 * x + x_im) / (dx * dx)
    dy2 = (y_ip - 2.0 * x + y_im) / (dy * dy)

    return dx1, dy1, dx2, dy2