import torch
import numpy as np
import argparse
from types import SimpleNamespace
import sys
import socket

if socket.gethostname() == 'DESKTOP-157DQSC':
    sys.path.insert(0, r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\Torch_VFM')
else:
    sys.path.insert(0, r'/home/n.foster/Torch_VFM')

from src.physics.operators import Divergence_Operator, Gradient_2nd_Operator

def parse_args():
    parser = argparse.ArgumentParser("Train the Markov Neural Operator")
    parser.add_argument('--config', type=str, help="json configuration file")
    parser.add_argument('--wandb', type=str, help="wandb mode", choices=['online','offline','disabled'], default='disabled')
    parser.add_argument('--sweep', type=str, help="sweep json file")
    args = parser.parse_args()
    return args

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

    grad_dict = {'dwdx':dx1,
                 'dwdy':dy1,
                 'dwdxx':dx2,
                 'dwdyy':dy2
                 }
    
    return grad_dict

def create_periodic_fvm_connectivity(S):
    owner_list = []
    neighbour_list = []
    normal_list = []
    
    # Horizontal faces: (i,j) -> (i, j+1), normal points right (+x direction)
    for i in range(S):
        for j in range(S):
            owner = i * S + j
            neigh = i * S + (j + 1) % S
            owner_list.append(owner)
            neighbour_list.append(neigh)
            #normal_list.append([1.0, 0.0])  # unit normal: right (+x)
            normal_list.append([0.0, 1.0])
    
    # Vertical faces: (i,j) -> (i+1, j), normal points up (+y direction)  
    for i in range(S):
        for j in range(S):
            owner = i * S + j
            neigh = ((i + 1) % S) * S + j
            owner_list.append(owner)
            neighbour_list.append(neigh)
            #normal_list.append([0.0, 1.0])  # unit normal: up (+y)
            normal_list.append([1.0, 0.0])
    
    owner = torch.tensor(owner_list, dtype=torch.int64)
    neighbour = torch.tensor(neighbour_list, dtype=torch.int64)
    normals = torch.tensor(normal_list, dtype=torch.float32)
    
    return owner, neighbour, normals

class periodic_isometric_grid(object):
    def __init__(self, L=2*np.pi, S=10, device='cpu', dtype=torch.float32):
        
        self.device = torch.device(device)
        face_length = 2*np.pi/S
        cell_area = face_length**2

        self.mesh = SimpleNamespace()
        self.mesh.dim = 2 # default is 3, not sure if operators optimized for 2D
        self.mesh.n_cells = S**2
        self.mesh.patch_face_keys = {} # periodicity enforced through cell neighbours
        self.mesh.face_owners, self.mesh.face_neighbors, normals = create_periodic_fvm_connectivity(S)
        self.mesh.face_areas = normals.to(dtype)*face_length
        self.mesh.num_internal_faces = len(self.mesh.face_owners)
        self.mesh.internal_faces = torch.arange(self.mesh.num_internal_faces)
        self.mesh.face_areas_mag = torch.tensor([face_length],dtype=dtype).repeat(self.mesh.num_internal_faces)
        self.mesh.cell_volumes = torch.tensor([cell_area],dtype=dtype).repeat(self.mesh.n_cells) 

        # create cell_centres
        self.mesh.cell_centers = self.fetch_2d_grid(L, S)
        self.mesh.cell_center_vectors = self.mesh.cell_centers[self.mesh.face_neighbors] - self.mesh.cell_centers[self.mesh.face_owners]
        self.mesh.cell_center_vectors -= L * np.round(self.mesh.cell_center_vectors / L) # correct for periodicity

        # none `mesh` namespace objects
        self.correction_method = None
        self.delta = normals.to(dtype)
        self.delta_mag = torch.norm(self.delta, dim=-1, keepdim=False).to(dtype) # should be all ones because they are unit vectors
        self.d = self.mesh.cell_center_vectors.to(dtype)
        self.d_mag = torch.norm(self.d, dim=-1, keepdim=False).to(dtype)
        self.internal_face_weights = torch.tensor([0.5]).repeat(self.mesh.num_internal_faces) # equidistant on isometric mesh
        
        # expand to 3D:
        if self.mesh.dim == 3:
            self.mesh.face_areas = torch.nn.functional.pad(self.mesh.face_areas, (0, 1), value=0.0)
            self.delta = torch.nn.functional.pad(self.delta, (0, 1), value=0.0)
            self.d = torch.nn.functional.pad(self.delta, (0, 1), value=0.0)

        # send all to device
        self.to(device)
        
    def fetch_2d_grid(self,L, S):
        line_grid = np.linspace(0,2*np.pi,S+1, endpoint=True)[1:]
        line_grid = line_grid-(line_grid[-1]-line_grid[-2])/2
        X, Y = np.meshgrid(line_grid,line_grid, indexing='ij')
        coords = np.concatenate([X[...,None], Y[...,None]], axis=-1)
        coords_r = torch.tensor(coords.reshape(-1,2))
        return coords_r

    def to(self,device):
        for attr_name, attr_value in vars(self.mesh).items():
            if isinstance(attr_value, torch.Tensor):
                setattr(self.mesh, attr_name, attr_value.to(device))

        for attr_name, attr_value in vars(self).items():
            if isinstance(attr_value, torch.Tensor):
                setattr(self, attr_name, attr_value.to(device))

class periodic_FVM(object):
    def __init__(self, S, device='cpu', L=2*np.pi):
        self.mesh = periodic_isometric_grid(L=L, S=S, device=device)
        self.S = S

        if self.mesh.mesh.dim == 2:
            self.scalar_1st_idx = {'dwdx':[0],'dwdy':[1]}
            self.scalar_2nd_idx = {'dwdxx':[0],'dwdyy':[3]}
            self.vector_1st_idx = {'dwdx':[0,1],'dwdy':[2,3]}
            self.vector_2nd_idx = {'dwdxx':[0,1],'dwdyy':[6,7]}
        else:
            raise NotImplementedError

    def __call__(self, x, **kwargs):
        
        if len(x.shape) == 3:
            x = x.unsqueeze(-1)
        B, H, W, C = x.shape
        x = x.reshape(B,1,H*W,C)

        _, grad_pred = Divergence_Operator.caclulate(self.mesh, field=x)
        grad_2nd_pred = Gradient_2nd_Operator.caclulate(self.mesh, field=x)
        
        # If vorticity function (assuming 2D)
        if C == 1:
            idx_1st = self.scalar_1st_idx
            idx_2nd = self.scalar_2nd_idx
        elif C == 2:
            idx_1st = self.vector_1st_idx
            idx_2nd = self.vector_2nd_idx
        else:
            raise ValueError(f'Input x of shape {x.shape} has {C} channels, if passing more than 2 channels, either 3D or velocity-pressure couple needs to be indexed')

        grad_dict = {'dwdx':grad_pred[...,idx_1st['dwdx']],
                        'dwdy':grad_pred[...,idx_1st['dwdy']],
                        'dwdxx':grad_2nd_pred[...,idx_2nd['dwdxx']],
                        'dwdyy':grad_2nd_pred[...,idx_2nd['dwdyy']]
                        }

        for key, value in grad_dict.items():
            grad_dict[key] = value.reshape(B,H,W,C)

        return grad_dict

class FVM_2D(object):
    def __init__(self, mesh, Ravler, volume_weighting=False, device='cpu'):
        self.mesh = mesh
        self.Ravler = Ravler
        self.volume_weighting = volume_weighting

        if self.mesh.mesh.dim == 2:
            self.scalar_1st_idx = {'dwdx':[0],'dwdy':[1]}
            self.scalar_2nd_idx = {'dwdxx':[0],'dwdyy':[3]}
            self.vector_1st_idx = {'dwdx':[0,1],'dwdy':[2,3]}
            self.vector_2nd_idx = {'dwdxx':[0,1],'dwdyy':[6,7]}
        else:
            raise NotImplementedError

    def __call__(self, x, **kwargs):
        
        if len(x.shape) == 3:
            x = x.unsqueeze(-1)
        B, H, W, C = x.shape

        x = self.Ravler.to(x, forward=False)
        x = x.reshape(B,1,H*W,C)

        _, grad_pred = Divergence_Operator.caclulate(self.mesh, field=x)
        grad_2nd_pred = Gradient_2nd_Operator.caclulate(self.mesh, field=x)
        
        if self.volume_weighting:
            grad_pred /= self.mesh.mesh.cell_volumes.reshape(1,1,-1,1)
            grad_2nd_pred /= self.mesh.mesh.cell_volumes.reshape(1,1,-1,1)

        # If vorticity function (assuming 2D)
        assert C == 2
        if C == 1:
            raise NotImplementedError
            idx_1st = self.scalar_1st_idx
            idx_2nd = self.scalar_2nd_idx
        elif C == 2:
            idx_1st = self.vector_1st_idx
            idx_2nd = self.vector_2nd_idx
        else:
            raise ValueError(f'Input x of shape {x.shape} has {C} channels, if passing more than 2 channels, either 3D or velocity-pressure couple needs to be indexed')

        grad_dict = {'dwdx':grad_pred[...,idx_1st['dwdx']],
                        'dwdy':grad_pred[...,idx_1st['dwdy']],
                        'dwdxx':grad_2nd_pred[...,idx_2nd['dwdxx']],
                        'dwdyy':grad_2nd_pred[...,idx_2nd['dwdyy']]
                        }

        for key, value in grad_dict.items():
            grad_dict[key] = self.Ravler.to(value.squeeze(1), forward=True).reshape(B,H,W,C)

        return grad_dict
    
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
            self.a = [1,1,1]
        else:
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
    

    def __call__(self, x, y, k=None):
        if k is None:
            k = self.k
        if k == 0:
            return self.rel(x, y)

        dh = 2*np.pi/x.shape[-2]
        # TODO: extend to cartesian velocity
        x_grad_dict = self.grad_function(x, dx=dh, dy=dh)
        y_grad_dict = self.grad_function(y, dx=dh, dy=dh)
        x_dx1, x_dy1, x_dx2, x_dy2 = [x_grad_dict[i] for i in ['dwdx','dwdy','dwdxx','dwdyy']]
        y_dx1, y_dy1, y_dx2, y_dy2 = [y_grad_dict[i] for i in ['dwdx','dwdy','dwdxx','dwdyy']]

        if self.balanced==False:
            weight_x = x**2
            weight_y = y**2
            if k >= 1:
                weight_x += self.a[0]**2 * (x_dx1**2 + x_dy1**2) 
                weight_y += self.a[0]**2 * (y_dx1**2 + y_dy1**2) 
            if k >= 2:
                weight_x += self.a[1]**2 * (x_dx2**2 + x_dy2**2) 
                weight_y += self.a[1]**2 * (y_dx2**2 + y_dy2**2)
                
            weight_x = torch.sqrt(weight_x)
            weight_y = torch.sqrt(weight_y)
            loss = self.rel(weight_x, weight_y)
        
        # had to add 1e-08 to stop sqrt(zero) which has NAN derivative
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight_x = self.a[0] * torch.sqrt((x_dx1**2 + x_dy1**2) + 1e-8)
                weight_y = self.a[0] * torch.sqrt((y_dx1**2 + y_dy1**2) + 1e-8)
                loss += self.rel(weight_x, weight_y)
            if k >= 2:
                weight_x = self.a[1] * torch.sqrt((x_dx2**2 + x_dy2**2) + 1e-8)
                weight_y = self.a[1] * torch.sqrt((y_dx2**2 + y_dy2**2) + 1e-8)
                loss += self.rel(weight_x, weight_y)
            loss = loss / (k+1)

        return loss



def spectrum2(u, s):
    T = u.shape[0]
    u = u.reshape(T, s, s)
    u = torch.fft.fft2(u)

    # 2d wavenumbers following Pytorch fft convention
    k_max = s // 2
    wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1), \
                            torch.arange(start=-k_max, end=0, step=1)), 0).repeat(s, 1)
    k_x = wavenumers.transpose(0, 1)
    k_y = wavenumers
    
    # Sum wavenumbers
    sum_k = torch.abs(k_x) + torch.abs(k_y)
    sum_k = sum_k.numpy()
    
    # Remove symmetric components from wavenumbers
    index = -1.0 * np.ones((s, s))
    index[0:k_max + 1, 0:k_max + 1] = sum_k[0:k_max + 1, 0:k_max + 1]

    spectrum = np.zeros((T, s))
    for j in range(1, s + 1):
        ind = np.where(index == j)
        spectrum[:, j - 1] = np.sqrt( (u[:, ind[0], ind[1]].sum(axis=1)).abs() ** 2)
        
    spectrum = spectrum.mean(axis=0)
    return spectrum
     

            