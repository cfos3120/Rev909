import torch
import numpy as np
import argparse
from types import SimpleNamespace
import sys

sys.path.insert(0, r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\Torch_VFM')
from src.physics.operators import Divergence_Operator

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

    def __call__(self, x, **kwargs):
        
        if len(x.shape) == 3:
            x = x.unsqueeze(-1)
        B, S, __, C = x.shape
        x = x.reshape(B,1,S*S,C)

        _, grad_pred = Divergence_Operator.caclulate(self.mesh, field=x)
        grad_2nd_pred = Gradient_2nd_Operator.caclulate(self.mesh, field=x)
        
        # If vorticity function (assuming 2D)
        if x.shape[-1] == 1 :
            grad_dict = {'dwdx':grad_pred[...,[0]],
                         'dwdy':grad_pred[...,[1]],
                         'dwdxx':grad_2nd_pred[...,[0]],
                         'dwdyy':grad_2nd_pred[...,[3]]
                         }

        # If velocity function (assuming 2D)
        else:
            grad_dict = {'dudx':grad_pred[...,[0]], 'dvdx':grad_pred[...,[1]],
                         'dudy':grad_pred[...,[2]], 'dvdy':grad_pred[...,[3]],
                         'dudxx':grad_2nd_pred[...,[0]], 'dvdxx':grad_2nd_pred[...,[1]],
                         'dudyy':grad_2nd_pred[...,[6]], 'dvdyy':grad_2nd_pred[...,[7]]
                         }

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
        
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight_x = self.a[0] * torch.sqrt(x_dx1**2 + x_dy1**2) 
                weight_y = self.a[0] * torch.sqrt(y_dx1**2 + y_dy1**2)
                loss += self.rel(weight_x, weight_y)
            if k >= 2:
                weight_x = self.a[1] * torch.sqrt(x_dx2**2 + x_dy2**2) 
                weight_y = self.a[1] * torch.sqrt(y_dx2**2 + y_dy2**2)
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
     
class AngularMeshRavel:
    def __init__(self, points, m=59, n=356):
        """
        Initialize the mesh unravel/ravel utility for an angular mesh.
        
        points : ndarray of shape [p, 2], Cartesian coordinates of mesh points
        m : int, number of radial layers expected
        n : int, number of angular points expected per layer
        
        This class sorts and groups points into radial layers and angular order,
        allowing forward unraveling to shape [m, n, 2] and inverse raveling to original [p, 2].
        """
        self.points = points
        self.m = m
        self.n = n
        
        # Compute polar coordinates for sorting and grouping
        x, y = points[:, 0], points[:, 1]
        self.r = np.sqrt(x**2 + y**2)
        self.theta = np.arctan2(y, x)
        
        # Sort points by radius and group indices to m radial layers
        sort_idx = np.argsort(self.r)
        p = points.shape[0]
        group_size = p // m
        self.radius_groups = []
        for i in range(m):
            start = i * group_size
            end = min((i+1) * group_size, p) if i < m-1 else p
            group_idx = sort_idx[start:end]
            self.radius_groups.append(group_idx)
    
    def _compute_unique_coords(self):
        """Compute representative radii and angles for each layer."""
        self.unique_radii = np.zeros(self.m)
        self.unique_angles = np.zeros(self.n)
        
        # Representative radius: mean radius per layer
        for i, group_idx in enumerate(self.radius_groups):
            self.unique_radii[i] = np.mean(self.r[group_idx])
        
        # Representative angles: evenly spaced (for plotting continuity)
        # Could use mean angles per angular position across radii if desired
        self.unique_angles = np.linspace(0, 2*np.pi, self.n, endpoint=False)
        return self.unique_angles, self.unique_radii

    def to(self, arr, forward=True):
        """
        Transform arr between unraveled [m, n, ...] and raveled [p, ...] formats.
        
        arr : input data array to transform (coordinates, scalar or vector values)
              Shape must align with either [p, ...] if forward=True or [m, n, ...] if forward=False
        forward : bool, True = forward unraveling [p, ...] -> [m, n, ...], 
                        False = inverse raveling [m, n, ...] -> [p, ...]
        
        Returns transformed array with reshaped/reordered data.
        """
        if forward:
            # Validate input shape
            assert arr.shape[0] == self.points.shape[0], f"Input length {arr.shape[0]} mismatch, expected {self.points.shape[0]}"
            # Initialize output container
            out_shape = (self.m, self.n) + arr.shape[1:]
            unraveled = np.zeros(out_shape, dtype=arr.dtype)
            
            for i, group_idx in enumerate(self.radius_groups):
                group_arr = arr[group_idx]
                # sort group by angle for continuity
                group_theta = self.theta[group_idx]
                sort_order = np.argsort(group_theta)
                sorted_arr = group_arr[sort_order]
                length = min(self.n, sorted_arr.shape[0])
                unraveled[i, :length] = sorted_arr[:length]
            
            return unraveled
        else:
            # inverse transform from [m, n, ...] to [p, ...]
            # initialize output container
            p = self.points.shape[0]
            raveled = np.zeros((p,) + arr.shape[2:], dtype=arr.dtype) if arr.ndim > 3 else np.zeros((p,), dtype=arr.dtype)
            idx_pos = 0
            for i, group_idx in enumerate(self.radius_groups):
                length = len(group_idx)
                data_to_ravel = arr[i, :min(self.n, length)]
                # reorder to original order before grouping (angular ordering was applied)
                group_theta = self.theta[group_idx]
                sort_order = np.argsort(group_theta)
                inv_sort_order = np.argsort(sort_order)
                data_original_order = data_to_ravel[inv_sort_order]
                raveled[group_idx] = data_original_order
                idx_pos += length
            
            return raveled
            