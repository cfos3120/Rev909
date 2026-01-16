import torch
import numpy as np
from timeit import default_timer
from .mesh_utils import BatchedAngularMeshRavel, AngularMeshRavel

def w_to_u(w):
    batchsize = w.size(0)
    nx = w.size(1)
    ny = w.size(2)

    device = w.device
    w = w.reshape(batchsize, nx, ny, -1)

    w_h = torch.fft.fft2(w, dim=[1, 2])
    # Wavenumbers in y-direction
    k_max = nx // 2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(N, 1).repeat(1,
                                                                                                        N).reshape(
        1, N, N, 1)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(1, N).repeat(N,
                                                                                                        1).reshape(
        1, N, N, 1)
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[0, 0, 0, 0] = 1.0
    f_h = w_h / lap

    ux_h = 1j * k_y * f_h
    uy_h = -1j * k_x * f_h

    ux = torch.fft.irfft2(ux_h[:, :, :k_max + 1], dim=[1, 2])
    uy = torch.fft.irfft2(uy_h[:, :, :k_max + 1], dim=[1, 2])
    u = torch.cat([ux, uy], dim=-1)
    return u

def KF_flow_data(path, split, batch_size=1, sub=1, T_in=100, T_out=400, longrollout=False, convert_to_u=False):
    t1 = default_timer()
    data = np.load(path)
    B, T, S, _ = data.shape
    assert T_in+T_out <= T
    T = T_out - T_in
    print(f'Raw Dataset has shape:', data.shape, f'with shape {S:n}x{S:n}')
    ntrain = int(split*B)
    ntest = B - ntrain
    print(f'Dataset is split for training ({100*split:.0f}%) to {ntrain:n} - {ntest:n} with downsample of {sub:n}x')
    
    S = int(S/sub) 
    data = torch.tensor(data, dtype=torch.float)[..., ::sub, ::sub]

    if longrollout:
        test_u = data[-ntest+1,T_in-1:T_out]
        if convert_to_u:
            test_u = w_to_u(test_u)
        print('returning single validation case of size', test_u.shape)
        return test_u
    else:
        max_norm = np.linalg.norm(data[:,T_in:,...], axis=(2, 3)).max()
        print(f'Regularisation of Dataset has maximum norm (at subsample of {sub}x): {max_norm:.2f}')

        train_a = data[:ntrain,T_in-1:T_out-1].reshape(ntrain*T, S, S)
        train_u = data[:ntrain,T_in:T_out].reshape(ntrain*T, S, S)

        test_a = data[-ntest:,T_in-1:T_out-1].reshape(ntest*T, S, S)
        test_u = data[-ntest:,T_in:T_out].reshape(ntest*T, S, S)

        
        if convert_to_u:
            train_a = w_to_u(train_a)
            train_u = w_to_u(train_u)
            test_a = w_to_u(test_a)
            test_u = w_to_u(test_u)

        assert (S == train_u.shape[2])
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

        t2 = default_timer()
        print(f'preprocessing finished, time used: {(t2-t1):.2f}')
        return train_loader, test_loader, S, max_norm
    
def Cylinder_data(path, split, Ravler, batch_size=1, sub=1, T_in=0, longrollout=False):
    t1 = default_timer()
    data = np.load(path)[T_in:,...,:2]
    print(data.shape)
    T, N, C = data.shape
    assert T_in < T

    data = Ravler.to(torch.tensor(data, dtype=torch.float32), forward=True)
    T, W, H, C = data.shape
    assert W*H == N

    T = T - T_in
    print(f'Raw Dataset has shape:', data.shape, f'with shape {W:n}x{H:n}')
    ntrain = int(split*T)
    ntest = T - ntrain
    print(f'Dataset is split for training ({100*split:.0f}%) to {ntrain:n} - {ntest:n} with downsample of {sub:n}x')
    
    W = int(W/sub) 
    H = int(H/sub) 
    data = data[T_in:, ::sub, ::sub, :]

    if longrollout:
        test_u = data[ntrain+1:,...]
        print('returning single validation case of size', test_u.shape)
        return test_u
    else:
        max_norm = np.linalg.norm(data[:,T_in:,...], axis=(2, 3)).max()
        print(f'Regularisation of Dataset has maximum norm (at subsample of {sub}x): {max_norm:.2f}')

        train_a = data[:ntrain,...].reshape(-1, W, H, C)
        train_u = data[1:ntrain+1,...].reshape(-1, W, H, C)

        test_a = data[ntrain+1:-1,...].reshape(-1, W, H, C)
        test_u = data[ntrain+2:,...].reshape(-1, W, H, C)
        assert (W == train_u.shape[1]) and (H == train_u.shape[2])
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

        t2 = default_timer()
        print(f'preprocessing finished, time used: {(t2-t1):.2f}')
        return train_loader, test_loader, W, H, max_norm

        # output is trainloader (x,y) where x and y have shape [B,W,H,C] and y(t) = x(t+1) i.e. markovian

if __name__ == "__main__":

    path = r"C:\Users\Noahc\Documents\USYD\PHD\0 - Work Space\Markov Studies v2\RAS_Re100k.npy"
    path2 = r"C:\Users\Noahc\Documents\USYD\PHD\0 - Work Space\Markov Studies v2\RAS_Re100k_coords.npy"
    split = 0.9

    points = np.load(path2)
    ravler_class = BatchedAngularMeshRavel(points,m=59,n=356)

    train_loader, test_loader, W, H, max_norm = Cylinder_data(path, split, Ravler=ravler_class, batch_size=1, sub=1, T_in=0, longrollout=False)

    x,y = next(iter(train_loader))

    print(x.shape, y.shape)

    print(len(test_loader))