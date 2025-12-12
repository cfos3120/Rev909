import torch
import numpy as np
from timeit import default_timer

def KF_flow_data(path, split, batch_size=1, sub=1, T_in=100, T_out=400, longrollout=False):
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
        print('returning single validation case of size', test_u.shape)
        return test_u
    else:
        max_norm = np.linalg.norm(data[:,T_in:,...], axis=(2, 3)).max()
        print(f'Regularisation of Dataset has maximum norm (at subsample of {sub}x): {max_norm:.2f}')

        train_a = data[:ntrain,T_in-1:T_out-1].reshape(ntrain*T, S, S)
        train_u = data[:ntrain,T_in:T_out].reshape(ntrain*T, S, S)

        test_a = data[-ntest:,T_in-1:T_out-1].reshape(ntest*T, S, S)
        test_u = data[-ntest:,T_in:T_out].reshape(ntest*T, S, S)

        assert (S == train_u.shape[2])
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

        t2 = default_timer()
        print(f'preprocessing finished, time used: {(t2-t1):.2f}')
        return train_loader, test_loader, S, max_norm
    
    