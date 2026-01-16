import numpy as np
import torch

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
        
class BatchedAngularMeshRavel_ss:
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
        Transform arr between unraveled [m, n, ...] and raveled [p, ...] formats,
        with optional batch dimension.

        forward=True:
            (p, ...)        -> (m, n, ...)
            (B, p, ...)     -> (B, m, n, ...)

        forward=False:
            (m, n, ...)     -> (p, ...)
            (B, m, n, ...)  -> (B, p, ...)
        """

        arr = np.asarray(arr)

        # -------------------------
        # Detect batch dimension
        # -------------------------
        batched = arr.ndim >= 2 and (
            (forward and arr.shape[0] != self.points.shape[0]) or
            (not forward and arr.shape[0] != self.m)
        )

        if batched:
            B = arr.shape[0]
            data = arr
        else:
            B = 1
            data = arr[None]

        # ============================================================
        # Forward: [B, p, ...] -> [B, m, n, ...]
        # ============================================================
        if forward:
            assert data.shape[1] == self.points.shape[0], \
                f"Input length {data.shape[1]} mismatch, expected {self.points.shape[0]}"

            out_shape = (B, self.m, self.n) + data.shape[2:]
            unraveled = np.zeros(out_shape, dtype=data.dtype)

            for i, group_idx in enumerate(self.radius_groups):
                group_theta = self.theta[group_idx]
                sort_order = np.argsort(group_theta)

                for b in range(B):
                    group_arr = data[b, group_idx]
                    sorted_arr = group_arr[sort_order]
                    length = min(self.n, sorted_arr.shape[0])
                    unraveled[b, i, :length] = sorted_arr[:length]

            return unraveled[0] if not batched else unraveled

        # ============================================================
        # Inverse: [B, m, n, ...] -> [B, p, ...]
        # ============================================================
        else:
            p = self.points.shape[0]
            out_shape = (B, p) + data.shape[3:]
            raveled = np.zeros(out_shape, dtype=data.dtype)

            for i, group_idx in enumerate(self.radius_groups):
                group_theta = self.theta[group_idx]
                sort_order = np.argsort(group_theta)
                inv_sort_order = np.argsort(sort_order)
                length = len(group_idx)

                for b in range(B):
                    data_to_ravel = data[b, i, :min(self.n, length)]
                    data_original_order = data_to_ravel[inv_sort_order]
                    raveled[b, group_idx] = data_original_order

            return raveled[0] if not batched else raveled
        
class BatchedAngularMeshRavel:
    def __init__(self, points, m=59, n=356, device=None):
        """
        points : array-like [p, 2] (NumPy or torch)
        """
        if torch.is_tensor(points):
            pts = points.detach().cpu().numpy()
        else:
            pts = np.asarray(points)

        self.m = m
        self.n = n
        self.p = pts.shape[0]

        x, y = pts[:, 0], pts[:, 1]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)

        sort_idx = np.argsort(r)
        group_size = self.p // m

        radius_groups = []
        sort_orders = []
        inv_sort_orders = []

        for i in range(m):
            start = i * group_size
            end = (i + 1) * group_size if i < m - 1 else self.p
            group_idx = sort_idx[start:end]

            group_theta = theta[group_idx]
            so = np.argsort(group_theta)
            iso = np.argsort(so)

            radius_groups.append(group_idx)
            sort_orders.append(so)
            inv_sort_orders.append(iso)

        # store as torch tensors (no gradients needed)
        self.radius_groups = [
            torch.as_tensor(g, dtype=torch.long, device=device)
            for g in radius_groups
        ]
        self.sort_orders = [
            torch.as_tensor(s, dtype=torch.long, device=device)
            for s in sort_orders
        ]
        self.inv_sort_orders = [
            torch.as_tensor(s, dtype=torch.long, device=device)
            for s in inv_sort_orders
        ]

    def to(self, arr, forward=True):
        """
        forward:
            (p, ...)       -> (m, n, ...)
            (B, p, ...)    -> (B, m, n, ...)

        inverse:
            (m, n, ...)    -> (p, ...)
            (B, m, n, ...) -> (B, p, ...)
        """

        if not torch.is_tensor(arr):
            raise TypeError("Input must be a torch.Tensor")

        # -------------------------
        # Detect batching
        # -------------------------
        batched = (
            arr.ndim >= 2 and
            ((forward and arr.shape[0] != self.p) or
             (not forward and arr.shape[0] != self.m))
        )

        if batched:
            data = arr
            B = arr.shape[0]
        else:
            data = arr.unsqueeze(0)
            B = 1

        device = data.device
        dtype = data.dtype

        # ============================================================
        # Forward: [B, p, ...] -> [B, m, n, ...]
        # ============================================================
        if forward:
            assert data.shape[1] == self.p, \
                f"Expected p={self.p}, got {data.shape[1]}"

            out = torch.zeros(
                (B, self.m, self.n) + data.shape[2:],
                dtype=dtype,
                device=device,
            )

            for i, (gidx, so) in enumerate(zip(self.radius_groups, self.sort_orders)):
                gidx = gidx.to(device)
                so = so.to(device)

                length = min(self.n, gidx.numel())
                gathered = data[:, gidx]              # [B, k, ...]
                sorted_vals = gathered[:, so]         # angular sort
                out[:, i, :length] = sorted_vals[:, :length]

            return out[0] if not batched else out

        # ============================================================
        # Inverse: [B, m, n, ...] -> [B, p, ...]
        # ============================================================
        else:
            out = torch.zeros(
                (B, self.p) + data.shape[3:],
                dtype=dtype,
                device=device,
            )

            for i, (gidx, iso) in enumerate(
                zip(self.radius_groups, self.inv_sort_orders)
            ):
                gidx = gidx.to(device)
                iso = iso.to(device)

                length = gidx.numel()
                vals = data[:, i, :length]
                reordered = vals[:, iso]
                out[:, gidx] = reordered

            return out[0] if not batched else out
