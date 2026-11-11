"""KNN layers on the foveated sensor manifold, and the KNN patch embedding.

Ported from fovi 27b1bde (``fovi/arch/knn.py`` and ``KNNPartitioningPatchEmbedding``
from ``fovi/arch/knnvit.py``): only what canvit's patchers reach. Method bodies are
verbatim (including the embedding's two ``KNNConvLayer.__init__`` calls, whose two
weight inits advance the RNG), so the KNN routing, the weights and their keys are
bit-identical to fovi's and old checkpoints load unchanged.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from canvit.core.foveation.coords import SamplingCoords, auto_match_num_coords, normalize


class KNNBaseLayer:
    """
    Abstract base class implementing basic KNN functionalities for foveated vision.
    
    This class provides the foundation for KNN-based operations in foveated neural networks,
    including distance computation in both visual and cortical spaces.
    """
    
    def _compute_knns(self, batch_size=None):
        """
        Compute distances between input and output coordinates in visual or cortical space using batched approach to limit memory demands for high resolution coordinate systems.
        
        This method supports multiple distance computation strategies:
        - Euclidean distances in visual space (interesting baseline)
        - Euclidean distances in cortical space (most typical -- fast and bio-plausible)
        - Geodesic distances on cortical surface (slower, more ideal, but very well approximated by euclidean for typical small local RFs, so not typically necessary. we tend to use it for ViTs where there are fewer coordinates with larger kernels and only one KNNConv, but not for CNNs where there are many layers of KNNConvs and more coordinates with smaller kernels.)
        
        Args:
            batch_size (int, optional): Number of output coordinates to process at once. 
                                      If None, uses a default based on available memory.
        
        Returns:
            tuple: (knn_indices, knn_distances) where:
                - knn_indices: Tensor of shape (k, num_output_coords) with indices of k nearest neighbors
                - knn_distances: Tensor of shape (k, num_output_coords) with distances to k nearest neighbors
        """
        
        # Set default batch size if not provided
        if batch_size is None:
            # Default batch size: try to keep memory usage reasonable
            # Aim for less than ~500M elements in the distance matrix per batch
            num_input_coords = self.in_coords.shape[0] + self.in_coords.cartesian_pad_coords.shape[0]
            num_output_coords = self.out_coords.shape[0]
            num_pairs = num_input_coords*num_output_coords
            batch_size = np.minimum(num_output_coords / np.maximum(0.0001, (num_pairs / 500000000)), num_output_coords).astype(int)
        
        num_output_coords = self.out_coords.shape[0]
        batch_size = min(batch_size, num_output_coords)
        
        if self.sample_cortex == 'geodesic':
            return self._compute_geodesic_distances()
        else:
            return self._compute_euclidean_distances_batched(batch_size)
    
    def _compute_geodesic_distances(self):
        """
        Compute geodesic distances using the existing approach (already memory efficient).
        """
        import pygeodesic.geodesic as geodesic
        from scipy.spatial import Delaunay
        m = 4 # multiple of k nearest euclidean neighbors to consider for geodesic distance computation. speeds things up substantially. works well because euclidean is a good local approximation. 
        num_neighbors = torch.minimum(torch.tensor(self.k*m), torch.tensor(self.in_coords.shape[0]))
        self.sample_coords_cart = torch.concatenate([self.in_coords.cartesian, self.in_coords.cartesian_pad_coords], dim=0)
        self.sample_coords = torch.concatenate([self.in_coords.cortical, self.in_coords.cortical_pad_coords], dim=0)
        out_coords = self.out_coords.cortical.clone()

        # Initialize output tensors
        knn_indices = torch.zeros((self._k, self.out_coords.shape[0]), dtype=torch.long, device=self.device)
        knn_distances = torch.zeros((self._k, self.out_coords.shape[0]), dtype=torch.float32, device=self.device)

        # iterate over output units and compute geodesic distances to the m*k nearest euclidean neighbors
        for i in range(self.out_coords.shape[0]):
            # Compute euclidean distances for the single output coordinate
            single_out_coord = out_coords[i:i+1]
            D_euc = torch.cdist(self.sample_coords, single_out_coord).squeeze(1)
            
            # Find m*k nearest euclidean neighbors
            target_points = D_euc.topk(num_neighbors, dim=0, largest=False).indices
            
            # Compute geodesic distances
            tri = Delaunay(self.sample_coords_cart[target_points].cpu())
            F = tri.simplices  # shape: [M, 3]
            V = self.sample_coords[target_points].cpu() # [N, 3]  3D positions
            algorithm = geodesic.PyGeodesicAlgorithmExact(V, F)
            dists = algorithm.geodesicDistances([0])
            
            # Store geodesic distances for all candidates
            geodesic_dists = torch.full((self.sample_coords.shape[0],), float('inf'), dtype=torch.float32, device=self.device)
            geodesic_dists[target_points] = torch.tensor(dists[0], dtype=torch.float32, device=self.device)
            
            # Find k nearest neighbors from geodesic distances
            _, topk_indices = torch.topk(geodesic_dists, self._k, dim=0, largest=False)
            knn_indices[:, i] = topk_indices
            knn_distances[:, i] = geodesic_dists[topk_indices]

        return knn_indices, knn_distances
    
    def _compute_euclidean_distances_batched(self, batch_size):
        """
        Compute euclidean distances using batched approach.
        """
        # Setup coordinates
        wrap_row = False
        if self.sample_cortex:
            self.sample_coords_cart = torch.concatenate([self.in_coords.cartesian, self.in_coords.cartesian_pad_coords], dim=0)
            out_coords = self.out_coords.cortical.clone()
            if self.in_coords.style in ['image', 'logpolar']:
                assert self.out_coords.style == self.in_coords.style 
                self.sample_coords = self.in_coords.cortical.clone() # not bothering with pad coords here
                self.sample_coords = (self.sample_coords - self.sample_coords.min(dim=0)[0])/(self.sample_coords.max(dim=0)[0] - self.sample_coords.min(dim=0)[0])
                out_coords = (out_coords - out_coords.min(dim=0)[0])/(out_coords.max(dim=0)[0] - out_coords.min(dim=0)[0])
                wrap_row = True
            else:
                self.sample_coords = torch.concatenate([self.in_coords.cortical, self.in_coords.cortical_pad_coords], dim=0)
        else:
            self.sample_coords = torch.concatenate([self.in_coords.cartesian, self.in_coords.cartesian_pad_coords], dim=0)
            out_coords = self.out_coords.cartesian

        # Initialize output tensors
        num_output_coords = out_coords.shape[0]
        knn_indices = torch.zeros((self._k, num_output_coords), dtype=torch.long, device=self.device)
        knn_distances = torch.zeros((self._k, num_output_coords), dtype=torch.float32, device=self.device)
        
        # Process output coordinates in batches
        for start_idx in range(0, num_output_coords, batch_size):
            end_idx = min(start_idx + batch_size, num_output_coords)
            batch_out_coords = out_coords[start_idx:end_idx]
            
            # Compute distances for this batch
            if wrap_row:
                # for log polar image cortical distances, we wrap along the row dimension (angles)
                row_dists = torch.cdist(self.sample_coords[:,0].unsqueeze(1), batch_out_coords[:,0].unsqueeze(1), p=1)
                col_dists = torch.cdist(self.sample_coords[:,1].unsqueeze(1), batch_out_coords[:,1].unsqueeze(1), p=1)
                max_row = self.sample_coords[:,1].max()
                mid_row = self.sample_coords[:,1].min() + (max_row - self.sample_coords[:,1].min())/2
                wrap_inds = col_dists > mid_row
                col_dists[wrap_inds] = max_row - col_dists[wrap_inds]
                batch_distances = (row_dists**2 + col_dists**2)**(1/2)
            else:
                batch_distances = torch.cdist(self.sample_coords, batch_out_coords)
            
            # Find k nearest neighbors for this batch
            batch_knn_distances, batch_knn_indices = torch.topk(batch_distances, self._k, dim=0, largest=False)
            
            # Store results
            knn_indices[:, start_idx:end_idx] = batch_knn_indices
            knn_distances[:, start_idx:end_idx] = batch_knn_distances
        
        return knn_indices, knn_distances

    def _compute_all_distances(self):
        """
        Compute distances between all input and output coordinates in visual or cortical space. Typically this isn't used, but we need it for the PartitioningPatchEmbedding.
        
        This method supports multiple distance computation strategies:
        - Euclidean distances in visual space (interesting baseline)
        - Euclidean distances in cortical space (most typical -- fast and bio-plausible)
        - Geodesic distances on cortical surface (slower, more ideal, but very well approximated by euclidean for typical small local RFs, so not typically necessary. we tend to use it for ViTs where there are fewer coordinates with larger kernels and only one KNNConv, but not for CNNs where there are many layers of KNNConvs and more coordinates with smaller kernels.)
        
        Returns:
            torch.Tensor: Distance matrix between input and output coordinates.
        """

        if self.sample_cortex == 'geodesic':
            import pygeodesic.geodesic as geodesic
            from scipy.spatial import Delaunay
            m = 4 # multiple of k nearest euclidean neighbors to consider for geodesic distance computation. speeds things up substantially. works well because euclidean is a good local approximation. 
            num_neighbors = torch.minimum(torch.tensor(self.k*m), torch.tensor(self.in_coords.shape[0]))
            self.sample_coords_cart = torch.concatenate([self.in_coords.cartesian, self.in_coords.cartesian_pad_coords], dim=0)
            self.sample_coords = torch.concatenate([self.in_coords.cortical, self.in_coords.cortical_pad_coords], dim=0)
            out_coords = self.out_coords.cortical.clone()

            # compute euclidean distances between all input and output units
            D_euc = torch.cdist(self.sample_coords, out_coords)

            D_init = torch.inf * torch.ones_like(D_euc)

            # iterate over output units and compute geodesic distances to the m*k nearest euclidean neighbors
            all_dists = []
            all_indices = []
            for i in range(self.out_coords.shape[0]):
                source_idx = i
                target_points = D_euc[:,i].topk(num_neighbors, dim=0, largest=False).indices
                tri = Delaunay(self.sample_coords_cart[target_points].cpu())
                F = tri.simplices  # shape: [M, 3]
                V = self.sample_coords[target_points].cpu() # [N, 3]  3D positions
                algorithm = geodesic.PyGeodesicAlgorithmExact(V, F)
                dists = algorithm.geodesicDistances([0])
                all_dists.append(dists[0])
                all_indices.append(target_points)
                D_init[target_points,i] = torch.tensor(dists[0], dtype=D_euc.dtype, device=D_euc.device)

            distances = D_init
        else:
            wrap_row = False
            if self.sample_cortex:
                self.sample_coords_cart = torch.concatenate([self.in_coords.cartesian, self.in_coords.cartesian_pad_coords], dim=0)
                out_coords = self.out_coords.cortical.clone()
                if self.in_coords.style == 'image':
                    self.sample_coords = self.in_coords.cortical.clone() # not bothering with pad coords here
                    self.sample_coords = (self.sample_coords - self.sample_coords.min(dim=0)[0])/(self.sample_coords.max(dim=0)[0] - self.sample_coords.min(dim=0)[0])
                    out_coords = (out_coords - out_coords.min(dim=0)[0])/(out_coords.max(dim=0)[0] - out_coords.min(dim=0)[0])
                    wrap_row = True
                    assert self.out_coords.style == 'image'
                else:
                    self.sample_coords = torch.concatenate([self.in_coords.cortical, self.in_coords.cortical_pad_coords], dim=0)
            else:
                self.sample_coords = torch.concatenate([self.in_coords.cartesian, self.in_coords.cartesian_pad_coords], dim=0)
                out_coords = self.out_coords.cartesian

            if wrap_row:
                # for log polar image cortical distances, we wrap along the row dimension (angles)
                row_dists = torch.cdist(self.sample_coords[:,0].unsqueeze(1), out_coords[:,0].unsqueeze(1), p=1)
                col_dists = torch.cdist(self.sample_coords[:,1].unsqueeze(1), out_coords[:,1].unsqueeze(1), p=1)
                max_row = self.sample_coords[:,1].max()
                mid_row = self.sample_coords[:,1].min() + (max_row - self.sample_coords[:,1].min())/2
                wrap_inds = col_dists > mid_row
                col_dists[wrap_inds] = max_row - col_dists[wrap_inds]
                distances = (row_dists**2 + col_dists**2)**(1/2)
            else:
                distances = torch.cdist(self.sample_coords, out_coords)
        return distances


class KNNPoolingLayer(nn.Module, KNNBaseLayer):
    """
    K-Nearest Neighbors pooling layer for foveated vision.
    
    This layer performs pooling operations over k-nearest neighbors in either
    visual or cortical space, supporting various pooling modes.
    
    Args:
        k (int): Number of nearest neighbors to consider.
        in_coords (SamplingCoords): Input sampling coordinates object.
        out_coords (SamplingCoords): Output sampling coordinates object.
        mode (str): Pooling mode ('max', 'avg', 'sum', 'gaussian').
        device (str): PyTorch device to run on.
        sample_cortex (bool or str): Whether/how to sample cortical space:
            - False: Sample visual field
            - True: Sample cortical space using Euclidean distances (fast, approximate)
            - 'geodesic': Sample cortical space using geodesic distances (slower, more accurate)
        gauss_sigma (float): sigma for mode='gaussian' pooling; not used for other modes
        batch_size (int, optional): Number of output coordinates to process at once for memory efficiency.
                                   If None, uses a default based on available memory.
    """
    
    def __init__(self, k, in_coords, out_coords, mode='max', device='cuda', sample_cortex=True, gauss_sigma=10, batch_size=None):
        super().__init__()
        self.k = k
        self.in_coords = in_coords
        self.out_coords = out_coords
        self.device = device
        assert mode in ['max', 'avg', 'sum', 'gaussian']
        self.mode = mode
        self.sample_cortex = sample_cortex
        self.batch_size = batch_size
        assert isinstance(in_coords, SamplingCoords) and isinstance(out_coords, SamplingCoords)

        self.k = torch.minimum(torch.tensor(self.k), torch.tensor(self.in_coords.shape[0])) # ensure k is not greater than the number of input coordinates
        self._k = int(self.k.item())

        self.knn_indices, self.knn_distances = self._compute_knns(batch_size)

        # compute padding mask for use at inference
        self.knn_pad_token_val = self.in_coords.shape[0]
        self.knn_indices_pad_mask = self.knn_indices >= self.in_coords.shape[0]
        self.knn_indices_pad_token = self.knn_indices.clone()
        self.knn_indices_pad_token[self.knn_indices_pad_mask] = self.knn_pad_token_val # pad index

        if self.mode == 'gaussian':
            self.gauss_sigma = gauss_sigma
            self.gaussian_weights = self._compute_gaussian_weights()

    def _compute_gaussian_weights(self):
        """
        Compute Gaussian weights for local pooling based on distances between 
        output nodes and their input neighbors.
        
        Returns:
            torch.Tensor: Gaussian weights of shape (k, num_output_nodes)
        """

        # Normalize over each neighborhood
        k_distances = normalize(self.knn_distances, dim=0)
        
        # Compute Gaussian weights: exp(-distance^2 / (2 * sigma^2))
        gaussian_weights = torch.exp(-k_distances**2 / (2 * self.gauss_sigma**2))
        
        # Normalize weights so they sum to 1 for each output node
        gaussian_weights = gaussian_weights / (gaussian_weights.sum(dim=0, keepdim=True) + 1e-8)
        
        return gaussian_weights

    def forward(self, X_l):
        """
        Apply KNN pooling to input features.
        
        Args:
            X_l (torch.Tensor): Input features of shape [batch_size, channels, num_nodes].
            
        Returns:
            torch.Tensor: Pooled features of shape [batch_size, channels, num_output_nodes].
        """
        # pad X_l with a single nan-value that will be indexed by padding units
        X_l = torch.concatenate([X_l, torch.nan*torch.zeros(X_l.shape[0], X_l.shape[1], 1, device=X_l.device, dtype=X_l.dtype)], dim=2)

        # Gather the k neighbors' features for each node in layer l+1
        batch_size, d_l, N_l = X_l.shape
        N_l_plus_1 = self.knn_indices_pad_token.shape[1]
        knn_features = torch.gather(X_l, 2, self.knn_indices_pad_token.reshape(1, 1, -1).expand(batch_size, d_l, -1))
        knn_features = knn_features.reshape(batch_size, d_l, self._k, N_l_plus_1)

        # do pooling over the k neighbors, ignoring nans from the padding units
        if self.mode == 'avg':
            X_out = torch.nanmean(knn_features, dim=2)
        elif self.mode == 'max':
            knn_features[torch.isnan(knn_features)] = -float('inf')
            X_out, _ = torch.max(knn_features, dim=2)
        elif self.mode == 'sum':
            X_out = torch.nansum(knn_features, dim=2)
        elif self.mode == 'gaussian':
            # Apply Gaussian weighting: weighted average with Gaussian weights
            # knn_features shape: (batch_size, d_l, k, N_l_plus_1)
            # gaussian_weights shape: (k, N_l_plus_1)

            # Handle NaN values by setting them to 0 and adjusting weights
            nan_mask = torch.isnan(knn_features)
            knn_features_clean = torch.where(nan_mask, torch.zeros_like(knn_features), knn_features)
            
            # Adjust weights for NaN positions (set to 0)
            adjusted_weights = self.gaussian_weights.unsqueeze(0).unsqueeze(0)  # (1, 1, k, N_l_plus_1)
            adjusted_weights = torch.where(nan_mask, torch.zeros_like(adjusted_weights), adjusted_weights)
            
            # Renormalize weights after removing NaN contributions
            weight_sums = adjusted_weights.sum(dim=2, keepdim=True)  # (batch_size, d_l, 1, N_l_plus_1)
            adjusted_weights = adjusted_weights / (weight_sums + 1e-8)
            
            # Compute weighted average
            X_out = torch.sum(knn_features_clean * adjusted_weights, dim=2)
        else:
            raise NotImplementedError(f"Pooling style {self.mode} not implemented")
                
        return X_out

    def __repr__(self):
        return f'KNNPoolingLayer(\n\tmode={self.mode}\n\tk={self.k}\n\tin_coords={self.in_coords}\n\tout_coords={self.out_coords}\n\tsample_cortex={self.sample_cortex}\n)'


class KNNConvLayer(nn.Module, KNNBaseLayer):
    """
    K-Nearest Neighbors convolution layer for foveated vision.
    
    This layer performs convolution operations over k-nearest neighbors in either
    visual or cortical space, with learnable filters aligned to spatial kernels.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        k (int): Number of nearest neighbors to consider.
        in_coords (SamplingCoords): Input sampling coordinates object.
        out_coords (SamplingCoords): Output sampling coordinates object.
        device (str): PyTorch device to run on.
        arch_flag (str): Architecture flag for reference coordinate computation.
        sample_cortex (bool): Whether to sample cortical space.
        bias (bool): Whether to use bias in convolution.
        ref_frame_side_length (int, optional): Manual specification of reference frame side length.
        batch_size (int, optional): Number of output coordinates to process at once for memory efficiency.
                                   If None, uses a default based on available memory.
        per_ring_kernel (bool): If True, learn a separate convolution kernel for each
                                eccentricity ring of output coordinates instead of a single
                                shared kernel (see _init_ring_groups). Default False
                                (original shared-kernel behavior).
        rf_interp (str): How a neighbor reads the kernel grid (see compute_local_rf).
                         'nearest' (default, original behavior): its nearest reference cell.
                         'bilinear': its 4 surrounding cells, with bilinear weights (a
                         degree-1 B-spline kernel). Same grid and parameters either way.
        _precomputed_knns (tuple, optional): Internal override containing the
                                            precomputed neighbor indices and distances.
    """

    def __init__(self, in_channels, out_channels, k, in_coords, out_coords,
                 device='cuda', 
                 arch_flag='',
                 sample_cortex=True,
                 bias=False,
                 ref_frame_side_length=None,
                 batch_size=None,
                 padding="zero",
                 per_ring_kernel=False,
                 rf_interp="nearest",
                 _precomputed_knns=None,
                 ):
        super().__init__()
        assert rf_interp in ("nearest", "bilinear"), f"rf_interp must be 'nearest' or 'bilinear', got {rf_interp!r}"
        self.rf_interp = rf_interp
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.in_coords = in_coords
        self.out_coords = out_coords
        self.device = device
        self.arch_flag = arch_flag
        self.sample_cortex = sample_cortex
        self.ref_frame_side_length = ref_frame_side_length # if we want to specify manually
        self.batch_size = batch_size
        
        assert isinstance(in_coords, SamplingCoords) and isinstance(out_coords, SamplingCoords)

        self.k = torch.minimum(torch.tensor(self.k), torch.tensor(self.in_coords.shape[0])) # ensure k is not greater than the number of input coordinates
        self._k = int(self.k.item())

        if _precomputed_knns is None:
            self.knn_indices, self.knn_distances = self._compute_knns(batch_size)
        else:
            knn_indices, knn_distances = _precomputed_knns
            expected_shape = (self._k, self.out_coords.shape[0])
            if tuple(knn_indices.shape) != expected_shape:
                raise ValueError(
                    "Precomputed KNN indices have shape "
                    f"{tuple(knn_indices.shape)}; expected {expected_shape}"
                )
            if knn_indices.dtype != torch.long:
                raise TypeError("Precomputed KNN indices must have dtype torch.long")
            if knn_distances.shape != knn_indices.shape:
                raise ValueError(
                    "Precomputed KNN distances must match the indices shape"
                )
            if knn_distances.device != knn_indices.device:
                raise ValueError(
                    "Precomputed KNN indices and distances must be on the same device"
                )
            self.knn_indices = knn_indices
            self.knn_distances = knn_distances

        # compute padding mask for use at inference
        self.knn_indices_pad_mask = self.knn_indices >= self.in_coords.shape[0]
        self.knn_indices_pad_token = self.knn_indices.clone()
        self.knn_indices_pad_token[self.knn_indices_pad_mask] = self.in_coords.shape[0] # pad index

        # this will be updated to the correct size for a batch to be used for batches of the same size
        self.knn_indices_batch_cache = self.knn_indices_pad_token

        # compute reference coordinates
        self.compute_reference_coords(self.arch_flag)

        # Optional per-ring kernels: a separate kernel for each eccentricity ring
        # of output coordinates. Off by default (single shared kernel). When on,
        # the weight gains a leading ring dimension; ring assignment is fixed
        # geometry, computed once here.
        self.per_ring_kernel = bool(per_ring_kernel)
        if self.per_ring_kernel:
            self._init_ring_groups()

        # create and initialize weights
        n_ref = self.ref_coords.shape[0]
        if self.per_ring_kernel:
            self.weight = nn.Parameter(torch.empty(self.n_rings, self.out_channels, self.in_channels * n_ref))
        else:
            self.weight = nn.Parameter(torch.empty(self.out_channels, self.in_channels * n_ref))
        if bias:
            if self.per_ring_kernel:
                self.bias = nn.Parameter(torch.empty(self.n_rings, self.out_channels))
            else:
                self.bias = nn.Parameter(torch.empty(self.out_channels))
        else:
            self.register_parameter('bias', None)

        self._init_conv_like()

        # initialize mapping between knns and reference coords
        self.local_rf = self.compute_local_rf()

        # Optional learned padding. 'zero' (default) keeps the original behavior
        # (out-of-field neighbor slots contribute 0). 'learned' replaces that
        # zero with a learned per-(input-channel, reference-cell) value on the
        # out-of-field kernel cells (see _init_learned_padding).
        assert padding in ("zero", "learned", "learned_shared"), (
            f"padding must be 'zero', 'learned', or 'learned_shared', got {padding!r}"
        )
        self.padding = padding
        self._learned_padding = padding in ("learned", "learned_shared")
        if self._learned_padding:
            self._init_learned_padding()

    def _init_conv_like(self):
        """
        Initialize convolution-like parameters for the KNN layer.
        """
        if getattr(self, "per_ring_kernel", False):
            # Initialize a single kernel slab and share it across all rings, so
            # that enabling per-ring kernels is bit-identical to the shared
            # kernel at initialization (rings diverge only through training).
            slab = torch.empty(self.out_channels, self.weight.shape[-1],
                               device=self.weight.device, dtype=self.weight.dtype)
            nn.init.kaiming_normal_(slab)
            with torch.no_grad():
                self.weight.copy_(slab.unsqueeze(0).expand_as(self.weight))
        else:
            # Initialize weight like nn.Linear does with kaiming
            nn.init.kaiming_normal_(self.weight)

        # Initialize bias if present
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _init_ring_groups(self):
        """Group output coordinates by eccentricity ring for per-ring kernels.

        In an isotropic foveated sensor the output coordinates lie on discrete
        concentric rings: the sampler stores an identical polar radius for every
        coordinate on a ring, and radii across rings are strictly distinct (see
        ``fovi.sensing.coords.get_isotropic_sampling_coords``). Grouping on the
        polar radius therefore recovers the rings exactly -- one group per
        eccentricity, never split. Radii are rounded first only to guard against
        float noise from non-isotropic / derived coordinate paths.

        Sets ``self.n_rings`` and registers ``self.ring_ids`` ([n_out] long,
        ring index per output coordinate).
        """
        radii = self.out_coords.polar[:, 0]
        radii_round = torch.round(radii * 1e5) / 1e5
        unique = torch.unique(radii_round)
        n_rings = int(unique.numel())
        n_out = int(radii_round.numel())
        assert n_rings < n_out or n_out <= 1, (
            f"per_ring_kernel: got {n_rings} rings for {n_out} output coordinates -- "
            "coordinates are not grouping into shared-radius rings. per_ring_kernel "
            "expects an isotropic foveated sampling (polar radii shared within rings)."
        )
        ring_ids = torch.searchsorted(unique, radii_round).to(torch.long)
        self.n_rings = n_rings
        # Non-persistent: regenerated from fixed geometry on construction.
        self.register_buffer("ring_ids", ring_ids, persistent=False)

    def _apply_per_ring_linear(self, knn_features):
        """Project with a separate kernel per eccentricity ring.

        ``knn_features``: [batch, n_out, in_channels * n_ref]. ``self.weight`` is
        [n_rings, out_channels, in_channels * n_ref]; each output coordinate uses
        its ring's slab (``ring_ids`` maps coordinate -> ring). Implemented as a
        gather + einsum so all shapes are static (torch.compile friendly) -- the
        FLOPs match the shared-kernel ``F.linear``.
        """
        ring_ids = self.ring_ids.to(knn_features.device)
        w = self.weight[ring_ids]  # [n_out, out_channels, in_channels*n_ref]
        out = torch.einsum("bnf,nof->bno", knn_features, w)  # [batch, n_out, out_channels]
        if self.bias is not None:
            out = out + self.bias[ring_ids]  # broadcast [n_out, out_channels] over batch
        return out.transpose(1, 2)  # [batch, out_channels, n_out]

    @torch.no_grad()
    def prune_output_coords(self, keep_mask):
        """Drop a subset of the output coordinates (patches) in place.

        ``keep_mask``: boolean tensor of length ``n_out`` (current number of
        output coordinates); ``True`` keeps that patch. Every tensor indexed by
        the output-coordinate axis is subset consistently so the layer behaves
        exactly as if those patches had never been part of the partition:

          * ``out_coords`` (the per-patch sampling coordinates),
          * ``knn_indices`` / ``knn_indices_pad_mask`` / ``knn_indices_pad_token``
            / ``knn_indices_batch_cache`` (``[k, n_out]``),
          * ``local_rf`` (``[n_out, k, n_ref]``) and the cached ``knn_coords``,
          * ``distances`` (``[n_in(+pad), n_out]``) when present,
          * ``out_of_field_mask`` (learned padding, ``[n_out, n_ref]``),
          * per-ring kernels: ``ring_ids`` are subset and re-indexed to a
            contiguous range, and any ring with no surviving patch has its
            weight/bias slab dropped (each kept patch keeps its own kernel).

        The *input* coordinates (``in_coords``) and the neighbor count ``k`` are
        left untouched: dropping output patches never changes which input
        sensors exist, only which patches pool them. The kernel ``weight`` is
        unchanged in the shared-kernel case (it is not output-indexed).

        Returns ``self``. A no-op (and returns immediately) if ``keep_mask`` is
        all-True, so callers can guard cheaply for the "no pruning" path.
        """
        keep = torch.as_tensor(keep_mask, device=self.knn_indices.device)
        if keep.dtype != torch.bool:
            keep = keep.to(torch.bool)
        n_out = self.knn_indices.shape[1]
        assert keep.shape == (n_out,), (
            f"keep_mask must have length n_out={n_out}, got {tuple(keep.shape)}"
        )
        if bool(keep.all()):
            return self

        # Per-patch sampling coordinates.
        self.out_coords.subset(keep)

        # KNN bookkeeping: [k, n_out] tensors -> subset the output (column) axis.
        for name in (
            "knn_indices", "knn_indices_pad_mask",
            "knn_indices_pad_token", "knn_indices_batch_cache",
        ):
            t = getattr(self, name, None)
            if isinstance(t, torch.Tensor) and t.shape[-1] == n_out:
                setattr(self, name, t[:, keep].contiguous())

        # Local receptive field one-hot routing: [n_out, k, n_ref].
        if isinstance(getattr(self, "local_rf", None), torch.Tensor):
            self.local_rf = self.local_rf[keep].contiguous()

        # Cached normalized KNN coords [k, n_out, 2] (viz / re-init only).
        if isinstance(getattr(self, "knn_coords", None), torch.Tensor) and self.knn_coords.shape[1] == n_out:
            self.knn_coords = self.knn_coords[:, keep].contiguous()

        # Distance matrix [n_in(+pad), n_out] (not used in forward; kept consistent).
        if isinstance(getattr(self, "distances", None), torch.Tensor) and self.distances.shape[-1] == n_out:
            self.distances = self.distances[:, keep].contiguous()

        # Learned-padding out-of-field mask [n_out, n_ref].
        if isinstance(getattr(self, "out_of_field_mask", None), torch.Tensor):
            self.out_of_field_mask = self.out_of_field_mask[keep].contiguous()

        # Per-ring kernels: subset ring ids, drop now-empty rings, re-index.
        if getattr(self, "per_ring_kernel", False):
            ring_ids = self.ring_ids[keep]
            remaining = torch.unique(ring_ids)                 # sorted, kept ring slabs
            new_ring_ids = torch.searchsorted(remaining, ring_ids)
            self.register_buffer("ring_ids", new_ring_ids.contiguous(), persistent=False)
            self.weight = nn.Parameter(self.weight.data[remaining].clone())
            if self.bias is not None:
                self.bias = nn.Parameter(self.bias.data[remaining].clone())
            self.n_rings = int(remaining.numel())

        return self

    def _init_learned_padding(self):
        """Set up kernel-position learned padding.

        Replaces the zero contribution of out-of-field neighbor slots with a
        learned value. A reference cell is treated as out-of-field for an output
        unit when a padding (out-of-FOV) neighbor routed into it but no real
        neighbor did -- i.e. that kernel tap points outside the sampled region.
        ``padding='learned'`` learns an independent value per (input-channel,
        reference-cell); ``padding='learned_shared'`` learns a single per-channel
        value broadcast to all out-of-field cells (classic CNN learned padding).
        Values are zero-initialized, so the layer is identical to
        ``padding='zero'`` at initialization.
        """
        n_ref = self.ref_coords.shape[0]
        local_rf = self.local_rf  # [n_out, k, n_ref], one-hot routing of slots to cells
        real_mask = (~self.knn_indices_pad_mask).to(local_rf.dtype)  # [k, n_out]
        pad_mask = self.knn_indices_pad_mask.to(local_rf.dtype)      # [k, n_out]
        # Per (output, cell): how many real / padding slots routed into that cell.
        real_count = torch.einsum("nkv,kn->nv", local_rf, real_mask)  # [n_out, n_ref]
        pad_count = torch.einsum("nkv,kn->nv", local_rf, pad_mask)    # [n_out, n_ref]
        out_of_field = ((real_count == 0) & (pad_count > 0)).to(torch.float32)  # [n_out, n_ref]
        # Non-persistent: regenerated from fixed geometry on construction.
        self.register_buffer("out_of_field_mask", out_of_field, persistent=False)
        # 'learned': one value per reference cell. 'learned_shared': a single
        # per-channel value (broadcast over cells in _apply_learned_padding).
        n_cols = 1 if self.padding == "learned_shared" else n_ref
        self.pad_embed = nn.Parameter(torch.zeros(self.in_channels, n_cols))

    def _apply_learned_padding(self, knn_features):
        """Add learned padding to the out-of-field reference cells.

        ``knn_features``: [batch, n_out, in_channels * n_ref], laid out
        (channel-major, cell-minor) -- the same layout as ``self.weight`` -- so
        the term added at flat index ``c * n_ref + v`` is multiplied by that
        cell's weights in the subsequent linear projection.
        """
        n_out, n_ref = self.out_of_field_mask.shape
        pad = self.out_of_field_mask.unsqueeze(1) * self.pad_embed.unsqueeze(0)  # [n_out, in_channels, n_ref]
        pad = pad.reshape(n_out, self.in_channels * n_ref)  # (channel-major, cell-minor)
        return knn_features + pad.unsqueeze(0).to(knn_features.dtype)

    def _pad_and_gather_knns(self, X_l):
        """
        Pad input features and gather KNN features.
        
        Args:
            X_l (torch.Tensor): Input features.
            
        Returns:
            torch.Tensor: Gathered KNN features.
        """
        # Pad with zeros for padding units
        X_l_padded = torch.concatenate([X_l, torch.zeros(X_l.shape[0], X_l.shape[1], 1, device=X_l.device, dtype=X_l.dtype)], dim=2)
        
        # Gather KNN features
        batch_size, d_l, N_l = X_l.shape
        N_l_plus_1 = self.knn_indices_pad_token.shape[1]
        knn_features = torch.gather(X_l_padded, 2, self.knn_indices_pad_token.reshape(1, 1, -1).expand(batch_size, d_l, -1))
        knn_features = knn_features.reshape(batch_size, d_l, self._k, N_l_plus_1)
        
        return knn_features

    def _apply_local_rf(self, knn_features):
        """
        Reweights features according to local receptive field (i.e., kernel mapping to align KNNs and reference kernel)
        
        Args:
            knn_features (torch.Tensor): KNN features tensor to process
            
        Returns:
            torch.Tensor: Processed features after applying local receptive field
        """
        self.local_rf = self.local_rf.to(device=knn_features.device, dtype=knn_features.dtype)

        # new einsum approach
        b, d, k, n = knn_features.shape
        v = self.local_rf.shape[2]
        # the permute makes einsum faster
        knn_features = rearrange(knn_features, 'b d k n -> (b d) n k')
        knn_features = torch.einsum('ank,nkv->anv', knn_features, self.local_rf)
                    
        knn_features = rearrange(knn_features, '(b d) n v -> b n (d v)', b=b, d=d)

        return knn_features # (batch, num_coords, k*d_l)

    def forward(self, X_l):
        """
        Apply convolution using k-nearest neighbors.
        
        Args:
            X_l (torch.Tensor): Node features from layer l [batch, d_l, N_l]
            
        Returns:
            torch.Tensor: Node features from layer l+1 [batch, d_l+1, N_l+1]
        """
        
        knn_features = self._pad_and_gather_knns(X_l)

        # apply local RF.  shape: [batch, d_l, k, num_coords] -> [batch, d_l, v, n], where v is the number of reference coordinates
        knn_features = self._apply_local_rf(knn_features)

        # Optional: fill out-of-field reference cells with learned padding.
        if getattr(self, "_learned_padding", False):
            knn_features = self._apply_learned_padding(knn_features)

        # Apply the linear transformation to map to d_l+1 features. With
        # per-ring kernels each coordinate uses its ring's kernel; otherwise a
        # single shared kernel is applied to all coordinates.
        if getattr(self, "per_ring_kernel", False):
            X_out = self._apply_per_ring_linear(knn_features)  # Shape: [batch, d_l+1, num_coords]
        else:
            X_out = F.linear(knn_features, self.weight, self.bias).transpose(1,2)  # Shape: [batch, d_l+1, num_coords]

        return X_out

    def compute_reference_coords(self, arch_flag):
        """
        Compute reference coordinates for the convolution kernel.
        
        The reference coordinates define the shape of the convolution kernel and can be:
        - A square grid matching the KNN size
        - A circular grid matching the KNN size 
        - The first neighborhood of coordinates (foveal reference)

        Args:
            arch_flag (str): Architecture flag indicating reference coordinate style. Checks for containing:
                - 'fovref': Use first neighborhood as reference (not typically used)
                - 'circref': Use circular reference grid (not typically used)
                - 'doubleres': Double the resolution of reference grid (smoother alignment)
                - default: Use square reference grid of same total number of elements as the neighborhood
                
        Note:
            Sets self.ref_coords (torch.Tensor): Reference coordinates tensor of shape [num_ref_coords, 2]
        """
        if self.ref_frame_side_length is not None:
            assert not len(arch_flag), 'self.ref_frame_side_length is incompatible with use of arch_flag'
            side_length = self.ref_frame_side_length
            ref_coords = torch.linspace(-1, 1, side_length)
            ref_coords = torch.meshgrid(ref_coords, ref_coords)
            ref_coords = torch.stack(ref_coords, -1).reshape(-1, 2)
        else:
            # square reference grid of similar size to the KNN graph
            side_length = torch.ceil(torch.sqrt(self.k)).long()

            if 'doubleres' in arch_flag:
                side_length = 2*side_length

            ref_coords = torch.linspace(-1, 1, side_length)
            ref_coords = torch.meshgrid(ref_coords, ref_coords)
            ref_coords = torch.stack(ref_coords, -1).reshape(-1, 2)

        self.ref_grid_size = side_length

        self.ref_coords = ref_coords.to(self.device, dtype=self.in_coords.dtype)

    def compute_local_rf(self):
        """
        Compute local receptive field weights for each output coordinate.
        
        This method computes the mapping between input KNN coordinates and reference coordinates
        to determine how each input point contributes to the output. Each KNN neighbor is mapped
        to its nearest reference grid position (one-hot assignment).
            
        Returns:
            torch.Tensor: Local receptive field weights of shape [n_out, k, n_ref] where:
                - n_out is number of output coordinates
                - k is number of nearest neighbors
                - n_ref is number of reference coordinates
        """

        # get all KNN coordinates
        knn_coords = self.sample_coords[self.knn_indices]
        if self.sample_cortex:
            out_coords = self.out_coords.cortical
            out_coords_cart = self.out_coords.cartesian

            # make sure the RFs are aligned in cartesian space
            knn_coords_cart = self.sample_coords_cart[self.knn_indices]
            tmp_theta = torch.arctan2(knn_coords_cart[:,:,1] - out_coords_cart[:,1], knn_coords_cart[:,:,0] - out_coords_cart[:,0]) #.squeeze()
            tmp_r = []
            for i in range(knn_coords.shape[1]):
                tmp_r_ = torch.cdist(out_coords[i].unsqueeze(0), knn_coords[:,i]).squeeze(0)
                tmp_r.append(tmp_r_)
            tmp_r = torch.stack(tmp_r, dim=1)
            tmp_r[self.knn_indices < 0] = 0 # only used for PartitioningPatchEmbedding, where we pad with non-padding units to fill neighborhoods to a standard size
            knn_coords = torch.stack([tmp_r*torch.cos(tmp_theta), tmp_r*torch.sin(tmp_theta)], dim=2)

        # normalize KNN coords to -1,1 within each neighborhood
        knn_coords_min = knn_coords.min(dim=0, keepdim=True)[0]
        knn_coords_max = knn_coords.max(dim=0, keepdim=True)[0]
        self.knn_coords = 2 * (knn_coords - knn_coords_min) / (knn_coords_max - knn_coords_min + 1e-6) - 1

        if self.rf_interp == "bilinear":
            return self._bilinear_local_rf()

        # compute distance between KNN coords and reference coords
        distances = torch.cdist(self.knn_coords, self.ref_coords) # (k, n_out, n_ref)
        
        dim = 2
        # map the nearest neighbor on the reference grid to the output unit
        local_rf = torch.zeros_like(distances)
        nn = torch.argmin(distances, dim=dim)
        local_rf.scatter_(dim, nn.unsqueeze(dim), 1)

        local_rf = local_rf.permute(1,0,2) # (n_out, k, n_ref)

        return local_rf

    def _bilinear_local_rf(self):
        """Bilinear version of ``compute_local_rf``: each neighbor spreads weight 1 over the
        4 reference cells around it, with bilinear weights, instead of all of it on the
        nearest cell. Requires the square reference grid of ``compute_reference_coords``
        (``linspace(-1, 1, S)`` per axis, 'ij' meshgrid, so cell (a, b) is index a * S + b).

        Returns:
            torch.Tensor: [n_out, k, n_ref], each neighbor's weights summing to 1.
        """
        S = int(self.ref_grid_size)
        assert S >= 2, f"rf_interp='bilinear' needs a reference grid of side >= 2, got {S}"
        u = (self.knn_coords + 1) / 2 * (S - 1)              # (k, n_out, 2) fractional cell index per axis
        i0 = u.floor().clamp(0, S - 2).long()                # lower corner of the surrounding cell square
        t = (u - i0).clamp(0, 1)                             # position inside that square
        local_rf = torch.zeros(*u.shape[:2], S * S, device=u.device, dtype=u.dtype)
        for da in (0, 1):
            for db in (0, 1):
                w = (t[..., 0] if da else 1 - t[..., 0]) * (t[..., 1] if db else 1 - t[..., 1])
                cell = (i0[..., 0] + da) * S + (i0[..., 1] + db)
                local_rf.scatter_add_(2, cell.unsqueeze(2), w.unsqueeze(2))
        return local_rf.permute(1, 0, 2)  # (n_out, k, n_ref)

    def __repr__(self):
        n_ref = self.ref_coords.shape[0]
        return f'{self.__class__.__name__}(\n\tin_channels={self.in_channels}\n\tout_channels={self.out_channels}\n\tk={self.k}\n\tn_ref={n_ref}\n\tin_coords={self.in_coords}\n\tout_coords={self.out_coords}\n\tsample_cortex={self.sample_cortex}\n)'


def get_in_out_coords(in_res, fov, cmf_a, stride, style='isotropic', auto_match_cart_resources=1, in_cart_res=None, device='cuda', in_coords=None, force_out_match_less_than=True, max_out_coord_val=1):
    """
    Convenience function to generate input and output coordinates for KNN layers.
    
    Args:
        in_res (int): Input resolution.
        fov (float): Field of view diameter in degrees.
        cmf_a (float): a parameter in CMF: M(r) = 1/(r+a). Smaller = stronger foveation.
        stride (int): Stride factor for downsampling.
        style (str, optional): Sampling style. Defaults to 'isotropic'.
        auto_match_cart_resources (int, optional): Auto-match parameter. Defaults to 1.
        in_cart_res (int, optional): Input cartesian resolution. Defaults to None.
        device (str, optional): PyTorch device. Defaults to 'cuda'.
        in_coords (SamplingCoords, optional): Pre-computed input coordinates. Defaults to None.
        force_out_match_less_than (bool, optional): If auto_match_cart_resources, this determines whether the output number is constrained to not be greater than the target cartesian resolution (if false, chooses the closest match, which could be greater). Defaults to True.
        max_out_coord_val (int or str, optional): Maximum output coordinate value. Defaults to 1.
        
    Returns:
        tuple: A tuple containing:
            - SamplingCoords: Input coordinates
            - SamplingCoords: Output coordinates
            - int: Output cartesian resolution
    """
    # Generate input coordinates if not provided
    if in_coords is None:
        if auto_match_cart_resources:
            in_res, in_cart_res = auto_match_num_coords(fov, cmf_a, in_cart_res, style, auto_match_cart_resources, device, force_less_than=True, quiet=True)
        in_coords = SamplingCoords(fov, cmf_a, in_res, device=device, style=style)

    if max_out_coord_val == 'auto':
        tmp_max_val = 1
    else:
        tmp_max_val = max_out_coord_val
    
    # Generate output coordinates with stride
    out_coords, _, out_cart_res = in_coords.get_strided_coords(stride, auto_match_cart_resources=auto_match_cart_resources, in_cart_res=in_cart_res, force_less_than=force_out_match_less_than, max_val=tmp_max_val)

    if max_out_coord_val == 'auto':
        # adjust to put between final two radii
        final_out_coord_val = (torch.unique(out_coords.polar[:,0])[-1] + torch.unique(out_coords.polar[:,0])[-2]) / 2
        
        out_coords, _, out_cart_res = in_coords.get_strided_coords(stride, auto_match_cart_resources=auto_match_cart_resources, in_cart_res=in_cart_res, force_less_than=force_out_match_less_than, max_val=final_out_coord_val)

    
    return in_coords, out_coords, out_cart_res


class KNNPartitioningPatchEmbedding(KNNConvLayer):
    """KNN patch embedding whose neighbourhoods partition the input samples: every sample
    lands in at least one patch, with the smallest k that achieves it (fovi 27b1bde's
    ``KNNPartitioningPatchEmbedding``, with the ``forward`` it inherited from
    ``KNNPatchEmbedding``)."""

    @staticmethod
    def _stable_neighbor_order(distances):
        """Order neighbors by distance, breaking ties by candidate index."""
        return torch.argsort(distances, dim=0, stable=True)

    @classmethod
    def _minimum_k_covering_inputs(
            cls, distances, num_inputs, neighbor_order=None):
        num_outputs = distances.shape[1]
        lower_bound = max(1, math.ceil(num_inputs / num_outputs))
        upper_bound = min(num_inputs, distances.shape[0])
        if neighbor_order is None:
            neighbor_order = cls._stable_neighbor_order(distances)
        elif neighbor_order.shape != distances.shape:
            raise ValueError(
                "Neighbor order must have the same shape as the distances"
            )

        def covers_all_inputs(num_neighbors):
            indices = neighbor_order[:num_neighbors]
            covered = torch.unique(indices[indices < num_inputs])
            return covered.numel() == num_inputs

        if not covers_all_inputs(upper_bound):
            raise RuntimeError(
                "No supported KNN size covers every non-padding input"
            )

        while lower_bound < upper_bound:
            candidate = (lower_bound + upper_bound) // 2
            if covers_all_inputs(candidate):
                upper_bound = candidate
            else:
                lower_bound = candidate + 1
        return lower_bound

    def __init__(self, 
                in_channels: int,
                embed_dim: int,
                in_res: int,
                fov: float,
                cmf_a: float,
                style: str = 'isotropic',
                auto_match_cart_resources: bool = True,
                in_cart_res: int = 224,
                cart_patch_size = 16,
                device='cuda',
                force_patches_less_than_matched=True,
                transposed=False,
                max_coord_val='auto',
                sample_cortex='geodesic',
                 **kwargs,
                 ):
        """Initialize KNN partitioning patch embedding layer.
        
        Args:
            in_channels: Number of input channels
            embed_dim: Embedding dimension for tokens
            in_res: Input resolution
            fov: Field of view parameter for foveated sampling
            cmf_a: a parameter controlling foveated sampling via the CMF
            style: Sampling style ('isotropic', etc.)
            auto_match_cart_resources: Whether to automatically match cartesian resources
            in_cart_res: Resolution of input cartesian grid
            cart_patch_size: Size of cartesian patches
            device: Device to run on
            force_patches_less_than_matched: Whether to force patches to be less than matched
            transposed: Whether to transpose output
            max_coord_val: Maximum coordinate value
            sample_cortex: Cortex sampling method
            **kwargs: Additional arguments passed to parent class
        """    

        stride = cart_patch_size # match number of tokens exactly with cartesian version

        in_coords, out_coords, out_cart_res = get_in_out_coords(in_res, fov, cmf_a, stride, style=style, auto_match_cart_resources=auto_match_cart_resources, in_cart_res=in_cart_res, device=device, force_out_match_less_than=force_patches_less_than_matched, max_out_coord_val=max_coord_val)

        k = int(len(in_coords) / len(out_coords)) # set temporary k for use in geodesic dist computation, if necessary
        KNNConvLayer.__init__(self, in_channels, embed_dim, k, in_coords, out_coords, device=device, sample_cortex=sample_cortex, **kwargs) # temporary init to set distances
        distances = self._compute_all_distances()
        neighbor_order = self._stable_neighbor_order(distances)

        k = self._minimum_k_covering_inputs(
            distances,
            len(in_coords),
            neighbor_order=neighbor_order,
        )
        knn_indices = neighbor_order[:k]
        knn_distances = torch.gather(distances, 0, knn_indices)

        KNNConvLayer.__init__(
            self,
            in_channels,
            embed_dim,
            k,
            in_coords,
            out_coords,
            device=device,
            sample_cortex=sample_cortex,
            _precomputed_knns=(knn_indices, knn_distances),
            **kwargs,
        )
        self.distances = distances

        print(f'minimum k to use all inputs: {k}')
        
        self.transposed = transposed 

    def forward(self, x):
        # reshape tokens from grid if needed
        if len(x.shape) == 4:
            x = rearrange(x, 'b n h w -> b n (h w)')
        tokens = super().forward(x)
        # flip seq and embed dims to match standard transformer style
        if not self.transposed:
            tokens = rearrange(tokens, 'b d n -> b n d')

        return tokens
