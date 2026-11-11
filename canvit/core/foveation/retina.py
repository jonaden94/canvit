"""The retina: foveated sampling of an image around a fixation.

Ported from fovi 27b1bde (``RetinalTransform`` from ``fovi/sensing/retina.py``), with
the options canvit never sets removed: foveated colour decay (``sigma``), pre/post
transforms, ``no_color_val`` and sampler kwargs. Without them fovi's forward reduces to
sampling and the final dtype cast, which is what is left here; the rest is verbatim.
"""

import numpy as np
import torch
import torch.nn as nn

from canvit.core.foveation.coords import find_desired_res
from canvit.core.foveation.samplers import GaussianKNNGridSampler, GridSampler, KNNGridSampler


class RetinalTransform(nn.Module):
    """
    Spatially non-uniform (foveated) sampling of the visual field.

    Foveated spatial sampling is based on isotropic cortical magnification of the form CMF=1/(r+a),where: 
        - r=polar radius (eccentricity)
        - a is a parameter that controls the degree of foveation. smaller = more foveation. as a->infinity, we get uniform sampling.
        - Equal sampling in cortical space is assumed, and the visual coordinates are computed by back-projection to acquire the foveated sampling grid.
    This uses a CorticalSensorManifold module to represent the V1-like manifold of retina-like samples.
    """

    def __init__(self, resolution, start_res=256,
                 fov=16, cmf_a=0.5, 
                 style='isotropic',
                 sampler='grid_nn',
                 fixation_size=None, 
                 device='cuda', 
                 dtype=torch.float, 
                 auto_match_cart_resources=True,
                 ):
        """
        Initialize the RetinalTransform module.
        
        Args:
            resolution (int): Target resolution for the retinal transform.
            start_res (int, optional): Starting resolution. Defaults to 256.
            fov (float, optional): Field of view diameter in degrees. Defaults to 16.
            cmf_a (float, optional): Cortical magnification factor parameter. Defaults to 0.5.
            style (str, optional): Sampling style. Defaults to 'isotropic'.
            sampler (str, optional): Sampler type. Defaults to 'grid_nn'.
            fixation_size (int, optional): Fixation size in pixels. Defaults to None.
            device (str, optional): Device to use. Defaults to 'cuda'.
            dtype (torch.dtype, optional): Data type. Defaults to torch.float.
            auto_match_cart_resources (bool, optional): Whether to auto-match cartesian resources. Defaults to True.
        """
        super().__init__()
        self.cmf_a = cmf_a
        self.fov = fov
        full_fov = self.fov
        self.fixation_size = start_res if fixation_size is None else fixation_size # default forward window (not a cap; forward may pass any size). The sampling-grid resolution is res_mult*resolution, independent of this.
        self.start_res = start_res
        self.device = device
        self.dtype = dtype

        if auto_match_cart_resources != 0:
            in_resolution = resolution
            num_coords = resolution**2
            if 'fixn' in style:
                # if fixn, we are going to force the resolution later
                resolution = resolution
            else:
                # otherwise we determine it now
                resolution, num_coords = find_desired_res(fov, cmf_a, num_coords, style=style, device=self.device, force_less_than=True, quiet=True)

        self.resolution = resolution

        # for compatibility
        self.mode = f'pointcloud_{style}'
        self.out_size = resolution

        # for use with standard CNNs
        if '_as_grid' in self.mode:
            self.reshape_as_grid = True
        else:
            self.reshape_as_grid = False

        if sampler == 'gaussian_pooling':
            self.sampler = GaussianKNNGridSampler(self.fov, self.cmf_a, resolution, fixation_size=self.fixation_size, device=device, style=style)
        elif sampler == 'pooling':
            self.sampler = KNNGridSampler(self.fov, self.cmf_a, resolution, fixation_size=self.fixation_size, device=device, style=style)
        elif sampler == 'grid_nn':
            self.sampler = GridSampler(self.fov, self.cmf_a, resolution, device=device, mode='nearest', style=style)
        elif sampler == 'grid_bilinear':
            self.sampler = GridSampler(self.fov, self.cmf_a, resolution, device=device, mode='bilinear', style=style)

        else:
            raise ValueError(f'Invalid sampler: {sampler}')
        self.device = device
        self.dtype = dtype

        self.scatter_sizes = self.sampler.coords.get_scatter_sizes().cpu().numpy()

    @torch.autocast(device_type='cuda', enabled=False)
    def forward(self, x, fix_loc, fixation_size=None, **kwargs):
        """
        Forward pass of the retinal transform.
        
        Args:
            x (torch.Tensor): Input tensor.
            fix_loc (torch.Tensor or tuple): Fixation location.
            fixation_size (int, optional): Fixation size. Defaults to None.
            **kwargs: Additional arguments.
            
        Returns:
            torch.Tensor: Transformed tensor.
        """

        # check fixation_size
        fixation_size = self._check_fixation_size(fixation_size, x.shape[0])
            
        # check fix_loc
        fix_loc = self._check_fix_loc(fix_loc, x.shape[0])
        
        x = self.sampler(x, fix_loc=fix_loc, fixation_size=fixation_size, **kwargs)

        if self.reshape_as_grid:
            x = x.reshape(x.shape[0], x.shape[1], self.resolution, self.resolution)

        return x.to(self.dtype)

    def _check_fixation_size(self, fixation_size, batch_size):
        """
        Validate and format fixation size for batch processing.
        
        Args:
            fixation_size (int, tuple, or None): Fixation size specification.
            batch_size (int): Number of samples in the batch.
            
        Returns:
            np.ndarray: Formatted fixation size array of shape (batch_size, 2).
        """
        if fixation_size is None or (isinstance(fixation_size, str) and fixation_size == 'none'):
            if hasattr(self.fixation_size, '__len__'):
                fixation_size = np.array(self.fixation_size)
            else:
                fixation_size = np.array([self.fixation_size, self.fixation_size])
        elif isinstance(fixation_size, int) or isinstance(fixation_size, float):
            fixation_size = np.array([fixation_size, fixation_size])
        elif isinstance(fixation_size, torch.Tensor):
            # Must precede the generic ``len()==2`` branch: a per-sample fixation
            # tensor of shape [B, 2] has ``len == B``, but when B == 2 it would
            # otherwise fall into ``np.array(tensor)`` and raise on CUDA tensors.
            fixation_size = fixation_size.cpu().numpy()
        elif hasattr(fixation_size, '__len__') and len(fixation_size) == 2:
            fixation_size = np.array(fixation_size)
        fixation_size = fixation_size.squeeze()
        if len(fixation_size.shape) == 1:
            if fixation_size.shape[0] == 2:
                fixation_size = np.expand_dims(np.array(fixation_size), axis=0).repeat(batch_size, axis=0)
            elif fixation_size.shape[0] == batch_size:
                fixation_size = np.expand_dims(np.array(fixation_size), axis=1).repeat(2, axis=1)
            else:
                raise ValueError(f'fixation_size shape: {fixation_size.shape}')
        else:
            assert fixation_size.shape[0] == batch_size
        
        return fixation_size

    def _check_fix_loc(self, fix_loc, batch_size):
        """
        Validate and format fixation location for batch processing.
        
        Args:
            fix_loc (tuple, list, torch.Tensor, or None): Fixation location specification.
            batch_size (int): Number of samples in the batch.
            
        Returns:
            torch.Tensor: Formatted fixation location tensor of shape (batch_size, 2).
        """
        if fix_loc is None:
            fix_loc = np.array([0.5, 0.5])
        elif isinstance(fix_loc, float) or isinstance(fix_loc, int):
            fix_loc = np.array([fix_loc, fix_loc])
        elif isinstance(fix_loc, torch.Tensor):
            fix_loc = fix_loc.cpu().numpy()
        elif isinstance(fix_loc, list) or isinstance(fix_loc, tuple):
            fix_loc = np.array(fix_loc)
        if len(fix_loc.squeeze().shape) == 1:
            fix_loc = np.expand_dims(np.array(fix_loc.squeeze()), axis=0).repeat(batch_size, axis=0)
        else:
            assert fix_loc.shape[0] == batch_size

        fix_loc = torch.tensor(fix_loc, dtype=self.dtype, device=self.device)

        return fix_loc
