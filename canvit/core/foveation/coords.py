"""Visual-field sampling coordinates of the foveated sensor.

Ported from fovi 27b1bde (``fovi/sensing/coords.py``, plus ``normalize`` from
``fovi/utils``): only what canvit's patchers reach. Function bodies are verbatim, so
the sampling grids, and the KNN neighbourhoods built on them, are bit-identical to
fovi's.
"""

import numpy as np
import torch

from canvit.core.foveation.manifold import vis_cartesian_to_cortical_cartesian_coords as vis_to_sensor_manifold


def normalize(x, min=None, max=None, dim=None):
    """Normalize values to [0, 1] range using min-max normalization.
    
    Computes (x - min) / (max - min). If min/max are not provided,
    they are computed from the data.
    
    Args:
        x (torch.Tensor or np.ndarray): Input values to normalize.
        min (float or torch.Tensor, optional): Minimum value for normalization.
            If None, computed from x. Defaults to None.
        max (float or torch.Tensor, optional): Maximum value for normalization.
            If None, computed from x. Defaults to None.
        dim (int, optional): Dimension along which to compute min/max.
            If None, uses global min/max. Defaults to None.
            
    Returns:
        torch.Tensor or np.ndarray: Normalized values in [0, 1] range.
            Returns same type as input.
    """
    to_numpy = False
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        to_numpy = True

    if min is not None and not isinstance(min, torch.Tensor):
        min = torch.tensor(min, device=x.device)
    if max is not None and not isinstance(max, torch.Tensor):
        max = torch.tensor(max, device=x.device)

    if dim is not None:
        min = x.min(dim).values if min is None else min
        max = x.max(dim).values if max is None else max
        min = min.unsqueeze(dim)
        max = max.unsqueeze(dim)
    else:
        min = x.min() if min is None else min
        max = x.max() if max is None else max

    out = (x - min) / (max - min)
    if to_numpy:
        out = out.cpu().numpy()
    return out


class SamplingCoords():
    """Object for storing multiple coordinate systems relevant to a set of visual field samples.
    
    Args:
        fov (float): Field-of-view in degrees.
        cmf_a (float): A parameter from the CMF: M(r)=1/(r+a). Smaller a = stronger foveation.
        res (int): Resolution, corresponding to the side length of a cartesian grid or the number of radii in polar (before any adjustments are made to match a target cartesian grid).
        device (str): What device to operate on.
        style (str): What type of sampling.
        dtype (torch.dtype): What data type to use.
    """

    def __init__(self, fov, cmf_a, res, device='cpu', style='isotropic', 
                 dtype=torch.float,
                 max_val=1,
                 ):
        
        self.fov = fov
        self.cmf_a = cmf_a
        self.resolution = res
        self.device = device
        self.style = style
        self.dtype = dtype
        self.max_val = max_val

        if res == 1:
            self.cartesian = torch.zeros(1, 2, device=device, dtype=dtype)
            self.polar = torch.zeros(1, 2, device=device, dtype=dtype)
            self.plotting = torch.zeros(1, 2, device=device, dtype=dtype)
            self.cortical = torch.zeros(1, 3, device=device, dtype=dtype)
        else:
            self.cartesian, self.polar, self.plotting = get_sampling_coords(fov, cmf_a, res, device=device, style=style, max_val=max_val)
            # image format (row, col)
            self.cartesian_rowcol = xy_to_rowcol(self.cartesian, do_norm=False, format='-11')

            self.cartesian_pad_coords = self.pad_cartesian(device=device, dtype=dtype)

            if 'logpolar' in style:
                # log polar image, meaning it does not use the proper sensor manifold, but a simplified one
                self.cortical = self.polar.clone()
                radius = ((self.cartesian[:,0]**2 + self.cartesian[:,1]**2)**.5)*self.fov/2
                cmf_a_tensor = torch.tensor(self.cmf_a)
                self.cortical[:,0] = (torch.log(radius + self.cmf_a) - torch.log(cmf_a_tensor))/(torch.log(self.fov/2 + cmf_a_tensor) - torch.log(cmf_a_tensor))
                self.cortical = normalize(self.cortical, dim=0)
                self.plotting = self.cortical
            elif 'uniform' in style:
                self.cortical = None
            else:
                # cortical coordinates to be used for sampling RFs
                self.cortical = vis_to_sensor_manifold(self.cartesian.cpu().numpy(), cmf_a, fov, as_tensor=True, device=device) 
                self.cortical_pad_coords = vis_to_sensor_manifold(self.cartesian_pad_coords.cpu().numpy(), cmf_a, fov, as_tensor=True, device=device).to(dtype=dtype)

        self.cartesian = self.cartesian.to(dtype)
        if self.cortical is not None:
            self.cortical = self.cortical.to(dtype)
        self.polar = self.polar.to(dtype)
        self.plotting = self.plotting.to(dtype)
        self.shape = self.cartesian.shape

    def pad_cartesian(self, padding_distance=0.5, device=None, dtype=None):
        """Generate additional cartesian coordinates for padding around the sampling grid.
        
        Args:
            padding_distance (float): Distance to extend beyond the current sampling area.
            device (str, optional): Device to place the coordinates on. Defaults to None.
            dtype (torch.dtype, optional): Data type for the coordinates. Defaults to None.
            
        Returns:
            torch.Tensor: Additional cartesian coordinates for padding.
        """
        pad_coords = []
        sorted_radii = torch.sort(torch.unique(self.polar[:,0])).values
        radius_diff = sorted_radii[-1] - sorted_radii[-2]
        start_radius = sorted_radii[-1] + radius_diff
        for radius in torch.arange(start_radius, start_radius+padding_distance, radius_diff):
            for angle in torch.arange(0, 2*np.pi, radius_diff):
                pad_coords.append(torch.tensor([radius*np.cos(angle), radius*np.sin(angle)]))
        pad_coords = torch.stack(pad_coords, 0).to(device=device, dtype=dtype)
        return pad_coords

    def get_strided_coords(self, stride, auto_match_cart_resources=0, in_cart_res=None, force_less_than=True, max_val=1):
        """Return a strided version of the coordinates.
        
        Args:
            stride (int): Stride factor for downsampling coordinates.
            auto_match_cart_resources (int): Automatic matching parameter for cartesian resources.
            in_cart_res (int, optional): Input cartesian resolution. Required for 'fixn' style.
            force_less_than (bool, optional): Whether to force less than target resolution. Defaults to True.
            max_val (float, optional): Maximum value for coordinates. Defaults to 1.
            
        Returns:
            tuple: A tuple containing:
                - SamplingCoords: New SamplingCoords object with strided coordinates.
                - int: Number of output radii.
                - int: Corresponding output cartesian resolution.
        """
        out_cart_res = None
        if 'fixn' in self.style: 
            """
            fixn means to fix the number (n) of sampling coordinates exactly to that of the target cartesian grid
            this requires adding in some additional samples across radii, deviating a bit from local isotropy
            """
            assert in_cart_res is not None, 'in_cart_res must be provided if using fixn style'
            out_radii = in_cart_res//stride
            out_cart_res = in_cart_res//stride
        elif auto_match_cart_resources > 0 and 'fixn' not in self.style and in_cart_res is not None:
            """
            this means to get as close as possible to resolution of the cartesian grid (same or less), while maintaining isotropic sampling
            """
            assert in_cart_res is not None, 'in_cart_res must be provided if auto_match_cart_resources is True'
            out_cart_res = in_cart_res//stride
            out_radii, num_coords = find_desired_res(self.fov, self.cmf_a, out_cart_res**2, style=self.style, device=self.device, force_less_than=force_less_than, quiet=True)
        else:
            """
            no matching to cartesian sampling resolution -- not recommended since it makes comparisons very difficult
            """
            out_radii = self.resolution // stride
        if out_radii < 1:
            raise ValueError(f'Out radii decreased to less than 1: {out_radii}')
        
        out_coords = SamplingCoords(self.fov, self.cmf_a, out_radii, self.device, self.style, self.dtype, max_val=max_val)

        return out_coords, out_radii, out_cart_res

    def get_scatter_sizes(self):
        """Calculate approximate size to be used for plotting in scatter plots.

        Scales linearly with eccentricity, as occurs in KNN-sampling of the sensor manifold. 
        However, this is not precisely tuned to the particular warping parameters.
        
        Returns:
            torch.Tensor: Receptive field sizes for each sampling point.
        """
        # Calculate RF sizes based on eccentricity
        sizes = torch.zeros_like(self.polar[:,0])
        for i, radius in enumerate(self.polar[:,0]):
            # RF size increases with eccentricity
            sizes[i] = radius * self.fov / self.resolution
        return sizes

    def subset(self, keep_mask):
        """Keep only the coordinates selected by ``keep_mask`` (in place).

        ``keep_mask`` is a boolean tensor of length ``len(self)`` (or a long
        index tensor) selecting which sampling coordinates to retain. Only the
        per-coordinate tensors (``cartesian``, ``polar``, ``plotting``,
        ``cortical``, ``cartesian_rowcol``) are subset; the *padding* coordinate
        tensors (``cartesian_pad_coords`` / ``cortical_pad_coords``) describe the
        out-of-field padding ring and are independent of which output
        coordinates are kept, so they are left untouched. ``shape`` is refreshed.

        Returns ``self`` so callers can chain.
        """
        idx = keep_mask
        self.cartesian = self.cartesian[idx]
        self.polar = self.polar[idx]
        self.plotting = self.plotting[idx]
        if getattr(self, "cartesian_rowcol", None) is not None:
            self.cartesian_rowcol = self.cartesian_rowcol[idx]
        if getattr(self, "cortical", None) is not None:
            self.cortical = self.cortical[idx]
        self.shape = self.cartesian.shape
        return self

    def __len__(self):
        """Return the number of sampling coordinates."""
        return self.cartesian.shape[0]

    def __repr__(self):
        """String representation of the SamplingCoords object."""
        return f'SamplingCoords(length={len(self)}, fov={self.fov}, cmf_a={self.cmf_a}, resolution={self.resolution}, style={self.style})'


def get_isotropic_sampling_coords(fov, cmf_a, res, circular=True, device='cpu', constant_num_angles=False, force_n_points=None, max_norm_rad=1):
    """Sample coordinates isotropically with the cortical magnification function of the complex log mapping w=log(z+a), where z=x+iy.

    Args:
        fov (float): Field of view diameter in degrees.
        cmf_a (float): A parameter in the CMF controlling foveation; smaller = stronger foveation.
        res (int): Number of sampling points.
        circular (bool, optional): If True, the sampling is circular, otherwise it is square. Defaults to True.
        device (str, optional): Device to run the computation on. Defaults to 'cpu'.
        constant_num_angles (bool, optional): If True, the number of angles is constant for all radii, implementing log polar image sampling. Defaults to False.
        force_n_points (int, optional): If not None, forces the number of points to exactly this value. Useful for controlled comparisons with other sensors. Defaults to None.
        max_norm_rad (float, optional): Maximum normalized radius. Defaults to 1.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: Sampling cartesian coordinates in visual space, normalized to (-1,1).
            - torch.Tensor: Sampling polar coordinates in visual space.
            - torch.Tensor: Plotting coordinates in complex log space, useful for visualizing the sampling.
    """

    if force_n_points is not None:
        res, _ = find_desired_res(fov, cmf_a, force_n_points, style='isotropic', device=device, quiet=True)

    # compute log-sampled radii, and angles for each radius
    radius, n_angles = _compute_isotropic_r_and_num_theta(fov, cmf_a, res, circular=circular, device=device)
    if constant_num_angles:
        # overwrite isotropic angle sampling to use standard log polar image sampling
        n_angles = torch.tensor([res]*res, device=device)

    radius = radius * max_norm_rad

    # if force_n_points is not None, we need to adjust the number of angles for the last radius
    if force_n_points is not None:
        # fix # of points by removing angles evenly across radii
        diff = n_angles.sum() - force_n_points
        # insert/remove new radii
        if diff > 0:
            add = -1
            action = 'removing'
            verb = 'from'
        else:
            add = 1
            action = 'adding'
            verb = 'to'
        if diff != 0:
            rad_idx = []
            print(np.abs(diff))
            while len(rad_idx) < np.abs(diff):
                this_diff = np.minimum(np.abs(diff) - len(rad_idx), len(radius)-1).item()
                print(f'{action} {this_diff} angles {verb} {len(radius)} radii')
                # set some heuristic choices for which radii to presever
                rad_min = int(0.2*len(radius))
                rad_idx = np.concatenate([rad_idx, np.random.choice(np.arange(rad_min, len(radius)), size=np.minimum(this_diff, len(radius)-rad_min), replace=False)])
            rad_idx = rad_idx.astype(int)
            for ii in range(np.abs(diff)):
                # make sure we don't remove the only angle from a radius
                this_idx = rad_idx[ii]
                while n_angles[this_idx] <= 1 and action == 'removing':
                    this_idx = this_idx + 1
                n_angles[this_idx] = n_angles[this_idx] + add
            assert n_angles.min() > 0, 'some radii have no angles'
                
    # compute angles and store coordinates
    coords = []
    polar_coords = []
    hemi_inds = []
    for ii, radius_i in enumerate(radius):
        angles = np.linspace(0, 2*np.pi, n_angles[ii], endpoint=False)
        angles = torch.tensor(angles).to(device)
        for angle in angles:
            polar_coords.append(torch.stack([radius_i, angle]))
            coords.append(torch.stack([radius_i*torch.cos(angle), radius_i*torch.sin(angle)]))
            # ensure right goes to 0, left goes to 1
            hh = radius_i*torch.cos(angle) < 0 
            hemi_inds.append(hh)

    coords = torch.stack(coords)
    polar_coords = torch.stack(polar_coords)
    hemi_inds = torch.tensor(hemi_inds)

    # use log(z+a) model to compute plotting coordinates (i.e. cortical visualization) 
    fov_coords = coords*(fov/2)
    plotting_coords = torch.log(torch.abs(fov_coords[:,0]) + 1j*fov_coords[:,1] + cmf_a)
    plotting_coords = torch.stack([plotting_coords.real, -plotting_coords.imag],1)

    max_fov_rad = np.log(fov/2 + cmf_a)
    
    # make plotting coords separated nicely across hemifields/hemispheres
    std = torch.std(plotting_coords[:,0])*.5
    plotting_coords[hemi_inds == 1,0] = std + max_fov_rad - plotting_coords[hemi_inds == 1,0]
    plotting_coords[hemi_inds == 0,0] = plotting_coords[hemi_inds == 0,0] - (std + max_fov_rad)

    return coords, polar_coords, plotting_coords


def get_logpolar_image_sampling_coords(fov, cmf_a, res, device='cpu', force_n_points=None, max_norm_rad=1):
    """Convenience wrapper for log polar image sampling.
    
    Sample coordinates with the cortical magnification function of the complex log mapping w=log(z+a), where z=x+iy.
    This is not isotropic, rather, it produces a square log polar image using an equal number of angular samples for all radii.
    
    Args:
        fov (float): Field of view diameter in degrees.
        cmf_a (float): A parameter from the CMF: M(r)=1/(r+a). Smaller a = stronger foveation.
        res (int): Number of sampling points.
        device (str, optional): Device to run the computation on. Defaults to 'cpu'.
        force_n_points (int, optional): If not None, the number of points is forced to be exactly this, useful for controlled comparisons with other sensors. Defaults to None.
        max_norm_rad (float, optional): Maximum normalized radius. Defaults to 1.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: Sampling cartesian coordinates in visual space, normalized to (-1,1).
            - torch.Tensor: Sampling polar coordinates in visual space.
            - torch.Tensor: Plotting coordinates in complex log space, useful for visualizing the sampling.
    """
    return get_isotropic_sampling_coords(fov, cmf_a, res, circular=True, device=device, constant_num_angles=True, force_n_points=force_n_points, max_norm_rad=max_norm_rad)


def _compute_isotropic_r_and_num_theta(fov, cmf_a, res, circular=True, device='cpu'):
    """Compute the radii and angles for isotropic logarithmic sampling.
    
    Args:
        fov (float): Field of view diameter in degrees.
        cmf_a (float): A parameter from the CMF: M(r)=1/(r+a). Smaller a = stronger foveation.
        res (int): Number of sampling radii.
        circular (bool, optional): If True, use circular sampling. Defaults to True.
        device (str, optional): Device to run computation on. Defaults to 'cpu'.
        
    Returns:
        tuple: A tuple containing:
            - torch.Tensor: Radii for each sampling ring.
            - torch.Tensor: Number of angles for each radius.
    """
    res = int(res)
    if fov is not None:
        r_max = (fov/2 if circular else np.sqrt(2)*fov/2)
    else:
        r_max = None

    max_norm_radius = (1 if circular else np.sqrt(2))

    # sample evenly in cortical radius (w=log(r + cmf_a)) and solve for visual radius (r = exp(w) - cmf_a)
    w_min = np.log(cmf_a)
    w_max = np.log(r_max + cmf_a)
    # add one extra point to cortical radius so that we can accurately compute the angle delta for the last visual radius
    w_delta = (w_max - w_min)/(res-1)

    w = torch.linspace(w_min, w_max+w_delta, steps=res+1, device=device) # even sampling in the cortical radius
    radius = torch.exp(w) - cmf_a # back-projection into visual radius

    # fulfill approximate isotropy: make the difference between neighboring angles equal to the difference in neighboring radii
    n_angles_init = 1
    n_angles = [n_angles_init]
    for ii in range(1,res):
        # average curr to prev and curr to next radius dists
        radius_diff = ((radius[ii] - radius[ii-1]) + (radius[ii+1] - radius[ii])) / 2 
        angles = torch.arange(0,2*torch.pi*radius[ii],radius_diff,device=device)/2*torch.pi*radius[ii]
        n_angles.append(len(angles))
    n_angles = torch.tensor(n_angles)  

    # remove extra radius
    radius = radius[:-1]

    # normalize to [0,1]
    radius = max_norm_radius * (radius / (fov/2))

    return radius, n_angles


def num_sampling_coords_isotropic(fov, cmf_a, res, circular=True, device='cpu'):
    """Quickly compute the number of sampling coordinates for isotropic sampling.

    Useful for optimizing the res (# of radii) to match a certain output n (# of points).
    
    Args:
        fov (float): Field of view diameter in degrees.
        cmf_a (float): A parameter from the CMF: M(r)=1/(r+a). Smaller a = stronger foveation.
        res (int): Number of sampling radii.
        circular (bool, optional): If True, use circular sampling. Defaults to True.
        device (str, optional): Device to run computation on. Defaults to 'cpu'.
        
    Returns:
        int: Total number of sampling coordinates.
    """
    radius, n_angles = _compute_isotropic_r_and_num_theta(fov, cmf_a, res, circular=circular, device=device)
    return n_angles.sum().item()


def find_desired_res(fov, cmf_a, n_points_desired, style, device='cpu', bounds=(1,1000), force_less_than=False, quiet=False):
    """Find the resolution that gives the desired number of sampling points using binary search.
    
    Args:
        fov (float): Field of view diameter in degrees.
        cmf_a (float): A parameter from the CMF: M(r)=1/(r+a). Smaller a = stronger foveation.
        n_points_desired (int): Desired number of sampling points.
        style (str): Which sampling style, e.g. 'isotropic'.
        device (str, optional): Device to run computation on. Defaults to 'cpu'.
        bounds (tuple, optional): Bounds for resolution search. Defaults to (1,1000).
        force_less_than (bool, optional): Whether to force less than target resolution. Defaults to False.
        quiet (bool, optional): Whether to suppress output. Defaults to False.
        
    Returns:
        tuple: A tuple containing:
            - int: Resolution that gives the desired number of points.
            - int: Actual number of points achieved.
    """
    # Try a range of integer values directly instead of using minimize_scalar
    best_res = None
    best_diff = float('inf')
    
    # Binary search through the range since the function is monotonic
    left, right = bounds
    while left <= right:
        mid = (left + right) // 2
        n = num_sampling_coords(fov, cmf_a, mid, style=style, device=device)
        diff = abs(n - n_points_desired)
        
        if diff < best_diff:
            best_diff = diff
            best_res = mid
            
        if n < n_points_desired:
            left = mid + 1
        else:
            right = mid - 1
    
    n = num_sampling_coords(fov, cmf_a, best_res, style=style, device=device)

    if force_less_than:
        while n > n_points_desired:
            best_res = best_res - 1
            n = num_sampling_coords(fov, cmf_a, best_res, style=style, device=device)
    else:
        while n < n_points_desired:
            # make sure we overshoot slightly so that we can remove angles rather than adding them
            best_res = best_res + 1
            n = num_sampling_coords(fov, cmf_a, best_res, style=style, device=device)

    if not quiet:
        print(f'found resolution {best_res} giving {n} points (desired: {n_points_desired})')
        
    return best_res, n


def get_sampling_coords(fov, cmf_a, res, device='cpu', style='isotropic', max_val=1):
    """Generate sampling coordinates based on the specified style.
    
    Args:
        fov (float): Field of view diameter in degrees.
        cmf_a (float): A parameter from the CMF: M(r)=1/(r+a). Smaller a = stronger foveation.
        res (int): Resolution parameter.
        device (str, optional): Device to run computation on. Defaults to 'cpu'.
        style (str): Sampling style ('isotropic', 'logpolar', 'isotropic_fixn', 'uniform', 'uniform_as_grid', 'logpolar_as_grid').
        max_val (float, optional): Maximum x/y value. Defaults to 1.
        
    Returns:
        tuple: A tuple containing:
            - torch.Tensor: Cartesian coordinates.
            - torch.Tensor: Polar coordinates.
            - torch.Tensor: Plotting coordinates.
    """
    assert style in ['isotropic', 'logpolar', 'isotropic_fixn', 'uniform', 'uniform_as_grid', 'logpolar_as_grid']
    if style == 'uniform' or style == 'uniform_as_grid':
        coords = torch.linspace(-max_val, max_val, res)
        coords = torch.stack(torch.meshgrid(coords, coords), dim=2).reshape(-1,2).to(device)
        polar_coords = torch.stack([torch.sqrt(coords[:,0]**2 + coords[:,1]**2), torch.arctan2(coords[:,1], coords[:,0])], dim=1)
        plotting_coords = coords.clone()
    elif 'isotropic' in style or style == 'logpolar' or style == 'logpolar_as_grid':
        if 'fixn' in style:
            force_n_points = res**2
        else:
            force_n_points = None
        if style == 'logpolar' or style == 'logpolar_as_grid':
            coords, polar_coords, plotting_coords = get_logpolar_image_sampling_coords(fov, cmf_a, res, device=device, force_n_points=None, max_norm_rad=max_val)
        else:
            coords, polar_coords, plotting_coords = get_isotropic_sampling_coords(fov, cmf_a, res, device=device, force_n_points=force_n_points, max_norm_rad=max_val)
        if 'fixn' in style:
            assert coords.shape[0] == res**2
    else:
        raise NotImplementedError('')

    return coords, polar_coords, plotting_coords


def xy_to_rowcol(coords, do_norm=True, format='01'):
    """Convert xy coordinates to row-column coordinates.
    
    Args:
        coords (torch.Tensor): Input coordinates in xy format.
        do_norm (bool, optional): Whether to normalize coordinates. Defaults to True.
        format (str): Coordinate format ('01' for [0,1] or '-11' for [-1,1]). Defaults to '01'.
        
    Returns:
        torch.Tensor: Coordinates in row-column format.
    """
    assert format in ['01', '-11']
    if format == '01':
        min = 0
        max = 1
    else:
        min = -1
        max = 1
    if do_norm:
        coords = normalize(coords, min=min, max=max)
    x, y = coords[:,0], coords[:,1]
    col = x
    if format == '01':
        row = 1-y
    else:
        row = -y
    return torch.stack((row, col), dim=1)


def xy_to_colrow(coords, do_norm=True, format='01'):
    """Convert xy coordinates to column-row coordinates.
    
    Args:
        coords (torch.Tensor): Input coordinates in xy format.
        do_norm (bool, optional): Whether to normalize coordinates. Defaults to True.
        format (str): Coordinate format ('01' for [0,1] or '-11' for [-1,1]). Defaults to '01'.
        
    Returns:
        torch.Tensor: Coordinates in row-column format.
    """
    rowcol = xy_to_rowcol(coords, do_norm, format)
    return torch.stack((rowcol[:,1], rowcol[:,0]), dim=1)


def num_sampling_coords(fov, cmf_a, res, style='isotropic', device='cpu'):
    """Calculate the number of sampling coordinates for a given style.
    
    Args:
        fov (float): Field of view diameter in degrees.
        cmf_a (float): A parameter from the CMF: M(r)=1/(r+a). Smaller a = stronger foveation.
        res (int): Resolution parameter.
        style (str): Sampling style. Defaults to 'isotropic'.
        device (str, optional): Device to run computation on. Defaults to 'cpu'.
        
    Returns:
        int: Number of sampling coordinates.
    """
    if style == 'isotropic':
        return num_sampling_coords_isotropic(fov, cmf_a, res, circular=True, device=device)
    elif style in ['logpolar', 'logpolar_as_grid', 'isotropic_fixn', 'uniform', 'uniform_as_grid']:
        return res**2
    else:
        raise ValueError(f'Style {style} not recognized')


def transform_sampling_grid(sampling_grid, fix_loc, fixation_size, image_size):
    """Transform sampling grid coordinates from fixation space to full image space.

    Args:
        sampling_grid (torch.Tensor): Sampling grid of shape (1, n_coords, 2) in [-1, 1] range. It is in (x,y) format.
        fix_loc (tuple or torch.Tensor): Fixation center in normalized image coordinates (h, w), e.g., (0.5, 0.5).
        fixation_size (tuple or torch.Tensor): Size of the fixation region in pixels (fix_h, fix_w).
        image_size (tuple): Full image size (H, W).

    Returns:
        torch.Tensor: Transformed sampling grid in the full image space.
    """
    if isinstance(fix_loc, tuple) or isinstance(fix_loc, np.ndarray):
        fix_loc = torch.tensor(fix_loc)
        if fix_loc.ndim == 1:
            fix_loc = fix_loc.unsqueeze(0)
    fix_loc = fix_loc.clone()
    if isinstance(fixation_size, tuple) or isinstance(fixation_size, np.ndarray):
        fixation_size = torch.tensor(fixation_size)
        if fixation_size.ndim == 1:
            fixation_size = fixation_size.unsqueeze(0)
    fixation_size = fixation_size.clone()
    if isinstance(sampling_grid, np.ndarray):
        sampling_grid = torch.tensor(sampling_grid)

    # Unpack inputs
    fix_center_h = fix_loc[:,0].reshape(-1,1,1).to(sampling_grid.device)
    fix_center_w = fix_loc[:,1].reshape(-1,1,1).to(sampling_grid.device)
    fix_h = fixation_size[:,0].reshape(-1,1,1).to(sampling_grid.device)
    fix_w = fixation_size[:,1].reshape(-1,1,1).to(sampling_grid.device)
    H, W = image_size
    batch_size = fix_center_h.shape[0]

    # Convert fixation center from normalized to pixel space
    fix_center_w *= W
    fix_center_h *= H

    # Scale grid from [-1, 1] to fixation size in pixels
    scaled_grid = sampling_grid.clone()
    if scaled_grid.shape[0] == 1:
        scaled_grid = scaled_grid.repeat(batch_size, 1, 1, 1)

    scaled_grid[:, :, :, 0] *= (fix_w / 2)  # Scale x-coordinates
    scaled_grid[:, :, :, 1] *= (fix_h / 2)  # Scale y-coordinates

    # Offset grid by fixation center
    scaled_grid[:, :, :, 0] += fix_center_w  # Shift x-coordinates
    scaled_grid[:, :, :, 1] += fix_center_h  # Shift y-coordinates

    scaled_grid[:, :, :, 0] = (2 * scaled_grid[:, :, :, 0] / W) - 1  # Normalize x-coordinates
    scaled_grid[:, :, :, 1] = (2 * scaled_grid[:, :, :, 1] / H) - 1  # Normalize y-coordinates

    return scaled_grid


def auto_match_num_coords(fov, cmf_a, cart_res, style, auto_match_cart_resources, device, force_less_than=True, quiet=False):
    """Automatically match the number of coordinates to cartesian resolution.
    
    Args:
        fov (float): Field of view diameter in degrees.
        cmf_a (float): A parameter from the CMF: M(r)=1/(r+a). Smaller a = stronger foveation.
        cart_res (int): Cartesian resolution.
        style (str): Sampling style.
        auto_match_cart_resources (int): Auto-matching parameter (-1: auto-match in_res, 0: no auto-matching, >0: auto-match everything).
        device (str): Device to run computation on.
        force_less_than (bool, optional): Whether to force less than target resolution. Defaults to True.
        
    Returns:
        tuple: A tuple containing:
            - int: Input resolution.
            - int: Cartesian resolution.
    """
    if 'fixn' in style:
        in_res = cart_res
    elif auto_match_cart_resources != 0 and 'fixn' not in style:
        # -1: auto-match in_res, 0: no auto-matching, >0: auto-match everything
        in_res, num_coords = find_desired_res(fov, cmf_a, cart_res**2, style, device=device, force_less_than=force_less_than, quiet=quiet)
    else:
        in_res = cart_res
    return in_res, cart_res
