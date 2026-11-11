"""Cortical sensor manifold: visual-field (x, y) -> 3-D cortical coordinates.

Ported from fovi 27b1bde (``fovi/sensing/manifold.py``), PLANAR field geometry only —
the geometry ``SamplingCoords`` always used there (``field_geometry='planar'`` is
fovi's default and canvit never set another). fovi's ``spherical`` / ``legacy``
geometries (numerical integration of the CMF) and its mesh-sampling helpers are not
ported. Arithmetic is copied verbatim: the KNN neighbourhoods built on these
coordinates are discrete, so a last-bit change can change a checkpoint's patches.
"""

import numpy as np
import torch


class CorticalSensorManifold:
    r"""
    3-D cortical sensor manifold based on Rovamu and Virsu (1984) (see also Motter (2009)),
    planar visual field. Relevant coordinate systems:

    * :math:`(x, y)` -> visual cartesian coordinates
    * :math:`(r, \theta)` -> visual polar coordinates
    * :math:`(\rho, z, \phi)` -> cortical cylindrical coordinates
    * :math:`(x_c, y_c, z)` -> cortical cartesian coordinates

    Magnification function :math:`M(r)=\frac{k}{r+a}`; :math:`k` only scales the
    coordinates (irrelevant for foveated sampling), :math:`a` (``cmf_a``) controls the
    foveation, smaller = stronger.
    """

    def __init__(self, cmf_a, fov, k=10):
        self.cmf_a = cmf_a
        self.fov = fov
        self.k = k
        if not all(np.isfinite(v) and v > 0 for v in (cmf_a, fov, k)):
            raise ValueError("cmf_a, fov, and k must be finite and positive")
        self.max_radius = np.inf

    def m(self, r):
        """Cortical magnification (mm/deg) at eccentricity ``r``."""
        return self.k/(self.cmf_a + r) # mm/deg

    def rho_3d(self, r):
        """Cortical radius in 3-D cylindrical coordinates (planar field: ``m(r) * r``)."""
        return self.m(r) * r

    def phi_3d(self, theta):
        r"""Cortical :math:`\phi` in 3-D cylindrical coordinates: :math:`\phi=\theta`."""
        return theta

    def z_3d(self, r):
        """Cortical z in 3-D cylindrical coordinates (closed form for the planar field)."""
        if np.any(np.asarray(r) < 0) or np.any(np.asarray(r) > self.max_radius):
            raise ValueError(f"Visual radius outside manifold domain [0, {self.max_radius}] degrees")
        # Analytic integral avoids absolute mesh spacing breaking scale invariance.
        t = np.sqrt(np.maximum((np.asarray(r) / self.cmf_a + 1) ** 2 - 1, 0))
        return self.k * (np.arcsinh(t) - t / np.sqrt(1 + t * t))

    def map_3d(self, x, y):
        r"""Visual cartesian :math:`(x, y)` -> cortical cylindrical :math:`(\rho, z, \phi)`."""
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        rho = self.rho_3d(r)
        phi = self.phi_3d(theta)
        z = self.z_3d(r)
        return rho, z, phi

    def map_to_xyz(self, rho_z_phi):
        r"""Cortical cylindrical -> cortical cartesian :math:`(x_c, y_c, z)`."""
        rho, z, phi = rho_z_phi
        x_c = rho*np.cos(phi)
        y_c = rho*np.sin(phi)
        return x_c, y_c, z

    def vis_cartesian_to_cort_cartesian(self, x_y):
        """Visual cartesian ``(n, 2)`` -> cortical cartesian ``(n, 3)``."""
        return np.array([(self.map_to_xyz(self.map_3d(x,y))) for x, y in x_y])


def vis_cartesian_to_cortical_cartesian_coords(cartesian_coords, cmf_a, fov, as_tensor=False, device='cpu', k=10):
    """Map visual cartesian coordinates (``(n, 2)``, normalized to the field, (x, y)) to
    3-D cortical cartesian coordinates ``(n, 3)`` on the sensor manifold."""
    cartesian_fov_coords = cartesian_coords*(fov/2)
    model = CorticalSensorManifold(cmf_a, fov, k=k)
    grid_pts_3d = model.vis_cartesian_to_cort_cartesian(cartesian_fov_coords)

    if as_tensor:
        return torch.tensor(grid_pts_3d, device=device)
    else:
        return grid_pts_3d
