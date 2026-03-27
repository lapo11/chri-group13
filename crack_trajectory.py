# -*- coding: utf-8 -*-
"""
crack_trajectory.py — Procedural crack generation for Mars Habitat scenario
============================================================================
Generates realistic branching crack patterns in the Haply workspace.
Each crack is a smooth spline through random control points, with computed
normals, wall boundaries, and progress tracking.

The crack is defined in *physical* coordinates (meters) matching the Haply
pantograph workspace (~0.0 to 0.08 in x, ~0.02 to 0.12 in y).
"""

import numpy as np
from scipy.interpolate import CubicSpline


# ── Predefined crack shapes ─────────────────────────────────────────────────

def _crack_diagonal(n_pts=400):
    """Diagonal crack from top-left to bottom-right of workspace."""
    ctrl = np.array([
        [0.055, 0.030],
        [0.048, 0.042],
        [0.040, 0.048],
        [0.032, 0.055],
        [0.038, 0.065],
        [0.030, 0.072],
        [0.025, 0.082],
        [0.020, 0.090],
    ])
    return ctrl


def _crack_s_curve(n_pts=400):
    """S-shaped crack across the workspace."""
    ctrl = np.array([
        [0.055, 0.028],
        [0.050, 0.038],
        [0.040, 0.045],
        [0.030, 0.050],
        [0.025, 0.058],
        [0.030, 0.068],
        [0.040, 0.075],
        [0.045, 0.082],
        [0.035, 0.090],
        [0.025, 0.095],
    ])
    return ctrl


def _crack_zigzag(n_pts=400):
    """Zigzag crack pattern."""
    ctrl = np.array([
        [0.050, 0.030],
        [0.038, 0.040],
        [0.050, 0.050],
        [0.030, 0.060],
        [0.048, 0.070],
        [0.032, 0.078],
        [0.042, 0.088],
        [0.028, 0.095],
    ])
    return ctrl


def _crack_vertical(n_pts=400):
    """Nearly vertical crack with slight wobble."""
    ctrl = np.array([
        [0.040, 0.028],
        [0.042, 0.040],
        [0.038, 0.050],
        [0.041, 0.060],
        [0.037, 0.070],
        [0.040, 0.080],
        [0.039, 0.090],
        [0.041, 0.098],
    ])
    return ctrl


CRACK_LIBRARY = {
    "diagonal":  _crack_diagonal,
    "s_curve":   _crack_s_curve,
    "zigzag":    _crack_zigzag,
    "vertical":  _crack_vertical,
}

CRACK_NAMES = list(CRACK_LIBRARY.keys())


class CrackTrajectory:
    """
    A smooth crack trajectory with tube-like walls.
    
    Attributes
    ----------
    centerline : ndarray (N, 2)  — smooth crack path in physical coords
    normals    : ndarray (N, 2)  — unit normals at each centerline point
    wall_left  : ndarray (N, 2)  — left wall = centerline + normal * half_width
    wall_right : ndarray (N, 2)  — right wall = centerline - normal * half_width
    half_width : float           — half the crack width in meters
    start, end : ndarray (2,)    — first and last centerline points
    n_pts      : int             — number of discretized points
    """

    def __init__(self, name="s_curve", half_width=0.004, n_pts=400):
        """
        Parameters
        ----------
        name : str
            Key into CRACK_LIBRARY
        half_width : float
            Half-width of the crack tube in meters (default 4mm)
        n_pts : int
            Number of points to discretize the centerline
        """
        self.name = name
        self.half_width = half_width
        self.n_pts = n_pts

        ctrl = CRACK_LIBRARY[name]()
        self.centerline, self.normals = self._interpolate(ctrl, n_pts)
        self.wall_left  = self.centerline + self.normals * half_width
        self.wall_right = self.centerline - self.normals * half_width
        self.start = self.centerline[0].copy()
        self.end   = self.centerline[-1].copy()

        # Precompute cumulative arc-length for progress tracking
        diffs = np.diff(self.centerline, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        self._cum_arc = np.zeros(n_pts)
        self._cum_arc[1:] = np.cumsum(seg_lengths)
        self._total_length = self._cum_arc[-1]

    def _interpolate(self, ctrl, n_pts):
        """Fit a cubic spline through control points and sample uniformly."""
        t_ctrl = np.linspace(0, 1, len(ctrl))
        cs_x = CubicSpline(t_ctrl, ctrl[:, 0])
        cs_y = CubicSpline(t_ctrl, ctrl[:, 1])

        t = np.linspace(0, 1, n_pts)
        x = cs_x(t)
        y = cs_y(t)
        centerline = np.column_stack([x, y])

        # Compute tangents and normals
        dx = cs_x(t, 1)
        dy = cs_y(t, 1)
        tangents = np.column_stack([dx, dy])
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        tangents = tangents / norms

        # Normal = 90° rotation of tangent (pointing "left")
        normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
        return centerline, normals

    def closest_centerline_point(self, pos):
        """
        Find the closest point on the centerline to `pos`.
        
        Returns
        -------
        idx      : int       — index into centerline
        proj     : ndarray   — closest centerline point
        normal   : ndarray   — unit normal at that point
        signed_d : float     — signed distance (positive = left side)
        abs_d    : float     — absolute distance
        """
        pos = np.asarray(pos, dtype=float)
        dists = np.linalg.norm(self.centerline - pos, axis=1)
        idx = int(np.argmin(dists))
        proj = self.centerline[idx]
        normal = self.normals[idx]
        to_pos = pos - proj
        signed_d = float(np.dot(to_pos, normal))
        abs_d = abs(signed_d)
        return idx, proj, normal, signed_d, abs_d

    def progress(self, pos):
        """Return 0..1 progress along the crack based on closest centerline point."""
        _, _, _, _, _ = self.closest_centerline_point(pos)
        dists = np.linalg.norm(self.centerline - np.asarray(pos), axis=1)
        idx = int(np.argmin(dists))
        return self._cum_arc[idx] / max(self._total_length, 1e-8)

    def is_inside(self, pos):
        """Return True if pos is inside the crack tube."""
        _, _, _, _, abs_d = self.closest_centerline_point(pos)
        return abs_d <= self.half_width


def get_crack(name="s_curve", half_width=0.004, n_pts=400):
    """Factory function to create a CrackTrajectory."""
    return CrackTrajectory(name=name, half_width=half_width, n_pts=n_pts)
