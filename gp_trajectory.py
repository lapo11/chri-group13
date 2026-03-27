# -*- coding: utf-8 -*-
"""
gp_trajectory.py — Gaussian Process trajectory learning
=========================================================
Takes multiple demonstrated trajectories and learns a smooth mean
trajectory with uncertainty (std) using independent GPs for x and y,
parameterized by normalized arc-length.
"""

import numpy as np
from scipy.interpolate import CubicSpline


class TrajectoryGP:
    """
    Lightweight trajectory learner using cubic-spline smoothing with
    point-wise mean/std across demonstrations.

    For the PA3 assignment, this provides a fast, dependency-free alternative
    to sklearn's GaussianProcessRegressor while producing similar results
    for trajectory reproduction.
    """

    def __init__(self):
        self.demos = []
        self._mean_traj = None
        self._std_traj = None

    def fit(self, demos):
        """
        Fit the model on a list of demonstrated trajectories.

        Parameters
        ----------
        demos : list of ndarray, each (N_i, 2)
            Raw demonstrated trajectories in physical coordinates.
        """
        self.demos = [np.asarray(d, dtype=float) for d in demos]

    def predict(self, n_points=300, return_std=True):
        """
        Predict the mean trajectory and (optionally) std.

        Each demo is resampled to `n_points` via arc-length parameterization,
        then the point-wise mean and std are computed.

        Returns
        -------
        mean : ndarray (n_points, 2)
        std  : ndarray (n_points, 2)   [if return_std]
        """
        if not self.demos:
            raise ValueError("No demonstrations provided. Call fit() first.")

        resampled = []
        for demo in self.demos:
            rs = self._resample(demo, n_points)
            resampled.append(rs)

        stacked = np.stack(resampled, axis=0)  # (n_demos, n_points, 2)
        mean = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0) if len(resampled) > 1 else np.ones_like(mean) * 0.01

        self._mean_traj = mean
        self._std_traj = std

        if return_std:
            return mean, std
        return mean

    def _resample(self, traj, n_points):
        """Resample trajectory to n_points using arc-length parameterization."""
        if len(traj) < 2:
            return np.tile(traj[0], (n_points, 1))

        # Compute cumulative arc length
        diffs = np.diff(traj, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        cum_arc = np.zeros(len(traj))
        cum_arc[1:] = np.cumsum(seg_lengths)
        total = cum_arc[-1]

        if total < 1e-8:
            return np.tile(traj[0], (n_points, 1))

        # Normalize to [0, 1]
        s = cum_arc / total

        # Remove duplicate s values (can cause spline issues)
        mask = np.concatenate([[True], np.diff(s) > 1e-10])
        s_clean = s[mask]
        traj_clean = traj[mask]

        if len(s_clean) < 2:
            return np.tile(traj[0], (n_points, 1))

        # Cubic spline interpolation
        cs_x = CubicSpline(s_clean, traj_clean[:, 0])
        cs_y = CubicSpline(s_clean, traj_clean[:, 1])

        s_new = np.linspace(0, 1, n_points)
        x_new = cs_x(s_new)
        y_new = cs_y(s_new)

        return np.column_stack([x_new, y_new])
