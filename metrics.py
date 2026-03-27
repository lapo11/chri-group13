# -*- coding: utf-8 -*-
"""
metrics.py — Evaluation metrics for trajectory comparison
==========================================================
"""

import numpy as np
from scipy.spatial.distance import directed_hausdorff


def mean_nearest_distance(traj_a, traj_b):
    """
    Mean Nearest Distance (MND) from traj_a to traj_b.
    For each point in traj_a, find the nearest point in traj_b,
    then average those distances.
    """
    a = np.asarray(traj_a, dtype=float)
    b = np.asarray(traj_b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float('inf')

    # Vectorized: (N, 1, 2) - (1, M, 2) → (N, M)
    dists = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    nearest = np.min(dists, axis=1)
    return float(np.mean(nearest))


def hausdorff_distance(traj_a, traj_b):
    """Hausdorff distance between two trajectories."""
    a = np.asarray(traj_a, dtype=float)
    b = np.asarray(traj_b, dtype=float)
    d1 = directed_hausdorff(a, b)[0]
    d2 = directed_hausdorff(b, a)[0]
    return max(d1, d2)


def trajectory_smoothness(traj, dt=0.01):
    """
    Compute smoothness as mean absolute jerk (third derivative).
    Lower = smoother.
    """
    t = np.asarray(traj, dtype=float)
    if len(t) < 4:
        return 0.0
    vel = np.diff(t, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    return float(np.mean(np.linalg.norm(jerk, axis=1)))


def trajectory_length(traj):
    """Total path length of trajectory."""
    t = np.asarray(traj, dtype=float)
    if len(t) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(t, axis=0), axis=1)))


def compute_all_metrics(demos_concat, gp_traj, centerline):
    """
    Compute a dictionary of all metrics.
    
    Parameters
    ----------
    demos_concat : ndarray — all demo points concatenated
    gp_traj      : ndarray — GP mean trajectory
    centerline   : ndarray — crack centerline
    
    Returns
    -------
    dict with keys: gp_mnd, gp_hausdorff, gp_smoothness, gp_length, 
                    demo_mnd, demo_smoothness
    """
    metrics = {}

    if gp_traj is not None and len(gp_traj) > 0:
        metrics['gp_mnd'] = mean_nearest_distance(gp_traj, centerline)
        metrics['gp_hausdorff'] = hausdorff_distance(gp_traj, centerline)
        metrics['gp_smoothness'] = trajectory_smoothness(gp_traj)
        metrics['gp_length'] = trajectory_length(gp_traj)
    else:
        metrics['gp_mnd'] = 0.0
        metrics['gp_hausdorff'] = 0.0
        metrics['gp_smoothness'] = 0.0
        metrics['gp_length'] = 0.0

    if demos_concat is not None and len(demos_concat) > 0:
        metrics['demo_mnd'] = mean_nearest_distance(demos_concat, centerline)
        metrics['demo_smoothness'] = trajectory_smoothness(demos_concat)
    else:
        metrics['demo_mnd'] = 0.0
        metrics['demo_smoothness'] = 0.0

    return metrics
