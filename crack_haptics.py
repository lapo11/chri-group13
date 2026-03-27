# -*- coding: utf-8 -*-
"""
crack_haptics.py — Haptic force computation for crack-following task
=====================================================================
Proxy-based wall haptics + optional groove guidance along the crack centerline.
Reuses the proven architecture from the tube haptics but adapted for the
Mars crack-sealing scenario.
"""

import numpy as np


class CrackHaptics:
    """
    Computes haptic forces for following a crack trajectory.
    
    Features:
    - Virtual walls at crack boundaries (proxy-based, one-sided spring)
    - Groove guidance toward centerline (optional)
    - GP groove guidance toward learned trajectory (optional)
    """

    def __init__(self, crack,
                 groove_k=120.0,       # N/m — reduced for Haply 2
                 groove_f_max=0.6,     # N — reduced for Haply 2 motors
                 groove_damping=0.5,
                 wall_k=130.0,         # N/m — reduced for Haply 2
                 wall_damping=2.5,
                 groove_deadzone=0.0,
                 release_hysteresis=0.0,
                 local_search_window=80):
        self.crack = crack
        self.groove_k = groove_k
        self.groove_f_max = groove_f_max
        self.groove_damping = groove_damping
        self.wall_k = wall_k
        self.wall_damping = wall_damping
        self.groove_deadzone = groove_deadzone
        self.release_hysteresis = release_hysteresis
        self.local_search_window = local_search_window

        # Feature toggles
        self.groove_enabled = False
        self.walls_enabled = False

        # Proxy state
        self.proxy_pos = None
        self.proxy_idx = 0
        self.contact_wall = None
        self.prev_pos = None

        # GP groove
        self.gp_groove_enabled = False
        self.gp_groove_k = 150.0       # N/m — reduced for Haply 2
        self.gp_groove_f_max = 0.7     # N — reduced for Haply 2
        self.gp_groove_damping = 0.5
        self.gp_std_min = 0.001
        self.gp_std_max = 0.012

        self._gp_traj = None
        self._gp_traj_std = None
        self._gp_idx = 0
        self.n_demos = 0

        # Status outputs
        self.last_penetration = 0.0
        self.last_wall = None
        self.last_groove_force = 0.0
        self.last_wall_force = 0.0
        self.last_signed_d = 0.0
        self.last_proxy_pos = None

    def reset_proxy(self):
        self.proxy_pos = None
        self.proxy_idx = 0
        self.contact_wall = None
        self.prev_pos = None
        self.last_penetration = 0.0
        self.last_wall = None
        self.last_proxy_pos = None

    def _local_closest(self, pos):
        crack = self.crack
        w = self.local_search_window
        i_start = max(0, self.proxy_idx - w)
        i_end = min(crack.n_pts, self.proxy_idx + w + 1)
        local_cl = crack.centerline[i_start:i_end]
        dists = np.linalg.norm(local_cl - pos, axis=1)
        local_min = int(np.argmin(dists))
        idx = i_start + local_min
        proj = crack.centerline[idx]
        normal = crack.normals[idx]
        to_pos = pos - proj
        signed_d = float(np.dot(to_pos, normal))
        abs_d = abs(signed_d)
        return idx, proj, normal, signed_d, abs_d

    def set_gp_trajectory(self, traj, std, n_demos=1):
        self._gp_traj = np.asarray(traj, dtype=float)
        self._gp_traj_std = np.asarray(std, dtype=float)
        self._gp_idx = 0
        self.n_demos = n_demos

    def clear_gp_trajectory(self):
        self._gp_traj = None
        self._gp_traj_std = None
        self._gp_idx = 0

    def compute_force(self, pos_phys, dt=0.01):
        pos = np.asarray(pos_phys, dtype=float)
        crack = self.crack
        fe = np.zeros(2)

        if self.proxy_pos is None:
            idx_g, _, _, _, _ = crack.closest_centerline_point(pos)
            self.proxy_pos = pos.copy()
            self.proxy_idx = idx_g

        idx_g, proj_g, normal_g, signed_d_g, abs_d_g = crack.closest_centerline_point(pos)
        self.last_signed_d = signed_d_g

        # ── VIRTUAL WALLS ──
        if self.walls_enabled:
            if self.contact_wall is None:
                if abs_d_g <= crack.half_width:
                    self.proxy_pos = pos.copy()
                    self.proxy_idx = idx_g
                    self.last_penetration = 0.0
                    self.last_wall = None
                    self.last_wall_force = 0.0
                    self.last_proxy_pos = None
                else:
                    self.contact_wall = 'left' if signed_d_g > 0 else 'right'
                    self.proxy_idx = idx_g

            if self.contact_wall is not None:
                idx_l, proj_l, normal_l, signed_d_l, abs_d_l = self._local_closest(pos)
                self.proxy_idx = idx_l

                if self.contact_wall == 'left':
                    self.proxy_pos = proj_l + normal_l * crack.half_width
                else:
                    self.proxy_pos = proj_l - normal_l * crack.half_width

                self.last_proxy_pos = self.proxy_pos.copy()
                penetration = max(0.0, abs_d_l - crack.half_width)
                self.last_penetration = penetration
                self.last_wall = self.contact_wall

                if penetration > 0:
                    displacement = self.proxy_pos - pos
                    f_wall = self.wall_k * displacement
                    if self.prev_pos is not None and dt > 0:
                        vel = (pos - self.prev_pos) / dt
                        d_mag = np.linalg.norm(displacement)
                        if d_mag > 1e-8:
                            d_hat = displacement / d_mag
                            vel_deeper = -np.dot(vel, d_hat)
                            if vel_deeper > 0:
                                f_wall += self.wall_damping * vel_deeper * d_hat
                    fe += f_wall
                    self.last_wall_force = float(np.linalg.norm(f_wall))
                else:
                    self.last_wall_force = 0.0

                if abs_d_g < crack.half_width - self.release_hysteresis:
                    self.proxy_pos = pos.copy()
                    self.proxy_idx = idx_g
                    self.contact_wall = None
                    self.last_penetration = 0.0
                    self.last_wall = None
                    self.last_wall_force = 0.0
                    self.last_proxy_pos = None
                    fe = np.zeros(2)
        else:
            self.contact_wall = None
            self.proxy_pos = pos.copy()
            self.proxy_idx = idx_g
            self.last_penetration = 0.0
            self.last_wall = None
            self.last_wall_force = 0.0
            self.last_proxy_pos = None

        # ── GROOVE ──
        if self.groove_enabled and self.groove_k > 0:
            groove_d = abs_d_g - self.groove_deadzone

            if self.prev_pos is not None and dt > 0:
                vel = (pos - self.prev_pos) / dt
                vel_lateral = np.dot(vel, normal_g)
                fe -= self.groove_damping * vel_lateral * normal_g

            if groove_d > 0:
                groove_dir = -normal_g if signed_d_g > 0 else normal_g
                raw_force = self.groove_k * groove_d
                capped_force = min(raw_force, self.groove_f_max)
                f_groove = capped_force * groove_dir
                fe += f_groove
                self.last_groove_force = float(np.linalg.norm(f_groove))
            else:
                self.last_groove_force = 0.0

        # ── GP GROOVE ──
        if (self.gp_groove_enabled
                and self._gp_traj is not None
                and self.contact_wall is None):

            w = self.local_search_window
            i_start = max(0, self._gp_idx - w)
            i_end = min(len(self._gp_traj), self._gp_idx + w + 1)
            local = self._gp_traj[i_start:i_end]
            dists = np.linalg.norm(local - pos, axis=1)
            local_min = int(np.argmin(dists))
            self._gp_idx = i_start + local_min

            gp_pt = self._gp_traj[self._gp_idx]
            gp_std = float(np.mean(self._gp_traj_std[self._gp_idx]))

            std_factor = np.clip(
                (self.gp_std_max - gp_std) / (self.gp_std_max - self.gp_std_min), 0.0, 1.0)
            demo_factor = 1.0 - np.exp(-(self.n_demos - 1) / 3.0)
            alpha = std_factor * demo_factor

            k_eff = self.gp_groove_k * alpha

            if k_eff > 1.0:
                displacement = gp_pt - pos
                dist_to_gp = np.linalg.norm(displacement)

                if self.prev_pos is not None and dt > 0:
                    vel = (pos - self.prev_pos) / dt
                    if dist_to_gp > 1e-8:
                        d_hat = displacement / dist_to_gp
                        vel_lateral = np.dot(vel, d_hat)
                        fe -= self.gp_groove_damping * alpha * vel_lateral * d_hat

                raw = k_eff * dist_to_gp
                capped = min(raw, self.gp_groove_f_max)
                if dist_to_gp > 1e-8:
                    fe += capped * (displacement / dist_to_gp)

        self.prev_pos = pos.copy()

        # ── Global force saturation (Haply 2DIY safety) ──
        GLOBAL_F_MAX = 2.5  # N — absolute max total force (2DIY handles ~3-4N)
        f_mag = np.linalg.norm(fe)
        if f_mag > GLOBAL_F_MAX:
            fe = fe * (GLOBAL_F_MAX / f_mag)

        return fe
