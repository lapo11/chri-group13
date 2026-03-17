# -*- coding: utf-8 -*-
"""
PA3 — Haptic Force Computation (Proxy-based, one-sided spring)
---------------------------------------------------------------
Key design:
  - LOCAL search around proxy_idx prevents teleporting on S-curve inflections
  - ONE-SIDED spring: force only when abs_d > half_width (no pull-back = no bounce)
  - contact_wall stays latched for proxy tracking, but force is gated on penetration
  - Release happens when clearly inside (release_hysteresis=0 → exactly at boundary)
"""

import numpy as np


class TubeHaptics:

    def __init__(self, tube,
                 groove_k=230.0,
                 groove_f_max=1.0,   # N — max groove force 
                 groove_damping=0.8,   # N·s/m — damps lateral oscillation
                 wall_k=250.0,
                 wall_damping=4.0,
                 groove_deadzone=0.0,
                 release_hysteresis=0.0,
                 local_search_window=80,
                 exit_damping=5.0,       # N·s/m — absorbs rebound velocity
                 exit_band=0.0          # m — 4mm viscous layer inside wall

                 ):
        self.tube = tube
        self.groove_k = groove_k
        self.groove_f_max = groove_f_max
        self.groove_damping = groove_damping
        self.wall_k = wall_k
        self.wall_damping = wall_damping
        self.groove_deadzone = groove_deadzone
        self.release_hysteresis = release_hysteresis
        self.local_search_window = local_search_window
        self.exit_damping = exit_damping
        self.exit_band = exit_band

        
        # ── Independent feature toggles (set from PA3 via H / W keys) ──
        self.groove_enabled = True
        self.walls_enabled = True

        # ── Proxy state ──
        self.proxy_pos = None
        self.proxy_idx = 0
        self.contact_wall = None
        self.prev_pos = None


        # ── GP groove ──
        self.gp_groove_enabled = False      # toggled with K key
        self.gp_groove_k       = 300.0      # N/m  — max stiffness when fully confident
        self.gp_groove_f_max   = 1.5        # N    — saturating cap
        self.gp_groove_damping = 0.8        # N·s/m — lateral damping
        self.gp_std_min        = 0.001      # m — std below this = full confidence
        self.gp_std_max        = 0.012      # m — std above this = no guidance

        self.fading_groove_enabled = False   # J key — centerline groove fading with confidence


        # GP trajectory data (set after training)
        self._gp_traj     = None            # ndarray (N, 2)
        self._gp_traj_std = None            # ndarray (N, 2)
        self._gp_idx      = 0               # local search anchor
        self.n_demos = 0    # updated when set_gp_trajectory is called


        # ── Status outputs ──
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
        tube = self.tube
        w = self.local_search_window
        i_start = max(0, self.proxy_idx - w)
        i_end = min(tube.n_pts, self.proxy_idx + w + 1)
        local_cl = tube.centerline[i_start:i_end]
        dists = np.linalg.norm(local_cl - pos, axis=1)
        local_min = int(np.argmin(dists))
        idx = i_start + local_min
        proj = tube.centerline[idx]
        normal = tube.normals[idx]
        to_pos = pos - proj
        signed_d = float(np.dot(to_pos, normal))
        abs_d = abs(signed_d)
        return idx, proj, normal, signed_d, abs_d
    
    def set_gp_trajectory(self, traj, std, n_demos=1):
        """Call this after GP training to enable GP groove."""
        self._gp_traj     = np.asarray(traj, dtype=float)
        self._gp_traj_std = np.asarray(std,  dtype=float)
        self._gp_idx      = 0
        self.n_demos = n_demos

    def clear_gp_trajectory(self):
        self._gp_traj     = None
        self._gp_traj_std = None
        self._gp_idx      = 0


    def compute_force(self, pos_phys, dt=0.01):
        pos = np.asarray(pos_phys, dtype=float)
        tube = self.tube
        fe = np.zeros(2)

        if self.proxy_pos is None:
            idx_g, _, _, _, _ = tube.closest_centerline_point(pos)
            self.proxy_pos = pos.copy()
            self.proxy_idx = idx_g

        idx_g, proj_g, normal_g, signed_d_g, abs_d_g = tube.closest_centerline_point(pos)
        self.last_signed_d = signed_d_g

        # ── VIRTUAL WALLS ────────────────────────────────────────────────
        if self.walls_enabled:
            if self.contact_wall is None:
                if abs_d_g <= tube.half_width:
                    # Free: proxy tracks ee
                    self.proxy_pos = pos.copy()
                    self.proxy_idx = idx_g
                    self.last_penetration = 0.0
                    self.last_wall = None
                    self.last_wall_force = 0.0
                    self.last_proxy_pos = None
                else:
                    # First contact: latch wall
                    self.contact_wall = 'left' if signed_d_g > 0 else 'right'
                    self.proxy_idx = idx_g

            if self.contact_wall is not None:
                idx_l, proj_l, normal_l, signed_d_l, abs_d_l = self._local_closest(pos)
                self.proxy_idx = idx_l

                if self.contact_wall == 'left':
                    self.proxy_pos = proj_l + normal_l * tube.half_width
                else:
                    self.proxy_pos = proj_l - normal_l * tube.half_width

                self.last_proxy_pos = self.proxy_pos.copy()
                penetration = max(0.0, abs_d_l - tube.half_width)
                self.last_penetration = penetration
                self.last_wall = self.contact_wall

                # One-sided spring: only push when outside
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

                # Release when clearly inside
                if abs_d_g < tube.half_width - self.release_hysteresis:
                    self.proxy_pos = pos.copy()
                    self.proxy_idx = idx_g
                    self.contact_wall = None
                    self.last_penetration = 0.0
                    self.last_wall = None
                    self.last_wall_force = 0.0
                    self.last_proxy_pos = None
                    fe = np.zeros(2)

        else:
            # Walls disabled: release any contact, proxy follows ee
            self.contact_wall = None
            self.proxy_pos = pos.copy()
            self.proxy_idx = idx_g
            self.last_penetration = 0.0
            self.last_wall = None
            self.last_wall_force = 0.0
            self.last_proxy_pos = None

        # ── EXIT DAMPING: viscous band just inside wall ──────────────────
        depth_inside = tube.half_width - abs_d_g
        if 0 <= depth_inside < self.exit_band and self.prev_pos is not None and dt > 0:
            vel = (pos - self.prev_pos) / dt
            vel_normal = np.dot(vel, normal_g)
            fe -= self.exit_damping * vel_normal * normal_g

        # ── GROOVE ───────────────────────────────────────────────────────
        if self.groove_enabled and self.groove_k > 0 : #and self.contact_wall is None:
            groove_d = abs_d_g - self.groove_deadzone

            # Groove damping — always active laterally (prevents oscillation)
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


        # ── GP GROOVE (confidence-scaled, toward learned trajectory) ────────
        if (self.gp_groove_enabled
                and self._gp_traj is not None
                and self.contact_wall is None):

            # Local search around current index — same anti-teleport trick as wall proxy
            w = self.local_search_window
            i_start = max(0, self._gp_idx - w)
            i_end   = min(len(self._gp_traj), self._gp_idx + w + 1)
            local   = self._gp_traj[i_start:i_end]
            dists   = np.linalg.norm(local - pos, axis=1)
            local_min = int(np.argmin(dists))
            self._gp_idx = i_start + local_min

            gp_pt  = self._gp_traj[self._gp_idx]
            gp_std = float(np.mean(self._gp_traj_std[self._gp_idx]))


            # New confidence: with std, scaled by number of demos
            std_factor  = np.clip(
                (self.gp_std_max - gp_std) / (self.gp_std_max - self.gp_std_min), 0.0, 1.0)
            demo_factor = 1.0 - np.exp(-(self.n_demos - 1) / 3.0)  # 0 at N=1, grows with N
            alpha = std_factor * demo_factor



            k_eff = self.gp_groove_k * alpha

            if k_eff > 1.0:
                displacement = gp_pt - pos       # vector: ee → GP point
                dist_to_gp   = np.linalg.norm(displacement)

                # Groove damping — damp lateral velocity toward/away from GP line
                if self.prev_pos is not None and dt > 0:
                    vel = (pos - self.prev_pos) / dt
                    if dist_to_gp > 1e-8:
                        d_hat = displacement / dist_to_gp
                        vel_lateral = np.dot(vel, d_hat)
                        fe -= self.gp_groove_damping * alpha * vel_lateral * d_hat

                # Saturating spring toward GP point
                raw   = k_eff * dist_to_gp
                capped = min(raw, self.gp_groove_f_max)
                if dist_to_gp > 1e-8:
                    fe += capped * (displacement / dist_to_gp)

        # ── FADING GROOVE (centerline groove, stiffness DECREASES with confidence) ──
        if (self.fading_groove_enabled and self.groove_k > 0):
            # Reuse centerline distance already computed above (signed_d_g, normal_g)
            groove_d = abs_d_g - self.groove_deadzone

            # Compute confidence the same way as GP groove
            if self._gp_traj is not None and self._gp_traj_std is not None:
                w = self.local_search_window
                i_start = max(0, self._gp_idx - w)
                i_end   = min(len(self._gp_traj), self._gp_idx + w + 1)
                local   = self._gp_traj[i_start:i_end]
                dists   = np.linalg.norm(local - pos, axis=1)
                self._gp_idx = i_start + int(np.argmin(dists))
                gp_std = float(np.mean(self._gp_traj_std[self._gp_idx]))
                std_factor  = np.clip(
                    (self.gp_std_max - gp_std) / (self.gp_std_max - self.gp_std_min),
                    0.0, 1.0)
                demo_factor = 1.0 - np.exp(-(self.n_demos - 1) / 3.0)
                alpha = std_factor * demo_factor
            else:
                alpha = 0.0  # no GP yet → full stiffness

            k_eff = self.groove_k * (1.0 - alpha)   # fades OUT as confidence grows

            # Lateral damping (always active, also fades)
            if self.prev_pos is not None and dt > 0:
                vel = (pos - self.prev_pos) / dt
                vel_lateral = np.dot(vel, normal_g)
                fe -= self.groove_damping * (1.0 - alpha) * vel_lateral * normal_g

            if groove_d > 0 and k_eff > 1.0:
                groove_dir = -normal_g if signed_d_g > 0 else normal_g
                raw_force    = k_eff * groove_d
                capped_force = min(raw_force, self.groove_f_max)
                fe += capped_force * groove_dir



        self.prev_pos = pos.copy()
        return fe