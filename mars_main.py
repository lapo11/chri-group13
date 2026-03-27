# -*- coding: utf-8 -*-
"""
mars_main.py — Mars Habitat Crack-Sealing: Learning from Demonstration
========================================================================
PA3 Main Application

Navigate a crack on the Mars habitat wall from START to END, applying
sealant along the path. The task is to demonstrate the trajectory so a
robot can learn to reproduce it autonomously.

Three experimental conditions:
  1. Mouse            — standard mouse input, no haptic feedback
  2. Haply (no haptic) — Haply device without force feedback
  3. Haply (haptic)    — Haply device WITH haptic guidance (groove + walls)

Keys:
  SPACE   Start/stop recording a demonstration
  ENTER   Confirm demo / add more / go back to IDLE after GP
  D       Delete last recorded demo
  G       Train GP on all demonstrations
  P       Replay GP trajectory
  A       Auto-play GP (PD controller drives the device)
  N       NASA-TLX questionnaire
  H       Toggle groove haptic guidance
  W       Toggle virtual wall haptics
  T       Cycle crack shape (only in IDLE with no demos)
  C       Clear everything
  R       Toggle linkage display
  F       Toggle debug info
  Q       Quit and save results
"""

import sys
import os
import time
import json
import numpy as np
import pygame

from Physics import Physics
from mars_renderer import MarsRenderer
from crack_trajectory import CrackTrajectory, get_crack, CRACK_NAMES
from crack_haptics import CrackHaptics
from gp_trajectory import TrajectoryGP
from metrics import compute_all_metrics, mean_nearest_distance
from nasa_tlx import run_nasa_tlx


# ── State machine ──
IDLE, RECORDING, REVIEW, TRAINING, PLAYBACK, DONE, AUTO_PLAY = range(7)
STATE_NAMES = ["IDLE", "RECORDING", "REVIEW", "TRAINING", "PLAYBACK", "DONE", "AUTO_PLAY"]

# ── Demo colors ──
DEMO_COLORS = [
    (220, 100, 100), (100, 100, 220), (220, 180, 60),
    (180, 60, 220), (60, 200, 200), (200, 130, 80),
    (120, 220, 80), (220, 80, 180),
]

# ── Seal radius — how close the cursor must be to "seal" a crack point ──
SEAL_RADIUS = 0.005  # 5mm in physical coords


class MarsPA3:
    """Main application for the Mars habitat crack-sealing experiment."""

    def __init__(self):
        # ── Hardware ──
        self.physics = Physics(hardware_version=3)
        self.device_connected = self.physics.is_device_connected()

        # ── Renderer ──
        self.renderer = MarsRenderer(self.device_connected, window_size=(760, 720))

        # ── Crack ──
        self.crack_idx = 0
        self.crack_name = CRACK_NAMES[self.crack_idx]
        self.crack = get_crack(self.crack_name)
        self.haptics = CrackHaptics(self.crack)

        # ── State ──
        self.state = IDLE
        self.all_demos = []
        self.all_demo_times = []
        self.per_demo_metrics = []
        self.current_demo = []
        self.start_time = 0.0
        self.last_demo_time = 0.0
        self.wall_hits = 0
        self.recording_start = 0.0

        # ── Seal tracking ──
        self.sealed_indices = set()
        self.current_seal_indices = set()  # sealed during current demo

        # ── GP ──
        self.gp = TrajectoryGP()
        self.gp_traj_phys = None
        self.gp_traj_std = None
        self.playback_idx = 0
        self.trial_metrics = None
        self.tlx_result = None
        self.all_results = []

        # ── Auto-play PD controller (tuned for Haply 2DIY) ──
        self.auto_ref_idx = 0.0
        self.pd_kp = 100.0       # N/m — needs to overcome device friction
        self.pd_kd = 6.0         # N·s/m — damping (prevents overshoot)
        self.pd_f_max = 2.0      # N — force saturation (2DIY can handle ~2-3N)
        self.pd_speed_base = 0.8 # base reference advance per frame
        self.pd_leash = 0.015    # m — 15mm leash (generous)
        self.pd_lookahead = 12   # indices ahead for carrot pull
        self.prev_ee_phys = None
        self.vel_filtered = np.zeros(2)  # low-pass filtered velocity
        self.vel_alpha = 0.3     # filter coefficient

        # ── Condition label (set by user or experiment protocol) ──
        self.condition = "unknown"
        if self.device_connected:
            self.condition = "haply_haptic"  # default for connected device
        else:
            self.condition = "mouse"

    # ── Key legend drawing ──

    def _draw_key_legend(self, surface):
        h = self.haptics
        groove_str = "ON " if h.groove_enabled else "OFF"
        walls_str = "ON " if h.walls_enabled else "OFF"

        lines = [
            "── Controls ──────────",
            "SPACE  start/stop rec",
            "ENTER  confirm / next",
            "D      delete last demo",
            "G      train GP",
            "P      replay GP",
            "A      auto-play GP",
            f"W      walls   [{walls_str}]",
            f"H      groove  [{groove_str}]",
            "T      change crack",
            "N      NASA-TLX",
            "C      clear all",
            "Q      quit + save",
        ]

        font = pygame.font.SysFont("Consolas", 12)
        x, y = 8, 8
        pad = 4
        line_h = font.get_linesize()
        box_w = 200
        box_h = len(lines) * line_h + pad * 2

        bg = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 160))
        surface.blit(bg, (x - pad, y - pad))

        for i, line in enumerate(lines):
            color = (255, 200, 80) if i == 0 else (200, 200, 200)
            surf = font.render(line, True, color)
            surface.blit(surf, (x, y + i * line_h))

    def _draw_instructions(self, surface):
        """Draw state-specific instructions at bottom of VR panel."""
        font = pygame.font.SysFont("Consolas", 14)
        n = len(self.all_demos)
        y_base = self.renderer.window_size[1] - 130

        if self.state == IDLE:
            lines = [
                f"Demos: {n}  |  Crack: {self.crack_name}",
                "SPACE = start recording a demonstration",
            ]
            if n >= 1:
                lines.append("G = train GP on demos")
        elif self.state == RECORDING:
            elapsed = time.time() - self.start_time
            lines = [
                f"RECORDING demo #{n + 1}...  ({elapsed:.1f}s)",
                "SPACE = stop recording",
                "Follow the crack from START to END!",
            ]
        elif self.state == REVIEW:
            lines = [
                f"Demo #{n} saved! ({self.last_demo_time:.1f}s)",
                f"Accuracy (MND): {self.per_demo_metrics[-1] * 1000:.1f}mm",
                "ENTER = confirm & record next",
                "D = delete this demo",
                f"G = train GP on {n} demo{'s' if n > 1 else ''}",
            ]
        elif self.state == DONE:
            m = self.trial_metrics or {}
            lines = [
                f"GP trained on {m.get('n_demos', n)} demos",
                f"GP accuracy: {m.get('gp_mnd', 0) * 1000:.2f}mm",
                "A = auto-play  |  P = replay",
                "ENTER = add more demos  |  N = NASA-TLX",
            ]
        elif self.state == PLAYBACK:
            lines = ["Playing back GP trajectory..."]
        elif self.state == AUTO_PLAY:
            lines = ["Autonomous GP reproduction in progress..."]
        else:
            lines = []

        for i, line in enumerate(lines):
            surf = font.render(line, True, (220, 220, 220))
            surface.blit(surf, (10, y_base + i * 20))

    # ── Seal logic ──

    def _update_seal(self, pos_phys):
        """Mark crack points near the cursor as sealed."""
        dists = np.linalg.norm(self.crack.centerline - np.asarray(pos_phys), axis=1)
        near = np.where(dists < SEAL_RADIUS)[0]
        for idx in near:
            self.sealed_indices.add(int(idx))
            self.current_seal_indices.add(int(idx))

    def _get_seal_percentage(self):
        return len(self.sealed_indices) / max(self.crack.n_pts, 1)

    def _get_unsealed_mask(self):
        mask = np.ones(self.crack.n_pts, dtype=bool)
        for idx in self.sealed_indices:
            if 0 <= idx < self.crack.n_pts:
                mask[idx] = False
        return mask

    # ── PD Controller (safe for Haply 2) ──

    def _compute_pd_force(self, pos_phys, dt):
        """
        Carrot-on-a-leash PD controller for autonomous GP reproduction.
        
        Key safety features for the Haply 2:
        - Force is SATURATED to pd_f_max (prevents motor jerk/instability)
        - Reference only advances when the device is close enough (leash)
        - Velocity is low-pass filtered (noisy differentiation → oscillation)
        - Lookahead carrot pulls the device forward smoothly
        """
        traj = self.gp_traj_phys
        n = len(traj)
        pos = np.asarray(pos_phys, dtype=float)

        # ── Find closest point on trajectory (snap reference to reality) ──
        # This prevents the reference from running away if the device lags
        search_lo = max(0, int(self.auto_ref_idx) - 30)
        search_hi = min(n, int(self.auto_ref_idx) + 60)
        local_traj = traj[search_lo:search_hi]
        dists = np.linalg.norm(local_traj - pos, axis=1)
        closest_local = int(np.argmin(dists))
        closest_idx = search_lo + closest_local
        closest_dist = dists[closest_local]

        # ── Adaptive reference advance ──
        # Only move forward if the device is within the leash distance
        if closest_dist < self.pd_leash:
            speed = self.pd_speed_base * (1.0 - closest_dist / self.pd_leash)
        else:
            speed = 0.0  # pause — device is too far behind

        self.auto_ref_idx = max(self.auto_ref_idx, float(closest_idx))
        self.auto_ref_idx = min(self.auto_ref_idx + speed, n - 1)

        # ── Carrot: target is a few indices AHEAD of the reference ──
        carrot_idx = min(int(self.auto_ref_idx) + self.pd_lookahead, n - 1)
        pos_ref = traj[carrot_idx]

        # ── Position error ──
        pos_err = pos_ref - pos

        # ── Filtered velocity estimation ──
        if self.prev_ee_phys is not None and dt > 0:
            vel_raw = (pos - self.prev_ee_phys) / dt
            self.vel_filtered = (self.vel_alpha * vel_raw +
                                 (1.0 - self.vel_alpha) * self.vel_filtered)
        else:
            self.vel_filtered = np.zeros(2)
        self.prev_ee_phys = pos.copy()

        # ── PD force ──
        fe = self.pd_kp * pos_err - self.pd_kd * self.vel_filtered

        # ── Force saturation (CRITICAL for Haply 2 stability) ──
        f_mag = np.linalg.norm(fe)
        if f_mag > self.pd_f_max:
            fe = fe * (self.pd_f_max / f_mag)

        # ── Completion check ──
        if int(self.auto_ref_idx) >= n - 1 and closest_dist < self.pd_leash:
            self.state = DONE
            self.auto_ref_idx = 0
            return np.zeros(2)

        return fe

    # ── Main loop ──

    def run(self):
        p = self.physics
        g = self.renderer

        keyups, xm = g.get_events()

        # Get device position
        if self.device_connected:
            pA0, pB0, pA, pB, pE = p.get_device_pos()
            pA0, pB0, pA, pB, xh = g.convert_pos(pA0, pB0, pA, pB, pE)
        else:
            xh = g.haptic.center

        xh = np.array(xh, dtype=float)
        g.erase_screen()

        # ── Keyboard handling ──
        for key in keyups:
            if key == ord('q'):
                self._save_results()
                sys.exit(0)

            if key == ord('r'):
                g.show_linkages = not g.show_linkages

            if key == ord('f'):
                g.show_debug = not g.show_debug

            if key == ord('h'):
                self.haptics.groove_enabled = not self.haptics.groove_enabled

            if key == ord('w'):
                self.haptics.walls_enabled = not self.haptics.walls_enabled
                if not self.haptics.walls_enabled:
                    self.haptics.reset_proxy()

            if key == ord('t') and self.state == IDLE and len(self.all_demos) == 0:
                self.crack_idx = (self.crack_idx + 1) % len(CRACK_NAMES)
                self.crack_name = CRACK_NAMES[self.crack_idx]
                self.crack = get_crack(self.crack_name)
                self.haptics = CrackHaptics(self.crack)
                self.sealed_indices.clear()
                g.pressure = 100.0

            if key == pygame.K_SPACE:
                if self.state == IDLE:
                    self.state = RECORDING
                    self.current_demo = []
                    self.current_seal_indices = set()
                    self.wall_hits = 0
                    self.start_time = time.time()
                elif self.state == RECORDING:
                    self.last_demo_time = time.time() - self.start_time
                    if len(self.current_demo) > 10:
                        demo_arr = np.array(self.current_demo)
                        self.all_demos.append(demo_arr)
                        self.all_demo_times.append(self.last_demo_time)
                        mnd = mean_nearest_distance(demo_arr, self.crack.centerline)
                        self.per_demo_metrics.append(mnd)
                        self.state = REVIEW
                    else:
                        self.state = IDLE

            if key == ord('d') and self.state == REVIEW:
                if len(self.all_demos) > 0:
                    self.all_demos.pop()
                    self.all_demo_times.pop()
                    self.per_demo_metrics.pop()
                    # Unseal the points from the deleted demo
                    self.sealed_indices -= self.current_seal_indices
                self.current_demo = []
                self.state = IDLE

            if key == pygame.K_RETURN:
                if self.state == REVIEW:
                    self.state = IDLE
                elif self.state == DONE:
                    self.gp_traj_phys = None
                    self.gp_traj_std = None
                    self.trial_metrics = None
                    self.state = IDLE

            if key == ord('g') and self.state in (IDLE, REVIEW) and len(self.all_demos) >= 1:
                self.state = TRAINING

            if key == ord('p') and self.state == DONE:
                self.playback_idx = 0
                self.state = PLAYBACK

            if key == ord('a') and self.state == DONE and self.gp_traj_phys is not None:
                self.auto_ref_idx = 0.0
                self.prev_ee_phys = None
                self.state = AUTO_PLAY

            if key == ord('n') and self.state == DONE:
                g.close()
                self.tlx_result = run_nasa_tlx()
                self.renderer = MarsRenderer(self.device_connected, window_size=(760, 720))
                g = self.renderer
                if self.all_results:
                    self.all_results[-1]["tlx"] = self.tlx_result
                g.erase_screen()
                return

            if key == ord('c') and self.state in (IDLE, REVIEW, DONE):
                self.all_demos = []
                self.all_demo_times = []
                self.per_demo_metrics = []
                self.current_demo = []
                self.gp_traj_phys = None
                self.gp_traj_std = None
                self.trial_metrics = None
                self.haptics.reset_proxy()
                self.haptics.clear_gp_trajectory()
                self.sealed_indices.clear()
                g.pressure = 100.0
                self.state = IDLE

        # ── Physics position ──
        pos_phys = np.array(g.inv_convert_pos(xh), dtype=float)

        # ── Haptic force computation ──
        if self.state == AUTO_PLAY and self.gp_traj_phys is not None:
            fe_phys = self._compute_pd_force(pos_phys, dt=1.0 / g.FPS)
            fe = np.array([fe_phys[0], -fe_phys[1]])
        else:
            fe_phys = self.haptics.compute_force(pos_phys, dt=1.0 / g.FPS)
            fe = np.array([fe_phys[0], -fe_phys[1]])

        # Track wall hits
        if self.state == RECORDING and self.haptics.last_penetration > 0:
            self.wall_hits += 1

        # ── Recording ──
        if self.state == RECORDING:
            self.current_demo.append(pos_phys.copy())
            self._update_seal(pos_phys)

        # ── GP Training ──
        if self.state == TRAINING:
            if len(self.all_demos) >= 1:
                # Show training splash
                font_big = pygame.font.SysFont("Consolas", 28, bold=True)
                splash = font_big.render(
                    f"Training GP on {len(self.all_demos)} demos...", True, (255, 200, 60))
                g.screenVR.blit(splash, (g.window_size[0] // 2 - splash.get_width() // 2,
                                         g.window_size[1] // 2 - 20))
                g.window.blit(g.screenHaptics, (0, 0))
                g.window.blit(g.screenVR, (g.window_size[0], 0))
                pygame.display.flip()

                self.gp = TrajectoryGP()
                self.gp.fit(self.all_demos)
                self.gp_traj_phys, self.gp_traj_std = self.gp.predict(
                    n_points=300, return_std=True)
                self.haptics.set_gp_trajectory(
                    self.gp_traj_phys, self.gp_traj_std, n_demos=len(self.all_demos))

                self.trial_metrics = compute_all_metrics(
                    np.concatenate(self.all_demos),
                    self.gp_traj_phys, self.crack.centerline)
                self.trial_metrics["condition"] = self.condition
                self.trial_metrics["crack"] = self.crack_name
                self.trial_metrics["n_demos"] = len(self.all_demos)
                self.trial_metrics["demo_times"] = self.all_demo_times.copy()
                self.trial_metrics["per_demo_mnd"] = [float(m) for m in self.per_demo_metrics]
                self.trial_metrics["wall_hits"] = self.wall_hits
                self.trial_metrics["seal_pct"] = self._get_seal_percentage()
                self.trial_metrics["final_pressure"] = g.pressure
                self.all_results.append(self.trial_metrics.copy())

                self.playback_idx = 0
                self.state = PLAYBACK
            else:
                self.state = IDLE

        if self.state == PLAYBACK:
            self.playback_idx += 2
            if self.playback_idx >= len(self.gp_traj_phys):
                self.playback_idx = len(self.gp_traj_phys)
                self.state = DONE

        # ── Update pressure ──
        seal_pct = self._get_seal_percentage()
        g.update_pressure(seal_pct, dt=1.0 / g.FPS)

        # ── Elapsed time ──
        elapsed = time.time() - self.start_time if self.state == RECORDING else 0.0
        progress = self.crack.progress(pos_phys)
        unsealed_mask = self._get_unsealed_mask()

        wall_hit = self.haptics.last_wall if self.haptics.last_penetration > 0 else None

        # ══════════════════ DRAWING ══════════════════════════════════════

        # Suppress default rectangular cursor
        g.haptic.width = 0
        g.haptic.height = 0

        # ── Left panel: Haptic view ──
        g._draw_haptic_crack(g.screenHaptics, self.crack,
                              highlight_wall=wall_hit, draw_centerline=True)

        # Draw recorded trajectories on haptic panel
        if self.state == RECORDING and len(self.current_demo) > 1:
            g._draw_trajectory(g.screenHaptics, self.current_demo,
                               color=(255, 80, 40), width=2)

        # GP trajectory on haptic panel
        if self.gp_traj_phys is not None and self.haptics.gp_groove_enabled:
            g._draw_trajectory(g.screenHaptics, self.gp_traj_phys,
                               color=(60, 120, 255), width=1)

        # Proxy dot
        if self.haptics.last_proxy_pos is not None:
            ps = g.convert_pos(self.haptics.last_proxy_pos)
            pygame.draw.circle(g.screenHaptics, (255, 165, 0), (int(ps[0]), int(ps[1])), 8)
            pygame.draw.circle(g.screenHaptics, (200, 0, 0), (int(ps[0]), int(ps[1])), 8, 2)

        # End-effector dot (haptic panel)
        pygame.draw.circle(g.screenHaptics, (255, 60, 30), (int(xh[0]), int(xh[1])), 10)
        pygame.draw.circle(g.screenHaptics, (255, 200, 150), (int(xh[0]), int(xh[1])), 4)

        # Force arrow
        if np.linalg.norm(fe) > 0.01:
            fscale = 50.0
            pygame.draw.line(g.screenHaptics, (0, 100, 255),
                             (int(xh[0]), int(xh[1])),
                             (int(xh[0] - fe[0] * fscale),
                              int(xh[1] - fe[1] * fscale)), 2)

        # Auto-play reference dot
        if self.state == AUTO_PLAY and self.gp_traj_phys is not None:
            ref_pos = self.gp_traj_phys[int(self.auto_ref_idx)]
            ref_s = g.convert_pos(ref_pos)
            pygame.draw.circle(g.screenHaptics, (0, 255, 100),
                               (int(ref_s[0]), int(ref_s[1])), 7, 2)

        # Key legend on haptic panel
        self._draw_key_legend(g.screenHaptics)

        # ── Right panel: Mission view ──
        condition_info = f"Condition: {self.condition}"
        g.render_vr_panel(
            self.crack, self.sealed_indices, unsealed_mask,
            pos_phys, STATE_NAMES[self.state], len(self.all_demos),
            self.crack_name, progress, elapsed, seal_pct,
            condition_info=condition_info, highlight_wall=wall_hit)

        # Draw demos on VR panel
        for i, demo in enumerate(self.all_demos):
            c = DEMO_COLORS[i % len(DEMO_COLORS)]
            g._draw_trajectory(g.screenVR, demo, color=c, width=1)
        if self.state == RECORDING and len(self.current_demo) > 1:
            c = DEMO_COLORS[len(self.all_demos) % len(DEMO_COLORS)]
            # Draw sealant trail (thicker, blue)
            g._draw_trajectory(g.screenVR, self.current_demo, color=(60, 160, 255), width=3)

        # GP trajectory + uncertainty on VR panel
        if self.gp_traj_phys is not None:
            gp_idx = self.playback_idx if self.state == PLAYBACK else len(self.gp_traj_phys)
            g._draw_gp_uncertainty(g.screenVR,
                                   self.gp_traj_phys[:gp_idx], self.gp_traj_std[:gp_idx])
            g._draw_trajectory(g.screenVR, self.gp_traj_phys[:gp_idx],
                               color=(100, 180, 255), width=3)

        # End-effector on VR panel
        ee_vr = g.convert_pos(pos_phys)
        # Sealant applicator cursor (glowing circle)
        for r in [16, 12, 8]:
            alpha = int(120 * (16 - r) / 8)
            s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (60, 180, 255, alpha), (r, r), r)
            g.screenVR.blit(s, (int(ee_vr[0]) - r, int(ee_vr[1]) - r))
        pygame.draw.circle(g.screenVR, (200, 230, 255), (int(ee_vr[0]), int(ee_vr[1])), 5)

        # Auto-play reference on VR
        if self.state == AUTO_PLAY and self.gp_traj_phys is not None:
            ref_vr = g.convert_pos(self.gp_traj_phys[int(self.auto_ref_idx)])
            pygame.draw.circle(g.screenVR, (0, 255, 100),
                               (int(ref_vr[0]), int(ref_vr[1])), 7, 2)

        # Instructions
        self._draw_instructions(g.screenVR)

        # ── Physics sim (for mouse mode) ──
        if self.device_connected:
            p.update_force(fe)
        else:
            xh = g.sim_forces(xh, fe, xm, mouse_k=0.5, mouse_b=0.8)
            pos_phys_s = g.inv_convert_pos(xh)
            pA0, pB0, pA, pB, pE = p.derive_device_pos(pos_phys_s)
            pA0, pB0, pA, pB, xh = g.convert_pos(pA0, pB0, pA, pB, pE)

        g.render_haptic_panel(pA0, pB0, pA, pB, xh, fe, xm)
        g.finalize_render(pA0, pB0, pA, pB, xh, fe, xm)

    # ── Save results ──

    def _save_results(self):
        if not self.all_results and not self.all_demos:
            return

        os.makedirs("results", exist_ok=True)
        session_id = int(time.time())
        folder = f"results/session_{session_id}"
        os.makedirs(folder, exist_ok=True)

        # Metrics JSON
        if self.all_results:
            with open(f"{folder}/metrics.json", "w") as f:
                json.dump(self.all_results, f, indent=2, default=str)

        # Demo trajectories
        for i, demo in enumerate(self.all_demos):
            np.save(f"{folder}/demo_{i + 1}.npy", demo)

        # GP trajectory
        if self.gp_traj_phys is not None:
            np.save(f"{folder}/gp_trajectory.npy", self.gp_traj_phys)
        if self.gp_traj_std is not None:
            np.save(f"{folder}/gp_std.npy", self.gp_traj_std)

        # Session summary
        summary = {
            "session_id": session_id,
            "crack": self.crack_name,
            "condition": self.condition,
            "n_demos": len(self.all_demos),
            "seal_pct": self._get_seal_percentage(),
            "final_pressure": self.renderer.pressure,
            "demo_files": [f"demo_{i + 1}.npy" for i in range(len(self.all_demos))],
        }
        with open(f"{folder}/summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Session saved to {folder}/")

    def close(self):
        self._save_results()
        self.renderer.close()
        self.physics.close()


# ── Entry point ──

if __name__ == "__main__":
    pa = MarsPA3()
    try:
        while True:
            pa.run()
    except SystemExit:
        pass
    finally:
        pa.close()
