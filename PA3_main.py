# -*- coding: utf-8 -*-
"""
PA3 — Condition 1: Kinesthetic Teaching — Tube Navigation
----------------------------------------------------------
Navigate a curved tube from START to END without touching the walls.

Left panel  = Haptic view (pantograph + tube + force feedback)
Right panel = VR view (tube + all demos + GP reproduction)

Keys:
  SPACE   start/stop recording
  ENTER   next demo (REVIEW) / add more demos (DONE)
  D       delete last recorded demo (REVIEW)
  G       train GP on all demos
  P       replay GP trajectory (DONE)
  N       NASA-TLX (DONE)
  H       toggle groove guidance on/off
  W       toggle virtual walls on/off
  T       cycle tube shape (IDLE, no demos)
  C       clear all
  R       toggle linkages
  Q       quit
"""

import sys, time, os, json
import numpy as np
import pygame
from tomlkit import key

from Physics import Physics
from Graphics import Graphics
from targets import get_tube, TUBE_NAMES
from haptics import TubeHaptics
from gp_trajectory import TrajectoryGP
from metrics import compute_all_metrics, mean_nearest_distance
from nasa_tlx import run_nasa_tlx

IDLE, RECORDING, REVIEW, TRAINING, PLAYBACK, DONE, AUTO_PLAY = range(7)
STATE_NAMES = ["IDLE", "RECORDING", "REVIEW", "TRAINING", "PLAYBACK", "DONE", "AUTO_PLAY:"]

DEMO_COLORS = [
    (220, 60, 60), (60, 60, 220), (220, 160, 0),
    (160, 0, 220), (0, 180, 180), (180, 100, 60),
    (100, 200, 60), (200, 60, 160),
]


class PA3_Kinesthetic:
    def __init__(self):
        self.physics = Physics(hardware_version=3)
        self.device_connected = self.physics.is_device_connected()
        self.graphics = Graphics(self.device_connected, window_size=(760, 720))
        self.graphics.show_debug = False
        pygame.display.set_caption("PA3 — Kinesthetic Teaching (Tube)")

        # ── Tube ──
        self.tube_idx = 0
        self.tube_name = TUBE_NAMES[self.tube_idx]
        self.tube = get_tube(self.tube_name)
        self.haptics = TubeHaptics(self.tube)

        # ── State ──
        self.state = IDLE
        self.all_demos = []
        self.all_demo_times = []
        self.per_demo_metrics = []
        self.current_demo = []
        self.start_time = 0.0
        self.last_demo_time = 0.0
        self.wall_hits = 0
        self.auto_train = False


        # ── GP ──
        self.gp = TrajectoryGP()
        self.gp_traj_phys = None
        self.gp_traj_std = None
        self.playback_idx = 0
        self.trial_metrics = None
        self.tlx_result = None
        self.all_results = []

        # ── Auto-play PD controller ──
        self.auto_ref_idx = 0.0       # closest + lookahead (for visualization only)
        self.pd_kp = 170.0     # N/m  — position gain
        self.pd_kd = 50.0       # N·s/m — velocity damping
        self.pd_speed = 0.75
        self.prev_ee_phys = None      # was missing — needed by _compute_pd_force


    # ─── Tube drawing ───────────────────────────────────────────────────
    # def _draw_tube(self, surface, highlight_wall=None):
    #     g = self.graphics
    #     tube = self.tube

    #     left_pts  = [g.convert_pos(p) for p in tube.wall_left]
    #     right_pts = [g.convert_pos(p) for p in tube.wall_right]
    #     center_pts = [g.convert_pos(p) for p in tube.centerline]

    #     left_color  = (255, 50, 50) if highlight_wall == 'left'  else (100, 100, 100)
    #     right_color = (255, 50, 50) if highlight_wall == 'right' else (100, 100, 100)

    #     # Draw walls first (will be covered by fill, then redrawn)
    #     if len(left_pts) > 1:
    #         pygame.draw.lines(surface, left_color, False, left_pts, 3)
    #     if len(right_pts) > 1:
    #         pygame.draw.lines(surface, right_color, False, right_pts, 3)

    #     # Fill tube interior
    #     for i in range(0, len(center_pts) - 1, 2):
    #         pygame.draw.line(surface, (230, 240, 230),
    #                          (int(center_pts[i][0]),   int(center_pts[i][1])),
    #                          (int(center_pts[i+1][0]), int(center_pts[i+1][1])),
    #                          int(tube.half_width * g.window_scale * 2))

    #     # Re-draw walls on top of fill
    #     if len(left_pts) > 1:
    #         pygame.draw.lines(surface, left_color, False, left_pts, 3)
    #     if len(right_pts) > 1:
    #         pygame.draw.lines(surface, right_color, False, right_pts, 3)

    #     # Centerline on top
    #     if len(center_pts) > 1:
    #         pygame.draw.lines(surface, (120, 120, 120), False, center_pts, 1)

    #     # Start / end markers
    #     start_s = g.convert_pos(tube.start)
    #     end_s   = g.convert_pos(tube.end)
    #     pygame.draw.circle(surface, (0, 200, 0),   (int(start_s[0]), int(start_s[1])), 10)
    #     pygame.draw.circle(surface, (200, 0, 0),   (int(end_s[0]),   int(end_s[1])),   10)
    #     font = pygame.font.SysFont("Arial", 12, bold=True)
    #     surface.blit(font.render("START", True, (0, 200, 0)),
    #                  (int(start_s[0]) + 12, int(start_s[1]) - 6))
    #     surface.blit(font.render("END",   True, (200, 0, 0)),
    #                  (int(end_s[0])   + 12, int(end_s[1])   - 6))

    def _draw_tube(self, surface, highlight_wall=None,
                   draw_fill=True, draw_walls=True, draw_centerline=True):
        g = self.graphics
        tube = self.tube
        left_pts   = [g.convert_pos(p) for p in tube.wall_left]
        right_pts  = [g.convert_pos(p) for p in tube.wall_right]
        center_pts = [g.convert_pos(p) for p in tube.centerline]
        left_color  = (255, 50, 50) if highlight_wall == 'left'  else (100, 100, 100)
        right_color = (255, 50, 50) if highlight_wall == 'right' else (100, 100, 100)

        if draw_fill:
            for i in range(0, len(center_pts) - 1, 2):
                pygame.draw.line(surface, (230, 240, 230),
                                 (int(center_pts[i][0]),   int(center_pts[i][1])),
                                 (int(center_pts[i+1][0]), int(center_pts[i+1][1])),
                                 int(tube.half_width * g.window_scale * 2))

        if draw_walls:
            if len(left_pts) > 1:
                pygame.draw.lines(surface, left_color, False, left_pts, 3)
            if len(right_pts) > 1:
                pygame.draw.lines(surface, right_color, False, right_pts, 3)

        if draw_centerline and len(center_pts) > 1:
            pygame.draw.lines(surface, (120, 120, 120), False, center_pts, 1)

        if draw_walls:
            start_s = g.convert_pos(tube.start)
            end_s   = g.convert_pos(tube.end)
            pygame.draw.circle(surface, (0, 200, 0), (int(start_s[0]), int(start_s[1])), 10)
            pygame.draw.circle(surface, (200, 0, 0), (int(end_s[0]),   int(end_s[1])),   10)
            font = pygame.font.SysFont("Arial", 12, bold=True)
            surface.blit(font.render("START", True, (0, 200, 0)),
                         (int(start_s[0]) + 12, int(start_s[1]) - 6))
            surface.blit(font.render("END",   True, (200, 0, 0)),
                         (int(end_s[0])   + 12, int(end_s[1])   - 6))


    def _draw_trajectory(self, surface, traj_phys, color, width=2):
        g = self.graphics
        if len(traj_phys) < 2:
            return
        pts = [g.convert_pos(p) for p in traj_phys]
        pygame.draw.lines(surface, color, False, pts, width)

    def _draw_gp_uncertainty(self, surface, traj, std):
        g = self.graphics
        for i in range(0, len(traj), 3):
            pt = g.convert_pos(traj[i])
            r = max(2, int(np.mean(std[i]) * g.window_scale * 2))
            r = min(r, 40)
            s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (100, 100, 255, 60), (r, r), r)
            surface.blit(s, (int(pt[0]) - r, int(pt[1]) - r))

    def _draw_key_legend(self, surface):
        """Draw a compact key reference in the top-left corner of a panel."""
        h = self.haptics
        groove_str = "ON " if h.groove_enabled else "OFF"
        walls_str  = "ON " if h.walls_enabled  else "OFF"

        lines = [
            "── Keys ──────────────",
            "SPACE  start/stop rec",
            "ENTER  confirm/next",
            f"D      delete last demo",
            "G      train GP",
            "P      replay GP",
            "A      auto-reproduce GP",
            f"W      walls   [{walls_str}]",
            f"H      groove  [{groove_str}]",
            f"K      GP groove [{' ON' if h.gp_groove_enabled else 'OFF'}]",
            f"J      fading groove [{' ON' if h.fading_groove_enabled else 'OFF'}]",
            "1-4    s_curve orient",
            "T      change tube",
            "C      clear all",
            "Q      quit",
        ]

        font = pygame.font.SysFont("Courier", 13)
        x, y = 8, 8
        pad = 4
        line_h = font.get_linesize()
        box_w = 220
        box_h = len(lines) * line_h + pad * 2

        # Semi-transparent background
        bg = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 140))
        surface.blit(bg, (x - pad, y - pad))

        for i, line in enumerate(lines):
            color = (200, 200, 200) if i > 0 else (255, 220, 80)
            surf = font.render(line, True, color)
            surface.blit(surf, (x, y + i * line_h))

    # ─── Main loop ──────────────────────────────────────────────────────
    def run(self):
        p = self.physics
        g = self.graphics

        keyups, xm = g.get_events()

        if self.device_connected:
            pA0, pB0, pA, pB, pE = p.get_device_pos()
            pA0, pB0, pA, pB, xh = g.convert_pos(pA0, pB0, pA, pB, pE)
        else:
            xh = g.haptic.center

        xh = np.array(xh, dtype=float)
        g.erase_screen()

        # ── Keyboard ────────────────────────────────────────────────────
        for key in keyups:
            if key == ord('q'):
                self._save_results()
                sys.exit(0)

            if key == ord('r'):
                g.show_linkages = not g.show_linkages

            # # H → toggle groove guidance
            if key == ord('h'):
                self.haptics.groove_enabled = not self.haptics.groove_enabled
                if self.haptics.groove_enabled:
                    self.haptics.gp_groove_enabled = False  
                    self.haptics.fading_groove_enabled = False


            # K → toggle GP groove
            if key == ord('k'):
                self.haptics.gp_groove_enabled = not self.haptics.gp_groove_enabled
                if self.haptics.gp_groove_enabled:
                    self.haptics.groove_enabled = False    
                    self.haptics.fading_groove_enabled = False

            # J → fading centerline groove
            if key == ord('j'):
                self.haptics.fading_groove_enabled = not self.haptics.fading_groove_enabled
                if self.haptics.fading_groove_enabled:
                    self.haptics.groove_enabled      = False
                    self.haptics.gp_groove_enabled   = False

            # W → toggle virtual walls
            if key == ord('w'):
                self.haptics.walls_enabled = not self.haptics.walls_enabled
                if not self.haptics.walls_enabled:
                    self.haptics.reset_proxy()

            # T → cycle tube (only when idle and no demos)
            if key == ord('t') and self.state == IDLE and len(self.all_demos) == 0:
                self.tube_idx = (self.tube_idx + 1) % len(TUBE_NAMES)
                self.tube_name = TUBE_NAMES[self.tube_idx]
                self.tube = get_tube(self.tube_name)
                self.haptics = TubeHaptics(self.tube)
                self.haptics.reset_proxy()

            # 1–4 → select s_curve orientation directly
            _orientation_map = {
                ord('1'): "s_curve",
                ord('2'): "s_curve_90",
                ord('3'): "s_curve_180",
                ord('4'): "s_curve_270",
            }
            if key in _orientation_map and self.state == IDLE and len(self.all_demos) == 0:
                self.tube_name = _orientation_map[key]
                self.tube = get_tube(self.tube_name)
                self.haptics = TubeHaptics(self.tube)
                self.haptics.reset_proxy()
            

            if key == pygame.K_SPACE:
                if self.state == IDLE:
                    self.state = RECORDING
                    self.current_demo = []
                    self.wall_hits = 0
                    self.start_time = time.time()
                elif self.state == RECORDING:
                    self.last_demo_time = time.time() - self.start_time
                    if len(self.current_demo) > 10:
                        demo_arr = np.array(self.current_demo)
                        self.all_demos.append(demo_arr)
                        self.all_demo_times.append(self.last_demo_time)
                        mnd = mean_nearest_distance(demo_arr, self.tube.centerline)
                        self.per_demo_metrics.append(mnd)
                        self.state = REVIEW
                    else:
                        self.state = IDLE

            # D → delete last demo (only in REVIEW)
            if key == ord('d') and self.state == REVIEW:
                if len(self.all_demos) > 0:
                    self.all_demos.pop()
                    self.all_demo_times.pop()
                    self.per_demo_metrics.pop()
                self.current_demo = []
                self.state = IDLE

            if key == pygame.K_RETURN:
                if self.state == REVIEW:
                    if (self.haptics.gp_groove_enabled or self.haptics.fading_groove_enabled) and len(self.all_demos) >= 1:
                        self.auto_train = True    # K is on → silent retrain then back to IDLE
                        self.state = TRAINING
                    else:
                        self.state = IDLE         # K is off → behaves exactly as before
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

            if key == ord('n') and self.state == DONE:
                g.close()
                self.tlx_result = run_nasa_tlx()
                self.graphics = Graphics(self.device_connected, window_size=(800, 800))
                self.graphics.show_debug = False
                pygame.display.set_caption("PA3 — Kinesthetic Teaching (Tube)")
                if self.all_results:
                    self.all_results[-1]["tlx"] = self.tlx_result
                g = self.graphics
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
                self.state = IDLE

            # A → start autonomous GP reproduction (only when GP is trained)
            if key == ord('a') and self.state == DONE and self.gp_traj_phys is not None:
                self.auto_ref_idx = 0.0
                self.prev_ee_phys = None
                self.state = AUTO_PLAY


        pos_phys = np.array(g.inv_convert_pos(xh), dtype=float)

        if self.state == AUTO_PLAY and self.gp_traj_phys is not None:
            fe_phys = self._compute_pd_force(pos_phys, dt=1.0 / g.FPS)
            fe = np.array([fe_phys[0], -fe_phys[1]])
        else:
            fe_phys = self.haptics.compute_force(pos_phys, dt=1.0 / g.FPS)
            fe = np.array([fe_phys[0], -fe_phys[1]])


        # Track wall hits
        if self.state == RECORDING and self.haptics.last_penetration > 0:
            self.wall_hits += 1

        # ── Record ──────────────────────────────────────────────────────
        if self.state == RECORDING:
            self.current_demo.append(pos_phys.copy())

        # ── GP Training ─────────────────────────────────────────────────
        if self.state == TRAINING:
            if len(self.all_demos) >= 1:
                font_big = pygame.font.SysFont("Arial", 32, bold=True)
                splash = font_big.render(
                    f"Training GP on {len(self.all_demos)} demos...", True, (255, 255, 255))
                g.screenVR.blit(splash, (g.window_size[0]//2 - splash.get_width()//2,
                                         g.window_size[1]//2 - 20))
                g.window.blit(g.screenHaptics, (0, 0))
                g.window.blit(g.screenVR, (g.window_size[0], 0))
                pygame.display.flip()

                self.gp = TrajectoryGP()
                self.gp.fit(self.all_demos)
                self.gp_traj_phys, self.gp_traj_std = self.gp.predict(
                    n_points=300, return_std=True)

                self.haptics.set_gp_trajectory(self.gp_traj_phys, self.gp_traj_std, n_demos=len(self.all_demos))


                self.trial_metrics = compute_all_metrics(
                    np.concatenate(self.all_demos),
                    self.gp_traj_phys, self.tube.centerline)
                self.trial_metrics["condition"] = "kinesthetic"
                self.trial_metrics["tube"] = self.tube_name
                self.trial_metrics["n_demos"] = len(self.all_demos)
                self.trial_metrics["demo_times"] = self.all_demo_times.copy()
                self.trial_metrics["per_demo_mnd"] = self.per_demo_metrics.copy()
                self.all_results.append(self.trial_metrics.copy())

                if self.auto_train:
                    self.auto_train = False          # silent retrain: skip playback, back to IDLE
                    self.state = IDLE
                else:
                    self.playback_idx = 0            # G was pressed: show playback as before
                    self.state = PLAYBACK


            else:
                self.state = IDLE

        if self.state == PLAYBACK:
            self.playback_idx += 2
            if self.playback_idx >= len(self.gp_traj_phys):
                self.playback_idx = len(self.gp_traj_phys)
                self.state = DONE

        # ══════════════════════ DRAWING ══════════════════════════════════

        # Suppress default rectangular haptic cursor
        g.haptic.width = 0
        g.haptic.height = 0

        wall_hit = self.haptics.last_wall if self.haptics.last_penetration > 0 else None
        
        # ── Left: Haptic panel ──────────────────────────────────────────
        # Draw tube without centerline — we handle it conditionally below
        self._draw_tube(g.screenHaptics, highlight_wall=wall_hit, draw_centerline=False)

        if self.state == RECORDING and len(self.current_demo) > 1:
            self._draw_trajectory(g.screenHaptics, self.current_demo,
                                  color=(220, 60, 60), width=2)

        # Guidance line on haptic panel
        if self.haptics.groove_enabled or self.haptics.fading_groove_enabled:
            center_pts = [g.convert_pos(p) for p in self.tube.centerline]
            if len(center_pts) > 1:
                pygame.draw.lines(g.screenHaptics, (120, 120, 120), False, center_pts, 1)
        elif self.haptics.gp_groove_enabled and self.gp_traj_phys is not None:
            self._draw_trajectory(g.screenHaptics, self.gp_traj_phys,
                                  color=(40, 40, 220), width=1)


        # Proxy dot
        if self.haptics.last_proxy_pos is not None:
            ps = g.convert_pos(self.haptics.last_proxy_pos)
            pygame.draw.circle(g.screenHaptics, (255, 165, 0), (int(ps[0]), int(ps[1])), 8)
            pygame.draw.circle(g.screenHaptics, (200, 0, 0),   (int(ps[0]), int(ps[1])), 8, 2)

        # End-effector dot
        pygame.draw.circle(g.screenHaptics, (220, 0, 0), (int(xh[0]), int(xh[1])), 10)

        # Auto-play reference dot on haptic panel
        if self.state == AUTO_PLAY and self.gp_traj_phys is not None:
            ref_pos = self.gp_traj_phys[int(self.auto_ref_idx)]
            ref_s = g.convert_pos(ref_pos)
            pygame.draw.circle(g.screenHaptics, (0, 255, 100),
                               (int(ref_s[0]), int(ref_s[1])), 7, 2)

        # Force arrow
        if np.linalg.norm(fe) > 0.01:
            fscale = 50.0
            pygame.draw.line(g.screenHaptics, (0, 0, 255),
                             (int(xh[0]), int(xh[1])),
                             (int(xh[0] - fe[0] * fscale),
                              int(xh[1] - fe[1] * fscale)), 2)

        # ── Right: VR panel ─────────────────────────────────────────────

        # 1. Fill only — behind uncertainty
        self._draw_tube(g.screenVR, draw_walls=False, draw_centerline=False)

        # 2. GP uncertainty — above fill, below walls
        if self.gp_traj_phys is not None:
            gp_idx = self.playback_idx if self.state == PLAYBACK else len(self.gp_traj_phys)
            self._draw_gp_uncertainty(g.screenVR,
                                      self.gp_traj_phys[:gp_idx], self.gp_traj_std[:gp_idx])

        # 3. Walls + markers — on top of uncertainty, no fill, no centerline
        self._draw_tube(g.screenVR, highlight_wall=wall_hit,
                        draw_fill=False, draw_centerline=False)

        # 4. Demos
        for i, demo in enumerate(self.all_demos):
            c = DEMO_COLORS[i % len(DEMO_COLORS)]
            self._draw_trajectory(g.screenVR, demo, color=c, width=1)
        if self.state == RECORDING and len(self.current_demo) > 1:
            c = DEMO_COLORS[len(self.all_demos) % len(DEMO_COLORS)]
            self._draw_trajectory(g.screenVR, self.current_demo, color=c, width=2)

        # 5. GP mean trajectory
        if self.gp_traj_phys is not None:
            self._draw_trajectory(g.screenVR, self.gp_traj_phys[:gp_idx],
                                  color=(40, 40, 220), width=3)

        # 6. Proxy dot
        if self.haptics.last_proxy_pos is not None:
            ps = g.convert_pos(self.haptics.last_proxy_pos)
            pygame.draw.circle(g.screenVR, (255, 165, 0), (int(ps[0]), int(ps[1])), 8)
            pygame.draw.circle(g.screenVR, (200, 0, 0),   (int(ps[0]), int(ps[1])), 8, 2)

        # 7. End-effector dot — always on top
        ee_vr = g.convert_pos(pos_phys)
        pygame.draw.circle(g.screenVR, (220, 0, 0), (int(ee_vr[0]), int(ee_vr[1])), 10)

        # 8. Auto-play reference dot
        if self.state == AUTO_PLAY and self.gp_traj_phys is not None:
            ref_vr = g.convert_pos(self.gp_traj_phys[int(self.auto_ref_idx)])
            pygame.draw.circle(g.screenVR, (0, 255, 100),
                               (int(ref_vr[0]), int(ref_vr[1])), 7, 2)

        # 9. Confidence bar
        if self.gp_traj_phys is not None and self.gp_traj_std is not None:
            dists = np.linalg.norm(self.gp_traj_phys - pos_phys, axis=1)
            nearest_idx = int(np.argmin(dists))
            local_std   = float(np.mean(self.gp_traj_std[nearest_idx]))
            std_min     = self.haptics.gp_std_min
            std_max     = self.haptics.gp_std_max
            std_factor  = float(np.clip((std_max - local_std) / (std_max - std_min), 0.0, 1.0))
            demo_factor = 1.0 - np.exp(-(len(self.all_demos) - 1) / 3.0)
            confidence  = std_factor * demo_factor

            bar_w, bar_h = 120, 14
            bar_x = g.window_size[0] - bar_w - 10
            bar_y = 10
            conf_font = pygame.font.SysFont("Courier", 12)
            pygame.draw.rect(g.screenVR, (40, 40, 40), (bar_x, bar_y, bar_w, bar_h))
            fill_color = (int(255 * (1 - confidence)), int(255 * confidence), 0)
            pygame.draw.rect(g.screenVR, fill_color,
                             (bar_x, bar_y, int(bar_w * confidence), bar_h))
            pygame.draw.rect(g.screenVR, (180, 180, 180), (bar_x, bar_y, bar_w, bar_h), 1)
            g.screenVR.blit(conf_font.render(
                f"conf:{confidence*100:.0f}% σ={local_std*1000:.1f}mm",
                True, (220, 220, 220)), (bar_x - 55, bar_y + bar_h + 3))







        # Key legend (top-left of VR panel, always visible)
        self._draw_key_legend(g.screenVR)

        # ── Debug text (built manually, drawn bottom-center of haptic panel) ──
        n = len(self.all_demos)
        state_str = STATE_NAMES[self.state]
        debug_parts = [
            f"[{state_str}]",
            f"demos={n}",
            f"tube={self.tube_name}",
            f"groove={'ON' if self.haptics.groove_enabled else 'OFF'}",
            f"walls={'ON' if self.haptics.walls_enabled else 'OFF'}",
            f"gpgroove={'ON' if self.haptics.gp_groove_enabled else 'OFF'}",
            f"fadgr={'ON' if self.haptics.fading_groove_enabled else 'OFF'}",
        ]
        if self.state == RECORDING:
            elapsed = time.time() - self.start_time
            prog = self.tube.progress(pos_phys)
            debug_parts += [f"t={elapsed:.1f}s",
                            f"prog={prog*100:.0f}%",
                            f"wall_hits={self.wall_hits}"]
        if self.trial_metrics and self.state in (PLAYBACK, DONE):
            debug_parts.append(f"GP_MND={self.trial_metrics['gp_mnd']*1000:.1f}mm")

        debug_str = "   ".join(debug_parts)
        debug_font = pygame.font.SysFont("Courier", 13)
        debug_surf = debug_font.render(debug_str, True, (200, 200, 200))

        dw, dh = debug_surf.get_width(), debug_surf.get_height()
        W = g.screenHaptics.get_width()
        H = g.screenHaptics.get_height()
        pad = 6
        bg = pygame.Surface((dw + pad * 2, dh + 4), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 160))

        # x where the text starts in full-window coordinates
        x_full = W - dw // 2
        y = H - dh - 8

        # Portion on left panel (screenHaptics)
        if x_full < W:
            clip_w_left = min(dw, W - x_full)
            g.screenHaptics.blit(bg,        (x_full - pad, y - 2),
                                  (0, 0, clip_w_left + pad, dh + 4))
            g.screenHaptics.blit(debug_surf, (x_full, y),
                                  (0, 0, clip_w_left, dh))

        # Portion on right panel (screenVR)
        x_vr = x_full - W          # x in screenVR coords (may be negative)
        src_x = max(0, -x_vr)      # skip pixels already drawn on left panel
        dst_x = max(0,  x_vr)
        clip_w_right = dw - src_x
        if clip_w_right > 0:
            g.screenVR.blit(bg,        (dst_x - pad, y - 2),
                             (src_x, 0, clip_w_right + pad, dh + 4))
            g.screenVR.blit(debug_surf, (dst_x, y),
                             (src_x, 0, clip_w_right, dh))


        # ── State-specific instructions (bottom of VR panel) ────────────
        font = pygame.font.SysFont("Arial", 16)
        y_base = g.window_size[1] - 160
        if self.state == IDLE:
            lines = [f"Demos: {n}  |  Tube: {self.tube_name}"]
            if n >= 1: lines.append("G = train GP")
        elif self.state == REVIEW:
            lines = [f"Demo #{n} saved!  ({self.last_demo_time:.1f}s)",
                     f"MND from center: {self.per_demo_metrics[-1]*1000:.1f}mm",
                     "ENTER = keep & next demo",
                     "D = delete this demo",
                     f"G = train GP on {n} demo{'s' if n > 1 else ''}"]
        elif self.state == DONE:
            m = self.trial_metrics
            lines = [f"GP trained on {m['n_demos']} demos",
                     f"GP MND: {m['gp_mnd']*1000:.2f}mm  |  "
                     f"Hausdorff: {m['gp_hausdorff']*1000:.2f}mm",
                     "A = auto-play  |  P = replay  |  ENTER = add demos",
                     "N = NASA-TLX  |  C = clear  |  Q = quit"]
        else:
            lines = []

        for i, line in enumerate(lines):
            surf = font.render(line, True, (255, 255, 255))
            g.screenVR.blit(surf, (20, y_base + i * 22))


        # ── Physics sim ─────────────────────────────────────────────────
        if self.device_connected:
            p.update_force(fe)
        else:
            xh = g.sim_forces(xh, fe, xm, mouse_k=0.5, mouse_b=0.8)
            pos_phys_s = g.inv_convert_pos(xh)
            pA0, pB0, pA, pB, pE = p.derive_device_pos(pos_phys_s)
            pA0, pB0, pA, pB, xh = g.convert_pos(pA0, pB0, pA, pB, pE)

        g.render(pA0, pB0, pA, pB, xh, fe, xm)

    
    def _compute_pd_force(self, pos_phys, dt):
        traj = self.gp_traj_phys
        n = len(traj)

        # Advance reference at fixed speed — guaranteed forward pull
        # pd_speed = indices per frame. At 100fps, 300 points in ~4s = 0.75/frame
        self.auto_ref_idx = min(self.auto_ref_idx + self.pd_speed, n - 1)
        ref_idx = int(self.auto_ref_idx)

        pos_ref = traj[ref_idx]
        pos_err = pos_ref - pos_phys

        if self.prev_ee_phys is not None and dt > 0:
            vel = (pos_phys - self.prev_ee_phys) / dt
        else:
            vel = np.zeros(2)
        self.prev_ee_phys = pos_phys.copy()

        fe = self.pd_kp * pos_err - self.pd_kd * vel

        if ref_idx >= n - 1:
            self.state = DONE
            self.auto_ref_idx = 0
            return np.zeros(2)

        return fe


    # def _save_results(self):
    #     if not self.all_results: return
    #     os.makedirs("results", exist_ok=True)
    #     fname = f"results/kinesthetic_{int(time.time())}.json"
    #     with open(fname, "w") as f:
    #         json.dump(self.all_results, f, indent=2, default=str)
    #     print(f"Results saved to {fname}")


    def _save_results(self):
        if not self.all_results and not self.all_demos:
            return

        os.makedirs("results", exist_ok=True)

        # One folder per session
        session_id = int(time.time())
        folder = f"results/session_{session_id}"
        os.makedirs(folder, exist_ok=True)

        # ── Metrics JSON (same as before) ──
        if self.all_results:
            with open(f"{folder}/metrics.json", "w") as f:
                json.dump(self.all_results, f, indent=2, default=str)

        # ── Individual demo trajectories ──
        for i, demo in enumerate(self.all_demos):
            np.save(f"{folder}/demo_{i+1}.npy", demo)

        # ── GP trajectory and std (final trained version) ──
        if self.gp_traj_phys is not None:
            np.save(f"{folder}/gp_trajectory.npy", self.gp_traj_phys)
        if self.gp_traj_std is not None:
            np.save(f"{folder}/gp_std.npy", self.gp_traj_std)

        # ── Session summary: tube, condition, n_demos ──
        summary = {
            "session_id":  session_id,
            "tube":        self.tube_name,
            "condition":   "kinesthetic",
            "n_demos":     len(self.all_demos),
            "demo_files":  [f"demo_{i+1}.npy" for i in range(len(self.all_demos))],
        }
        with open(f"{folder}/summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Session saved to {folder}/")


    def close(self):
        self._save_results()
        self.graphics.close()
        self.physics.close()


if __name__ == "__main__":
    pa = PA3_Kinesthetic()
    try:
        while True:
            pa.run()
    except SystemExit:
        pass
    finally:
        pa.close()
