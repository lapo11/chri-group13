# -*- coding: utf-8 -*-
"""
mars_renderer.py — Mars Habitat Crack-Sealing 2D Environment
==============================================================
Renders the full space-themed visual environment:
  - Mars habitat wall with procedural texture
  - Glowing crack with atmosphere leak particles
  - Sealant trail that "cures" behind the cursor
  - Pressure gauge dropping over time
  - Star field and Mars landscape in viewport
  - Status HUD with mission info
"""

import pygame
import numpy as np
import math
import random
import time


class Particle:
    """A single escaping-atmosphere particle."""
    __slots__ = ['x', 'y', 'vx', 'vy', 'life', 'max_life', 'size']

    def __init__(self, x, y, vx, vy, life=1.0, size=2):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = life
        self.max_life = life
        self.size = size


class MarsRenderer:
    """
    Handles all visual rendering for the Mars habitat scenario.
    Uses the same dual-panel layout as the original Graphics class
    but with a fully custom space-themed environment.
    
    Left panel  = Haptic view (device linkages + crack + force feedback)
    Right panel = Mission view (full environment with effects)
    """

    def __init__(self, device_connected, window_size=(760, 720)):
        self.device_connected = device_connected
        self.window_size = window_size

        pygame.init()
        self.window = pygame.display.set_mode((window_size[0] * 2, window_size[1]))
        pygame.display.set_caption("MARS HABITAT — Emergency Crack Seal")

        self.screenHaptics = pygame.Surface(self.window_size)
        self.screenVR = pygame.Surface(self.window_size)

        # Try to load icon, fallback gracefully
        try:
            self.icon = pygame.image.load('robot.png')
            pygame.display.set_icon(self.icon)
        except:
            pass

        self.font = pygame.font.Font('freesansbold.ttf', 18)
        pygame.mouse.set_visible(True)

        self.clock = pygame.time.Clock()
        self.FPS = 100

        # ── Colors ──
        self.MARS_WALL     = (75, 50, 45)       # dark reddish-brown wall
        self.MARS_WALL_LT  = (95, 65, 55)       # lighter variant
        self.CRACK_GLOW    = (255, 80, 30)       # orange-red crack glow
        self.CRACK_CORE    = (255, 160, 60)      # bright crack center
        self.SEALANT_FRESH = (60, 160, 255)      # blue sealant just applied
        self.SEALANT_CURED = (40, 100, 200)      # darker cured sealant
        self.ATMOSPHERE    = (200, 220, 255, 180) # white-blue atmosphere wisps
        self.STAR_COLOR    = (255, 255, 255)
        self.HUD_GREEN     = (0, 255, 120)
        self.HUD_RED       = (255, 60, 60)
        self.HUD_ORANGE    = (255, 180, 40)
        self.HUD_BG        = (15, 15, 25)
        self.MARS_SKY      = (180, 100, 60)

        # ── Coordinate system (same as original Graphics) ──
        self.window_scale = 5000
        self.device_origin = (int(self.window_size[0] / 2.0 + 0.038 / 2.0 * self.window_scale), 0)
        self.show_linkages = True
        self.show_debug = False

        # ── Haptic simulation (for mouse mode) ──
        self.haptic_width = 48
        self.haptic_height = 48
        self.haptic = pygame.Rect(*self.screenHaptics.get_rect().center, 0, 0).inflate(
            self.haptic_width, self.haptic_height)
        self.effort_cursor = pygame.Rect(*self.haptic.center, 0, 0).inflate(
            self.haptic_width, self.haptic_height)
        self.sim_k = 0.5
        self.sim_b = 0.8
        self.effort_color = (255, 255, 255)

        # ── Particles ──
        self.particles = []
        self.max_particles = 150

        # ── Stars (pre-generated) ──
        self.stars = [(random.randint(0, window_size[0]),
                       random.randint(0, window_size[1] // 3),
                       random.uniform(0.5, 2.5),
                       random.randint(150, 255))
                      for _ in range(80)]

        # ── Mars terrain points (simple horizon) ──
        self._generate_mars_terrain()

        # ── Wall texture (procedural noise patches) ──
        self._generate_wall_texture()

        # ── Pressure system ──
        self.pressure = 100.0        # starts at 100%
        self.pressure_leak_rate = 0.8  # % per second (when crack is unsealed)
        self.seal_progress = 0.0     # 0..1 fraction of crack sealed

        # ── Sealant trail ──
        self.sealed_points = []      # list of (screen_x, screen_y) where sealant was applied
        self.sealed_indices = set()  # set of crack centerline indices that are sealed

        # ── Animation timer ──
        self.anim_time = 0.0

        # ── Handle image ──
        try:
            self.hhandle = pygame.image.load('handle.png')
        except:
            self.hhandle = None

    def _generate_mars_terrain(self):
        """Generate a simple Mars horizon silhouette."""
        w = self.window_size[0]
        base_y = self.window_size[1] // 3
        self.terrain_points = []
        for x in range(0, w + 20, 20):
            y = base_y + int(15 * math.sin(x * 0.008) + 8 * math.sin(x * 0.023) + 
                            5 * math.sin(x * 0.05))
            self.terrain_points.append((x, y))
        # Close the polygon at bottom
        self.terrain_points.append((w, self.window_size[1]))
        self.terrain_points.append((0, self.window_size[1]))

    def _generate_wall_texture(self):
        """Pre-generate random texture spots for the wall."""
        w, h = self.window_size
        self.wall_spots = []
        for _ in range(200):
            x = random.randint(0, w)
            y = random.randint(0, h)
            r = random.randint(2, 8)
            shade = random.randint(-15, 15)
            self.wall_spots.append((x, y, r, shade))

    # ── Coordinate conversion (identical to original Graphics) ──
    def convert_pos(self, *positions):
        converted_positions = []
        for physics_pos in positions:
            x = self.device_origin[0] - physics_pos[0] * self.window_scale
            y = self.device_origin[1] + physics_pos[1] * self.window_scale
            converted_positions.append([x, y])
        if len(converted_positions) <= 0:
            return None
        elif len(converted_positions) == 1:
            return converted_positions[0]
        else:
            return converted_positions

    def inv_convert_pos(self, *positions):
        converted_positions = []
        for screen_pos in positions:
            x = (self.device_origin[0] - screen_pos[0]) / self.window_scale
            y = (screen_pos[1] - self.device_origin[1]) / self.window_scale
            converted_positions.append([x, y])
        if len(converted_positions) <= 0:
            return None
        elif len(converted_positions) == 1:
            return converted_positions[0]
        else:
            return converted_positions

    def get_events(self):
        events = pygame.event.get()
        keyups = []
        for event in events:
            if event.type == pygame.QUIT:
                import sys
                sys.exit(0)
            elif event.type == pygame.KEYUP:
                keyups.append(event.key)
        mouse_pos = pygame.mouse.get_pos()
        return keyups, mouse_pos

    def sim_forces(self, pE, f, pM, mouse_k=None, mouse_b=None):
        if mouse_k is not None:
            self.sim_k = mouse_k
        if mouse_b is not None:
            self.sim_b = mouse_b
        if not self.device_connected:
            pP = self.haptic.center
            diff = np.array((pM[0] - pE[0], pM[1] - pE[1]))
            scale = self.window_scale / 1e3
            scaled_vel_from_force = np.array(f) * scale / self.sim_b
            vel_from_mouse_spring = (self.sim_k / self.sim_b) * diff
            dpE = vel_from_mouse_spring - scaled_vel_from_force
            if abs(dpE[0]) < 1:
                dpE[0] = 0
            if abs(dpE[1]) < 1:
                dpE[1] = 0
            pE = np.round(pE + dpE)

            cg = 255 - np.clip(np.linalg.norm(self.sim_k * diff / self.window_scale) * 255 * 20, 0, 255)
            cb = 255 - np.clip(np.linalg.norm(self.sim_k * diff / self.window_scale) * 255 * 20, 0, 255)
            self.effort_color = (255, cg, cb)
        return pE

    # ──────────────────────────────────────────────────────────────────────
    # Particle system
    # ──────────────────────────────────────────────────────────────────────

    def _spawn_leak_particles(self, crack, unsealed_mask, screen):
        """Spawn particles from unsealed sections of the crack."""
        if len(self.particles) >= self.max_particles:
            return

        # Pick random unsealed points to emit from
        unsealed_indices = np.where(unsealed_mask)[0]
        if len(unsealed_indices) == 0:
            return

        n_spawn = min(3, self.max_particles - len(self.particles))
        for _ in range(n_spawn):
            idx = random.choice(unsealed_indices)
            pt = crack.centerline[idx]
            normal = crack.normals[idx]
            sx, sy = self.convert_pos(pt)

            # Emit along the normal direction (outward) with some randomness
            speed = random.uniform(15, 45)
            angle_offset = random.uniform(-0.5, 0.5)
            vx = normal[0] * speed * (1 if random.random() > 0.5 else -1) + random.uniform(-10, 10)
            # Particles float "upward" (negative y in screen coords)
            vy = -speed * 0.5 + random.uniform(-20, 5)

            life = random.uniform(0.8, 2.0)
            size = random.randint(1, 3)
            self.particles.append(Particle(sx, sy, vx, vy, life, size))

    def _update_particles(self, dt):
        """Update and cull particles."""
        alive = []
        for p in self.particles:
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.vy -= 5 * dt  # slight upward drift
            p.life -= dt
            if p.life > 0:
                alive.append(p)
        self.particles = alive

    def _draw_particles(self, surface):
        """Draw atmosphere leak particles."""
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p.life / p.max_life))))
            color = (200, 220, 255, alpha)
            s = pygame.Surface((p.size * 4, p.size * 4), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p.size * 2, p.size * 2), p.size * 2)
            surface.blit(s, (int(p.x) - p.size * 2, int(p.y) - p.size * 2))

    # ──────────────────────────────────────────────────────────────────────
    # Environment drawing
    # ──────────────────────────────────────────────────────────────────────

    def _draw_space_background(self, surface):
        """Draw the Mars sky and star field visible through a 'viewport'."""
        w, h = self.window_size
        viewport_h = h // 3

        # Gradient sky (dark space to Mars horizon glow)
        for y in range(viewport_h):
            t = y / viewport_h
            r = int(5 + t * 60)
            g = int(5 + t * 25)
            b = int(15 + t * 15)
            pygame.draw.line(surface, (r, g, b), (0, y), (w, y))

        # Stars (twinkle based on time)
        for sx, sy, size, brightness in self.stars:
            twinkle = int(brightness * (0.7 + 0.3 * math.sin(self.anim_time * 2 + sx * 0.1)))
            twinkle = max(0, min(255, twinkle))
            pygame.draw.circle(surface, (twinkle, twinkle, twinkle),
                             (sx, sy), max(1, int(size)))

        # Mars terrain
        if len(self.terrain_points) > 2:
            pygame.draw.polygon(surface, (140, 70, 40), self.terrain_points)
            # Terrain edge highlight
            pygame.draw.lines(surface, (180, 100, 60), False,
                            self.terrain_points[:-2], 2)

    def _draw_wall_surface(self, surface):
        """Draw the habitat wall background with texture."""
        w, h = self.window_size
        viewport_h = h // 3

        # Main wall
        pygame.draw.rect(surface, self.MARS_WALL, (0, viewport_h, w, h - viewport_h))

        # Texture spots
        for sx, sy, sr, shade in self.wall_spots:
            if sy > viewport_h:
                c = (max(0, min(255, self.MARS_WALL[0] + shade)),
                     max(0, min(255, self.MARS_WALL[1] + shade)),
                     max(0, min(255, self.MARS_WALL[2] + shade)))
                pygame.draw.circle(surface, c, (sx, sy), sr)

        # Viewport frame
        pygame.draw.line(surface, (100, 70, 55), (0, viewport_h), (w, viewport_h), 4)

        # Panel lines (rivet-like details on the wall)
        for x_offset in range(0, w, 120):
            pygame.draw.line(surface, (60, 40, 35),
                           (x_offset, viewport_h + 5), (x_offset, h), 1)
        for y_offset in range(viewport_h, h, 80):
            pygame.draw.line(surface, (60, 40, 35), (0, y_offset), (w, y_offset), 1)

    def _draw_crack(self, surface, crack, sealed_indices, highlight_wall=None):
        """
        Draw the crack with glow effect and sealant overlay.
        
        Parameters
        ----------
        surface : pygame.Surface
        crack   : CrackTrajectory
        sealed_indices : set of int — which centerline indices are sealed
        highlight_wall : 'left' or 'right' or None — which wall is being hit
        """
        # Convert crack points to screen coords
        center_pts = [self.convert_pos(p) for p in crack.centerline]
        left_pts = [self.convert_pos(p) for p in crack.wall_left]
        right_pts = [self.convert_pos(p) for p in crack.wall_right]

        # Draw crack glow (wide, semi-transparent)
        glow_surf = pygame.Surface(self.window_size, pygame.SRCALPHA)
        if len(center_pts) > 1:
            glow_width = int(crack.half_width * self.window_scale * 3.5)
            for i in range(len(center_pts) - 1):
                if i not in sealed_indices:
                    # Pulsating glow for unsealed sections
                    pulse = 0.7 + 0.3 * math.sin(self.anim_time * 4 + i * 0.05)
                    alpha = int(80 * pulse)
                    color = (*self.CRACK_GLOW, alpha)
                    p1 = (int(center_pts[i][0]), int(center_pts[i][1]))
                    p2 = (int(center_pts[i + 1][0]), int(center_pts[i + 1][1]))
                    pygame.draw.line(glow_surf, color, p1, p2, glow_width)
        surface.blit(glow_surf, (0, 0))

        # Draw crack core (bright center line)
        for i in range(len(center_pts) - 1):
            p1 = (int(center_pts[i][0]), int(center_pts[i][1]))
            p2 = (int(center_pts[i + 1][0]), int(center_pts[i + 1][1]))
            if i in sealed_indices:
                # Sealed = blue sealant
                pygame.draw.line(surface, self.SEALANT_CURED, p1, p2, 5)
                # Sealant glow
                gs = pygame.Surface((12, 12), pygame.SRCALPHA)
                pygame.draw.circle(gs, (40, 100, 200, 40), (6, 6), 6)
                surface.blit(gs, (p1[0] - 6, p1[1] - 6))
            else:
                # Unsealed = glowing crack
                pygame.draw.line(surface, self.CRACK_CORE, p1, p2, 3)
                pygame.draw.line(surface, (255, 220, 150), p1, p2, 1)

        # Draw walls
        left_color = (255, 50, 50) if highlight_wall == 'left' else (160, 80, 50)
        right_color = (255, 50, 50) if highlight_wall == 'right' else (160, 80, 50)

        if len(left_pts) > 1:
            pts = [(int(p[0]), int(p[1])) for p in left_pts]
            pygame.draw.lines(surface, left_color, False, pts, 2)
        if len(right_pts) > 1:
            pts = [(int(p[0]), int(p[1])) for p in right_pts]
            pygame.draw.lines(surface, right_color, False, pts, 2)

        # Start / end markers
        start_s = self.convert_pos(crack.start)
        end_s = self.convert_pos(crack.end)
        # Start = green beacon
        for r in [14, 10, 6]:
            alpha = int(80 * (14 - r) / 8)
            s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (0, 255, 100, alpha + 60), (r, r), r)
            surface.blit(s, (int(start_s[0]) - r, int(start_s[1]) - r))
        pygame.draw.circle(surface, (0, 255, 120), (int(start_s[0]), int(start_s[1])), 5)

        # End = red beacon
        for r in [14, 10, 6]:
            alpha = int(80 * (14 - r) / 8)
            s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (255, 60, 60, alpha + 60), (r, r), r)
            surface.blit(s, (int(end_s[0]) - r, int(end_s[1]) - r))
        pygame.draw.circle(surface, (255, 80, 80), (int(end_s[0]), int(end_s[1])), 5)

        # Labels
        label_font = pygame.font.SysFont("Consolas", 11, bold=True)
        surface.blit(label_font.render("START", True, (0, 255, 120)),
                     (int(start_s[0]) + 14, int(start_s[1]) - 5))
        surface.blit(label_font.render("END", True, (255, 80, 80)),
                     (int(end_s[0]) + 14, int(end_s[1]) - 5))

    def _draw_pressure_gauge(self, surface, pressure, x, y):
        """Draw a circular pressure gauge."""
        # Background arc
        radius = 45
        pygame.draw.circle(surface, (30, 30, 40), (x, y), radius + 3)
        pygame.draw.circle(surface, (15, 15, 25), (x, y), radius)

        # Pressure arc (270° sweep)
        start_angle = math.radians(135)
        sweep = math.radians(270)
        frac = pressure / 100.0

        # Draw filled arc segments
        n_segments = 40
        for i in range(int(n_segments * frac)):
            t = i / n_segments
            angle = start_angle - t * sweep
            x1 = x + int((radius - 8) * math.cos(angle))
            y1 = y - int((radius - 8) * math.sin(angle))
            x2 = x + int((radius - 2) * math.cos(angle))
            y2 = y - int((radius - 2) * math.sin(angle))

            if t < 0.3:
                color = self.HUD_RED
            elif t < 0.6:
                color = self.HUD_ORANGE
            else:
                color = self.HUD_GREEN

            pygame.draw.line(surface, color, (x1, y1), (x2, y2), 3)

        # Needle
        needle_angle = start_angle - frac * sweep
        nx = x + int((radius - 15) * math.cos(needle_angle))
        ny = y - int((radius - 15) * math.sin(needle_angle))
        pygame.draw.line(surface, (255, 255, 255), (x, y), (nx, ny), 2)
        pygame.draw.circle(surface, (200, 200, 200), (x, y), 4)

        # Label
        gauge_font = pygame.font.SysFont("Consolas", 12, bold=True)
        pct_text = gauge_font.render(f"{pressure:.0f}%", True, (220, 220, 220))
        surface.blit(pct_text, (x - pct_text.get_width() // 2, y + 14))

        label_text = gauge_font.render("PRESSURE", True, (140, 140, 150))
        surface.blit(label_text, (x - label_text.get_width() // 2, y + radius + 8))

    def _draw_mission_hud(self, surface, state_name, n_demos, crack_name,
                          progress, elapsed, seal_pct, condition_info=""):
        """Draw the heads-up display overlay."""
        w = self.window_size[0]
        hud_font = pygame.font.SysFont("Consolas", 13)
        title_font = pygame.font.SysFont("Consolas", 16, bold=True)

        # Top bar
        bar_h = 28
        bar_surf = pygame.Surface((w, bar_h), pygame.SRCALPHA)
        bar_surf.fill((0, 0, 0, 180))
        surface.blit(bar_surf, (0, 0))

        # Mission title
        title = title_font.render("MARS HABITAT — EMERGENCY CRACK SEAL", True, self.HUD_ORANGE)
        surface.blit(title, (10, 5))

        # Status
        status_text = f"[{state_name}]  Crack: {crack_name}  Demos: {n_demos}"
        if condition_info:
            status_text += f"  | {condition_info}"
        status = hud_font.render(status_text, True, (180, 180, 180))
        surface.blit(status, (w - status.get_width() - 10, 7))

        # Bottom info bar
        bottom_y = self.window_size[1] - 30
        bot_surf = pygame.Surface((w, 30), pygame.SRCALPHA)
        bot_surf.fill((0, 0, 0, 160))
        surface.blit(bot_surf, (0, bottom_y))

        info_parts = [
            f"Progress: {progress * 100:.0f}%",
            f"Sealed: {seal_pct * 100:.0f}%",
            f"Time: {elapsed:.1f}s",
        ]
        info_text = hud_font.render("   |   ".join(info_parts), True, (200, 200, 200))
        surface.blit(info_text, (10, bottom_y + 7))

    def _draw_seal_progress_bar(self, surface, seal_pct, x, y, w_bar=200, h_bar=16):
        """Draw a horizontal seal progress bar."""
        # Background
        pygame.draw.rect(surface, (30, 30, 40), (x, y, w_bar, h_bar), border_radius=3)

        # Fill
        fill_w = int(w_bar * seal_pct)
        if fill_w > 0:
            # Gradient from blue to green
            r = int(40 * (1 - seal_pct))
            g = int(100 + 155 * seal_pct)
            b = int(200 * (1 - seal_pct * 0.5))
            pygame.draw.rect(surface, (r, g, b), (x, y, fill_w, h_bar), border_radius=3)

        # Border
        pygame.draw.rect(surface, (100, 100, 120), (x, y, w_bar, h_bar), 1, border_radius=3)

        # Label
        bar_font = pygame.font.SysFont("Consolas", 11)
        label = bar_font.render(f"SEAL: {seal_pct * 100:.0f}%", True, (220, 220, 220))
        surface.blit(label, (x + w_bar // 2 - label.get_width() // 2, y + 1))

    def _draw_trajectory(self, surface, traj_phys, color, width=2):
        """Draw a trajectory on a surface."""
        if len(traj_phys) < 2:
            return
        pts = [self.convert_pos(p) for p in traj_phys]
        pts_int = [(int(p[0]), int(p[1])) for p in pts]
        pygame.draw.lines(surface, color, False, pts_int, width)

    def _draw_gp_uncertainty(self, surface, traj, std):
        """Draw GP uncertainty as translucent circles."""
        for i in range(0, len(traj), 3):
            pt = self.convert_pos(traj[i])
            r = max(2, int(np.mean(std[i]) * self.window_scale * 2))
            r = min(r, 40)
            s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (60, 120, 255, 50), (r, r), r)
            surface.blit(s, (int(pt[0]) - r, int(pt[1]) - r))

    def _draw_alert_flash(self, surface):
        """Draw red alert flash around edges when pressure is critically low."""
        if self.pressure < 30:
            intensity = int(40 * (0.5 + 0.5 * math.sin(self.anim_time * 6)))
            w, h = self.window_size
            alert = pygame.Surface((w, h), pygame.SRCALPHA)
            # Red border
            border = 8
            pygame.draw.rect(alert, (255, 0, 0, intensity), (0, 0, w, border))
            pygame.draw.rect(alert, (255, 0, 0, intensity), (0, h - border, w, border))
            pygame.draw.rect(alert, (255, 0, 0, intensity), (0, 0, border, h))
            pygame.draw.rect(alert, (255, 0, 0, intensity), (w - border, 0, border, h))
            surface.blit(alert, (0, 0))

    # ──────────────────────────────────────────────────────────────────────
    # Drawing for the haptic panel (left side)
    # ──────────────────────────────────────────────────────────────────────

    def _draw_haptic_crack(self, surface, crack, highlight_wall=None, draw_centerline=True):
        """Simplified crack drawing for the haptic panel."""
        center_pts = [self.convert_pos(p) for p in crack.centerline]
        left_pts = [self.convert_pos(p) for p in crack.wall_left]
        right_pts = [self.convert_pos(p) for p in crack.wall_right]

        # Fill the crack interior
        crack_width = int(crack.half_width * self.window_scale * 2)
        for i in range(0, len(center_pts) - 1, 2):
            p1 = (int(center_pts[i][0]), int(center_pts[i][1]))
            p2 = (int(center_pts[i + 1][0]), int(center_pts[i + 1][1]))
            pygame.draw.line(surface, (230, 200, 180), p1, p2, crack_width)

        # Walls
        left_color = (255, 50, 50) if highlight_wall == 'left' else (120, 80, 60)
        right_color = (255, 50, 50) if highlight_wall == 'right' else (120, 80, 60)

        if len(left_pts) > 1:
            pts = [(int(p[0]), int(p[1])) for p in left_pts]
            pygame.draw.lines(surface, left_color, False, pts, 3)
        if len(right_pts) > 1:
            pts = [(int(p[0]), int(p[1])) for p in right_pts]
            pygame.draw.lines(surface, right_color, False, pts, 3)

        if draw_centerline and len(center_pts) > 1:
            pts = [(int(p[0]), int(p[1])) for p in center_pts]
            pygame.draw.lines(surface, (150, 130, 110), False, pts, 1)

        # Start / end
        start_s = self.convert_pos(crack.start)
        end_s = self.convert_pos(crack.end)
        pygame.draw.circle(surface, (0, 200, 0), (int(start_s[0]), int(start_s[1])), 8)
        pygame.draw.circle(surface, (200, 0, 0), (int(end_s[0]), int(end_s[1])), 8)

    # ──────────────────────────────────────────────────────────────────────
    # Main render interface
    # ──────────────────────────────────────────────────────────────────────

    def erase_screen(self):
        self.screenHaptics.fill((240, 235, 225))
        self.screenVR.fill((0, 0, 0))
        self.debug_text = ""

    def render_haptic_panel(self, pA0, pB0, pA, pB, xh, fe, xm):
        """Render the left (haptic device) panel."""
        self.haptic.center = xh
        self.effort_cursor.center = self.haptic.center

        if self.device_connected:
            self.effort_color = (255, 255, 255)

        # Effort cursor
        pygame.draw.rect(self.screenHaptics, self.effort_color,
                        self.effort_cursor, border_radius=8)

        # Robot linkages
        if self.show_linkages:
            pantographColor = (150, 150, 150)
            pygame.draw.lines(self.screenHaptics, pantographColor, False, [pA0, pA], 15)
            pygame.draw.lines(self.screenHaptics, pantographColor, False, [pB0, pB], 15)
            pygame.draw.lines(self.screenHaptics, pantographColor, False, [pA, xh], 15)
            pygame.draw.lines(self.screenHaptics, pantographColor, False, [pB, xh], 15)

            for p in (pA0, pB0, pA, pB, xh):
                p_int = (int(p[0]), int(p[1]))
                pygame.draw.circle(self.screenHaptics, (0, 0, 0), p_int, 15)
                pygame.draw.circle(self.screenHaptics, (200, 200, 200), p_int, 6)

        # Handle image
        if self.hhandle:
            self.screenHaptics.blit(self.hhandle, self.effort_cursor)

        # Mouse tether line
        if not self.device_connected:
            pygame.draw.lines(self.screenHaptics, (0, 0, 0), False,
                            [self.effort_cursor.center, pM], 2) if hasattr(self, '_pM') else None

    def render_vr_panel(self, crack, sealed_indices, unsealed_mask,
                        pos_phys, state_name, n_demos, crack_name,
                        progress, elapsed, seal_pct, condition_info="",
                        highlight_wall=None):
        """Render the right (mission view) panel."""
        dt = 1.0 / max(self.FPS, 1)
        self.anim_time += dt

        # Background
        self._draw_space_background(self.screenVR)
        self._draw_wall_surface(self.screenVR)

        # Particles
        self._spawn_leak_particles(crack, unsealed_mask, self.screenVR)
        self._update_particles(dt)
        self._draw_particles(self.screenVR)

        # Crack with sealant
        self._draw_crack(self.screenVR, crack, sealed_indices, highlight_wall)

        # Pressure gauge
        self._draw_pressure_gauge(self.screenVR, self.pressure,
                                  self.window_size[0] - 65, self.window_size[1] // 3 + 65)

        # Seal progress bar
        self._draw_seal_progress_bar(self.screenVR, seal_pct,
                                     self.window_size[0] - 220,
                                     self.window_size[1] // 3 + 125)

        # Alert flash
        self._draw_alert_flash(self.screenVR)

        # HUD
        self._draw_mission_hud(self.screenVR, state_name, n_demos, crack_name,
                               progress, elapsed, seal_pct, condition_info)

    def finalize_render(self, pA0, pB0, pA, pB, xh, fe, xm):
        """Blit both panels and flip display."""
        self.window.blit(self.screenHaptics, (0, 0))
        self.window.blit(self.screenVR, (self.window_size[0], 0))

        # Debug overlay
        if self.show_debug:
            self.debug_text += f"FPS={round(self.clock.get_fps())} "
            self.debug_text += f"fe:[{np.round(fe[0], 1)},{np.round(fe[1], 1)}] "
            text = self.font.render(self.debug_text, True, (0, 0, 0), (255, 255, 255))
            self.window.blit(text, (10, 10))

        pygame.display.flip()
        self.clock.tick(self.FPS)

    def update_pressure(self, seal_pct, dt):
        """Update pressure based on how much of the crack is sealed."""
        leak_factor = max(0.0, 1.0 - seal_pct)
        self.pressure -= self.pressure_leak_rate * leak_factor * dt
        self.pressure = max(0.0, min(100.0, self.pressure))

    def close(self):
        pygame.display.quit()
        pygame.quit()
