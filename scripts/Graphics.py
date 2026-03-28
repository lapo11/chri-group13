# -*- coding: utf-8 -*-

import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pygame


ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"


class LeakParticle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "size", "color")

    def __init__(self, x, y, vx, vy, life, size, color):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = life
        self.max_life = life
        self.size = size
        self.color = color


class Graphics:
    def __init__(self,device_connected,window_size=(600,400)):
        self.device_connected = device_connected
        
        #initialize pygame window
        self.window_size = window_size #default (600,400)
        pygame.init()
        self.window = pygame.display.set_mode((window_size[0]*2, window_size[1]))   ##twice 600x400 for haptic and VR
        pygame.display.set_caption('Virtual Haptic Device')

        self.screenHaptics = pygame.Surface(self.window_size)
        self.screenVR = pygame.Surface(self.window_size)

        ##add nice icon from https://www.flaticon.com/authors/vectors-market
        self.icon = pygame.image.load(str(ASSET_DIR / "robot.png"))
        pygame.display.set_icon(self.icon)

        ##add text on top to debugToggle the timing and forces
        self.font = pygame.font.Font('freesansbold.ttf', 18)

        pygame.mouse.set_visible(True)     ##Hide cursor by default. 'm' toggles it
         
        ##set up the on-screen debugToggle
        self.text = self.font.render('Virtual Haptic Device', True, (0, 0, 0),(255, 255, 255))
        self.textRect = self.text.get_rect()
        self.textRect.topleft = (10, 10)

        #xc,yc = screenVR.get_rect().center ##center of the screen

        ##initialize "real-time" clock
        self.clock = pygame.time.Clock()
        self.FPS = 100   #in Hertz

        ##define some colors
        self.cWhite = (255,255,255)
        self.cDarkblue = (36,90,190)
        self.cLightblue = (0,176,240)
        self.cRed = (255,0,0)
        self.cOrange = (255,100,0)
        self.cYellow = (255,255,0)
        
        self.hhandle = pygame.image.load(str(ASSET_DIR / "handle.png")) #
        
        self.haptic_width = 48
        self.haptic_height = 48
        self.haptic  = pygame.Rect(*self.screenHaptics.get_rect().center, 0, 0).inflate(self.haptic_width, self.haptic_height)
        self.effort_cursor  = pygame.Rect(*self.haptic.center, 0, 0).inflate(self.haptic_width, self.haptic_height) 
        self.colorHaptic = self.cOrange ##color of the wall
        self.effort_color = (255, 255, 255)

        ####Pseudo-haptics dynamic parameters, k/b needs to be <1
        self.sim_k = 0.5 #0.1#0.5       ##Stiffness between cursor and haptic display
        self.sim_b = 0.8 #1.5#0.8       ##Viscous of the pseudohaptic display
        
        self.window_scale = 5000 #pixels per meter
        self.haptic_origin = (int(self.window_size[0]/2.0 + 0.038/2.0*self.window_scale), 0)
        self.vr_origin = (int(self.window_size[0] / 2.0), 0)
        self.device_origin = self.haptic_origin
        
        self.show_linkages = True
        self.show_debug = True
        self.hide_haptic_vr = False  # set True to suppress haptic square on screenVR

        # ── Mars greenhouse VR panel ────────────────────────────────────
        self.anim_time = 0.0
        # The breach is on the pressurized greenhouse envelope. The top of the
        # scene shows the Martian exterior; the lower scene shows the glazing,
        # structure and crop bay inside the greenhouse.
        self.MARS_WALL = (124, 146, 142)
        self.MARS_WALL_LT = (188, 210, 204)
        self.MARS_WALL_DK = (56, 78, 74)
        self.MARS_DUST = (178, 168, 154)
        self.VIEWPORT_METAL = (104, 118, 112)
        self.VIEWPORT_SHADOW = (32, 38, 36)
        self.HULL_TRIM = (156, 176, 170)
        self.HULL_STRIPE = (200, 168, 72)
        self.HULL_STRIPE_DK = (66, 58, 34)
        self.CRACK_GLOW = (255, 80, 30)
        self.CRACK_CORE = (255, 160, 60)
        self.CRACK_HOT = (255, 228, 180)
        self.HUD_CYAN = (110, 205, 235)
        self.HUD_ORANGE = (255, 180, 40)
        self.STAR_COLOR = (255, 255, 255)
        self.WARNING_YELLOW = (222, 176, 64)
        self.WARNING_BLACK = (26, 26, 26)
        self.LEAK_COLOR = (210, 225, 255)
        self.LEAK_HOT = (255, 210, 160)
        self.HULL_ACCENT = (82, 190, 154)
        self.HULL_ACCENT_DK = (28, 84, 68)
        self.BULKHEAD_CORE = (72, 88, 82)
        self.BULKHEAD_INNER = (40, 52, 48)
        self.SCORCH = (52, 28, 22)
        self.REGOLITH_DK = (104, 58, 34)
        self.REGOLITH_MD = (146, 82, 48)
        self.REGOLITH_LT = (188, 116, 72)
        self.GREEN_GROW = (108, 188, 104)
        self.GREEN_GROW_DK = (58, 118, 64)
        self.GROW_LIGHT = (255, 224, 170)
        self.SOLAR_PANEL = (74, 108, 140)
        self.HABITAT_WHITE = (196, 206, 198)
        self.HABITAT_TRIM = (116, 128, 124)

        self.stars = [
            (
                random.randint(0, window_size[0]),
                random.randint(0, window_size[1] // 3),
                random.uniform(0.5, 2.5),
                random.randint(150, 255),
                random.uniform(0.0, math.tau),
            )
            for _ in range(80)
        ]
        self.dust_motes = []
        self.leak_particles = []
        self.max_leak_particles = 180
        self._generate_mars_terrain()
        self._generate_wall_texture()
        self._generate_wall_panels()
        self._generate_dust_motes()
        self._generate_porthole()
        self._generate_exterior_outpost()
        self._generate_greenhouse_modules()

    def convert_pos(self,*positions):
        return self._convert_with_origin(self.haptic_origin, *positions)

    def convert_pos_vr(self,*positions):
        return self._convert_with_origin(self.vr_origin, *positions)

    def _convert_with_origin(self, origin, *positions):
        #invert x because of screen axes
        # 0---> +X
        # |
        # |
        # v +Y
        converted_positions = []
        for physics_pos in positions:
            x = origin[0]-physics_pos[0]*self.window_scale
            y = origin[1]+physics_pos[1]*self.window_scale
            converted_positions.append([x,y])
        if len(converted_positions)<=0:
            return None
        elif len(converted_positions)==1:
            return converted_positions[0]
        else:
            return converted_positions
        return [x,y]
    def inv_convert_pos(self,*positions):
        #convert screen positions back into physical positions
        converted_positions = []
        for screen_pos in positions:
            x = (self.device_origin[0]-screen_pos[0])/self.window_scale
            y = (screen_pos[1]-self.device_origin[1])/self.window_scale
            converted_positions.append([x,y])
        if len(converted_positions)<=0:
            return None
        elif len(converted_positions)==1:
            return converted_positions[0]
        else:
            return converted_positions
        return [x,y]
        
    def get_events(self):
        #########Process events  (Mouse, Keyboard etc...)#########
        events = pygame.event.get()
        keyups = []
        for event in events:
            if event.type == pygame.QUIT: #close window button was pressed
                sys.exit(0) #raises a system exit exception so any Finally will actually execute
            elif event.type == pygame.KEYUP:
                keyups.append(event.key)
        
        mouse_pos = pygame.mouse.get_pos()
        return keyups, mouse_pos

    def sim_forces(self,pE,f,pM,mouse_k=None,mouse_b=None):
        #simulated device calculations
        if mouse_k is not None:
            self.sim_k = mouse_k
        if mouse_b is not None:
            self.sim_b = mouse_b
        if not self.device_connected:
            pP = self.haptic.center
            #pM is where the mouse is
            #pE is where the position is pulled towards with the spring and damping factors
            #pP is where the actual haptic position ends up as
            diff = np.array(( pM[0]-pE[0],pM[1]-pE[1]) )
            #diff = np.array(( pM[0]-pP[0],pM[1]-pP[1]) )
            
            scale = self.window_scale/1e3
            scaled_vel_from_force = np.array(f)*scale/self.sim_b
            vel_from_mouse_spring = (self.sim_k/self.sim_b)*diff
            dpE = vel_from_mouse_spring - scaled_vel_from_force
            #dpE = -dpE
            #if diff[0]!=0:
            #    if (diff[0]+dpE[0])/diff[0]<0:
            #        #adding dpE has changed the sign (meaning the distance that will be moved is greater than the original displacement
            #        #prevent the instantaneous velocity from exceeding the original displacement (doesn't make physical sense)
            #        #basically if the force given is so high that in a single "tick" it would cause the endpoint to move back past it's original position...
            #        #whatever thing is exerting the force should basically be considered a rigid object
            #        dpE[0] = -diff[0]
            #if diff[1]!=1:
            #    if (diff[1]+dpE[1])/diff[1]<0:
            #        dpE[1] = -diff[1]
            if abs(dpE[0])<1:
                dpE[0] = 0
            if abs(dpE[1])<1:
                dpE[1] = 0
            pE = np.round(pE+dpE) #update new positon of the end effector
            
            #Change color based on effort
            cg = 255-np.clip(np.linalg.norm(self.sim_k*diff/self.window_scale)*255*20,0,255)
            cb = 255-np.clip(np.linalg.norm(self.sim_k*diff/self.window_scale)*255*20,0,255)
            self.effort_color = (255,cg,cb)
        return pE

    def _generate_mars_terrain(self):
        """Generate layered Martian ridges and the local regolith foreground."""
        w, h = self.window_size
        viewport_h = h // 3

        self.far_terrain_points = []
        far_base = int(viewport_h * 0.66)
        for x in range(0, w + 28, 28):
            y = far_base + int(
                10 * math.sin(x * 0.010)
                + 5 * math.sin(x * 0.024 + 0.8)
                + 3 * math.sin(x * 0.052 + 1.7)
            )
            self.far_terrain_points.append((x, y))
        self.far_terrain_points.append((w, viewport_h))
        self.far_terrain_points.append((0, viewport_h))

        self.mid_terrain_points = []
        mid_base = int(viewport_h * 0.77)
        for x in range(0, w + 24, 24):
            y = mid_base + int(
                14 * math.sin(x * 0.007 + 0.4)
                + 8 * math.sin(x * 0.020 + 1.3)
                + 4 * math.sin(x * 0.046 + 0.6)
            )
            self.mid_terrain_points.append((x, y))
        self.mid_terrain_points.append((w, viewport_h))
        self.mid_terrain_points.append((0, viewport_h))

        self.terrain_points = []
        near_base = int(viewport_h * 0.89)
        for x in range(0, w + 20, 20):
            y = near_base + int(
                17 * math.sin(x * 0.008 + 0.1)
                + 10 * math.sin(x * 0.022 + 0.9)
                + 5 * math.sin(x * 0.050 + 2.1)
            )
            self.terrain_points.append((x, y))
        self.terrain_points.append((w, viewport_h))
        self.terrain_points.append((0, viewport_h))

    def _generate_exterior_outpost(self):
        """Pre-compute a few exterior habitat elements visible beyond the dome."""
        w, h = self.window_size
        viewport_h = h // 3
        rng = random.Random(19)

        self.exterior_modules = []
        base_y = int(viewport_h * 0.80)
        self.exterior_modules.extend(
            [
                ("dome", int(w * 0.15), base_y + 8, 44, 24),
                ("greenhouse", int(w * 0.33), base_y + 6, 94, 28),
                ("tower", int(w * 0.52), base_y + 2, 22, 36),
                ("barrel", int(w * 0.66), base_y + 4, 62, 22),
                ("greenhouse", int(w * 0.83), base_y + 8, 86, 26),
            ]
        )

        self.exterior_solar_arrays = []
        for cx, cy, width in (
            (int(w * 0.09), base_y + 12, 70),
            (int(w * 0.74), base_y + 8, 82),
        ):
            tilt = rng.choice((-10, -6, 8, 12))
            self.exterior_solar_arrays.append((cx, cy, width, tilt))

        self.exterior_masts = [
            (int(w * 0.56), base_y - 20, 34),
            (int(w * 0.72), base_y - 14, 26),
        ]

        self.exterior_rocks = []
        for _ in range(28):
            x = rng.randint(10, w - 10)
            y = rng.randint(int(viewport_h * 0.78), viewport_h - 10)
            rx = rng.randint(4, 16)
            ry = rng.randint(2, 8)
            self.exterior_rocks.append((x, y, rx, ry))

        self.exterior_walkway = [
            (int(w * 0.10), base_y + 14),
            (int(w * 0.26), base_y + 12),
            (int(w * 0.40), base_y + 14),
            (int(w * 0.56), base_y + 10),
            (int(w * 0.74), base_y + 12),
            (int(w * 0.90), base_y + 14),
        ]
        self.exterior_rover = (int(w * 0.58), base_y + 14)

    def _generate_greenhouse_modules(self):
        """Pre-compute hydroponic racks, service tanks and interior hardware."""
        w, h = self.window_size
        viewport_h = h // 3
        rng = random.Random(27)

        self.grow_racks = []
        rack_xs = [44, 174, w - 272, w - 142]
        for x in rack_xs:
            top = viewport_h + rng.randint(30, 60)
            rack_h = rng.randint(178, 250)
            rack_w = rng.randint(58, 74)
            rect = pygame.Rect(x, top, rack_w, rack_h)
            shelves = []
            shelf_count = rng.randint(4, 5)
            for shelf_idx in range(shelf_count):
                y = rect.top + 22 + shelf_idx * ((rect.height - 40) // max(1, shelf_count - 1))
                shelves.append(y)
            leaf_clusters = []
            for shelf_y in shelves:
                for _ in range(rng.randint(5, 8)):
                    lx = rng.randint(rect.left + 10, rect.right - 10)
                    ly = shelf_y - rng.randint(2, 8)
                    rw = rng.randint(8, 18)
                    rh = rng.randint(5, 10)
                    tint = rng.choice(
                        [
                            (76, 148, 82),
                            (102, 172, 96),
                            (128, 196, 108),
                            (88, 158, 118),
                        ]
                    )
                    leaf_clusters.append((lx, ly, rw, rh, tint))
            self.grow_racks.append((rect, shelves, leaf_clusters))

        self.service_tanks = []
        for x in (w - 96, 18):
            tank = pygame.Rect(x, h - 172, 54, 112)
            self.service_tanks.append(tank)

        self.feed_pipes = []
        pipe_y = viewport_h + 36
        for x0, x1 in ((26, w - 26), (58, w - 58)):
            bend = rng.randint(12, 26)
            self.feed_pipes.append(((x0, pipe_y), (x1, pipe_y + bend)))
            pipe_y += 18

        self.planter_bins = []
        for x in range(126, w - 168, 118):
            self.planter_bins.append(pygame.Rect(x, h - 118 + rng.randint(-5, 5), 92, 38))

    def _generate_wall_texture(self):
        """Pre-generate dust, condensation and wear for greenhouse glazing."""
        w, h = self.window_size
        self.wall_spots = []
        for _ in range(200):
            x = random.randint(0, w)
            y = random.randint(0, h)
            r = random.randint(2, 8)
            shade = random.randint(-15, 15)
            self.wall_spots.append((x, y, r, shade))

        self.wall_streaks = []
        for _ in range(55):
            x = random.randint(15, w - 15)
            y = random.randint(h // 3 + 30, h - 40)
            length = random.randint(25, 110)
            alpha = random.randint(12, 34)
            self.wall_streaks.append((x, y, length, alpha))

    def _generate_wall_panels(self):
        """Generate glazing segments, trusses and service hardware for the greenhouse shell."""
        w, h = self.window_size
        viewport_h = h // 3
        self.wall_panels = []
        self.wall_rivets = []
        self.warning_plates = []
        panel_w = 148
        panel_h = 116

        for py in range(viewport_h + 18, h - 14, panel_h):
            for px in range(12, w - 12, panel_w):
                jitter_x = random.randint(-8, 8)
                jitter_y = random.randint(-6, 6)
                rect = pygame.Rect(px + jitter_x, py + jitter_y, panel_w - 14, panel_h - 12)
                self.wall_panels.append(rect)
                corners = [
                    (rect.left + 10, rect.top + 10),
                    (rect.right - 10, rect.top + 10),
                    (rect.left + 10, rect.bottom - 10),
                    (rect.right - 10, rect.bottom - 10),
                ]
                self.wall_rivets.extend(corners)
                if random.random() < 0.2:
                    self.warning_plates.append(
                        pygame.Rect(rect.left + 12, rect.top + 12, 78, 18)
                    )

        self.vent_slots = []
        for _ in range(7):
            vx = random.randint(40, w - 150)
            vy = random.randint(viewport_h + 65, h - 90)
            self.vent_slots.append(pygame.Rect(vx, vy, random.randint(78, 120), 16))

        self.crop_beds = []
        bed_y = h - 108
        for px in range(34, w - 74, 104):
            self.crop_beds.append(pygame.Rect(px, bed_y + random.randint(-6, 6), 84, 34))

    def _generate_dust_motes(self):
        """Create slow moving dust inside the greenhouse volume."""
        w, h = self.window_size
        self.dust_motes = []
        for _ in range(70):
            self.dust_motes.append([
                random.uniform(0, w),
                random.uniform(h // 3 + 20, h - 30),
                random.uniform(0.6, 2.2),
                random.uniform(10.0, 45.0),
                random.uniform(-6.0, 6.0),
                random.uniform(0.25, 1.0),
            ])

    def _generate_porthole(self):
        """Pre-compute a side greenhouse inspection window with visible crop rows."""
        w, h = self.window_size
        viewport_h = h // 3
        self.porthole_cx = int(w * 0.14)
        self.porthole_cy = int(viewport_h + (h - viewport_h) * 0.50)
        self.porthole_r = min(int(w * 0.095), int((h - viewport_h) * 0.30))

        rng = random.Random(7)
        self.porthole_stars = [
            (rng.uniform(-0.90, 0.90), rng.uniform(-0.90, 0.90), rng.randint(1, 2), rng.randint(100, 170))
            for _ in range(18)
        ]
        self.window_leaf_clusters = []
        for _ in range(18):
            self.window_leaf_clusters.append(
                (
                    rng.uniform(-0.62, 0.62),
                    rng.uniform(-0.35, 0.72),
                    rng.randint(7, 16),
                    rng.choice([(78, 148, 82), (98, 172, 96), (132, 194, 112)]),
                )
            )

    def _draw_porthole(self, surface):
        """Draw a circular inspection window looking into an adjacent crop bay."""
        cx = self.porthole_cx
        cy = self.porthole_cy
        r = self.porthole_r
        d = r * 2

        port = pygame.Surface((d, d), pygame.SRCALPHA)
        port.fill((18, 26, 20))

        for y in range(d):
            t = y / max(1, d - 1)
            col = (
                int(32 + 14 * (1.0 - t)),
                int(46 + 26 * (1.0 - t)),
                int(28 + 12 * (1.0 - t)),
            )
            pygame.draw.line(port, col, (0, y), (d, y))

        for shelf_y in (int(d * 0.58), int(d * 0.76)):
            pygame.draw.line(port, (88, 98, 84), (14, shelf_y), (d - 14, shelf_y), 4)

        for rel_x, rel_y, leaf_r, color in self.window_leaf_clusters:
            px = int(r + rel_x * r)
            py = int(r + rel_y * r)
            pygame.draw.ellipse(port, color, (px - leaf_r, py - leaf_r // 2, leaf_r * 2, leaf_r))
            pygame.draw.line(port, (178, 196, 154), (px, py + leaf_r // 2), (px, py + leaf_r), 1)

        for sx_n, sy_n, ssize, sbright in self.porthole_stars:
            px = int(r + sx_n * r)
            py = int(r + sy_n * r)
            if 0 < px < d and 0 < py < d:
                tw = int(sbright * (0.82 + 0.18 * math.sin(self.anim_time * 1.1 + px * 0.07)))
                tw = max(0, min(255, tw))
                pygame.draw.circle(port, (tw, tw, tw), (px, py), ssize)

        clip = pygame.Surface((d, d), pygame.SRCALPHA)
        clip.fill((0, 0, 0, 0))
        pygame.draw.circle(clip, (255, 255, 255, 255), (r, r), r)
        port.blit(clip, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        surface.blit(port, (cx - r, cy - r))

        pygame.draw.circle(surface, (22, 26, 34), (cx, cy), r + 17)
        pygame.draw.circle(surface, (66, 82, 76), (cx, cy), r + 13)
        pygame.draw.circle(surface, (122, 138, 132), (cx, cy), r + 10)
        pygame.draw.circle(surface, (84, 100, 96), (cx, cy), r + 7)
        pygame.draw.circle(surface, (36, 42, 52), (cx, cy), r + 4)
        pygame.draw.circle(surface, (14, 16, 22), (cx, cy), r + 1, 2)

        for i in range(8):
            angle = math.pi / 8 + i * math.tau / 8
            bx = int(cx + (r + 11) * math.cos(angle))
            by = int(cy + (r + 11) * math.sin(angle))
            pygame.draw.circle(surface, (28, 32, 40), (bx, by), 5)
            pygame.draw.circle(surface, (126, 136, 148), (bx - 1, by - 1), 2)

        gloss = pygame.Surface((d, d), pygame.SRCALPHA)
        gloss_pts = [(r // 3, r // 5), (r, r // 7),
                     (r * 3 // 4, r // 2), (r // 5, r // 2)]
        pygame.draw.polygon(gloss, (255, 255, 255, 20), gloss_pts)
        gloss.blit(clip, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        surface.blit(gloss, (cx - r, cy - r))

        pf = pygame.font.SysFont("Consolas", 10, bold=True)
        lbl = pf.render("ADJ. GROWTH BAY", True, (82, 94, 110))
        surface.blit(lbl, (cx - lbl.get_width() // 2, cy + r + 18))

    def _draw_space_background(self, surface):
        """Draw a Mars sky viewport with stars and horizon."""
        w, h = self.window_size
        viewport_h = h // 3

        for y in range(viewport_h):
            t = y / max(viewport_h, 1)
            r = int(5 + t * 60)
            g = int(5 + t * 25)
            b = int(15 + t * 15)
            pygame.draw.line(surface, (r, g, b), (0, y), (w, y))

        # Distant planet glow near the horizon
        sun_x = int(w * 0.78)
        sun_y = int(viewport_h * 0.62)
        for radius, alpha in ((90, 24), (58, 46), (24, 90)):
            glow = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow, (255, 170, 110, alpha), (radius, radius), radius)
            surface.blit(glow, (sun_x - radius, sun_y - radius))

        # Tiny moons to avoid a flat sky
        pygame.draw.circle(surface, (155, 132, 118), (int(w * 0.18), int(viewport_h * 0.38)), 7)
        pygame.draw.circle(surface, (105, 96, 112), (int(w * 0.25), int(viewport_h * 0.22)), 4)

        for sx, sy, size, brightness, phase in self.stars:
            twinkle = int(brightness * (0.7 + 0.3 * math.sin(self.anim_time * 2 + phase)))
            twinkle = max(0, min(255, twinkle))
            pygame.draw.circle(surface, (twinkle, twinkle, twinkle),
                               (sx, sy), max(1, int(size)))

        if len(self.far_terrain_points) > 2:
            pygame.draw.polygon(surface, (84, 44, 34), self.far_terrain_points)
            pygame.draw.lines(surface, (124, 72, 48), False, self.far_terrain_points[:-2], 2)

        if len(self.mid_terrain_points) > 2:
            pygame.draw.polygon(surface, self.REGOLITH_DK, self.mid_terrain_points)
            pygame.draw.lines(surface, self.REGOLITH_MD, False, self.mid_terrain_points[:-2], 2)

        self._draw_exterior_outpost(surface)

        if len(self.terrain_points) > 2:
            pygame.draw.polygon(surface, self.REGOLITH_MD, self.terrain_points)
            pygame.draw.lines(surface, self.REGOLITH_LT, False, self.terrain_points[:-2], 2)

        # Thin dust haze over the exterior horizon
        haze = pygame.Surface((w, viewport_h), pygame.SRCALPHA)
        for y in range(viewport_h):
            alpha = int(28 * max(0.0, 1.0 - y / max(viewport_h, 1)))
            pygame.draw.line(haze, (214, 118, 64, alpha), (0, y), (w, y))
        surface.blit(haze, (0, 0))

    def _draw_exterior_outpost(self, surface):
        """Draw distant habitat modules, solar arrays and rocks on the Martian plain."""
        if len(self.exterior_walkway) > 1:
            pygame.draw.lines(surface, (150, 112, 82), False, self.exterior_walkway, 5)
            pygame.draw.lines(surface, (214, 156, 108), False, self.exterior_walkway, 1)

        for x, y, rx, ry in self.exterior_rocks:
            pygame.draw.ellipse(surface, (102, 60, 42), (x - rx, y - ry, rx * 2, ry * 2))
            pygame.draw.ellipse(surface, (172, 110, 70), (x - rx + 2, y - ry, rx, max(2, ry - 1)))

        for cx, cy, width, tilt in self.exterior_solar_arrays:
            panel = pygame.Rect(cx - width // 2, cy - 16, width, 16)
            pygame.draw.line(surface, (92, 102, 98), (cx, cy), (cx, cy + 18), 3)
            pts = [
                (panel.left, panel.bottom),
                (panel.left + 8, panel.top + max(0, tilt)),
                (panel.right, panel.top + max(0, tilt)),
                (panel.right - 8, panel.bottom),
            ]
            pygame.draw.polygon(surface, (70, 106, 140), pts)
            pygame.draw.polygon(surface, (166, 198, 216), pts, 2)

        for kind, cx, cy, width, height in self.exterior_modules:
            body = pygame.Rect(cx - width // 2, cy - height, width, height)
            if kind == "tower":
                pygame.draw.rect(surface, (188, 194, 188), body, border_radius=4)
                pygame.draw.rect(surface, (102, 112, 118), body, 2, border_radius=4)
                pygame.draw.rect(surface, (104, 214, 232), (body.left + 4, body.top + 6, body.width - 8, 9), border_radius=2)
                pygame.draw.line(surface, (188, 192, 196), (cx, body.top), (cx, body.top - 16), 2)
                pygame.draw.circle(surface, (255, 176, 98), (cx, body.top - 16), 3)
                continue

            fill = (208, 218, 208) if kind == "dome" else (176, 188, 174)
            pygame.draw.rect(surface, fill, body, border_radius=10)
            pygame.draw.rect(surface, (112, 126, 122), body, 2, border_radius=10)
            inner = body.inflate(-10, -10)
            glow = pygame.Surface((inner.width, inner.height), pygame.SRCALPHA)
            glow.fill((108, 194, 110, 46 if kind == "greenhouse" else 20))
            surface.blit(glow, inner.topleft)
            for gx in range(inner.left + 4, inner.right - 2, 14):
                pygame.draw.line(surface, (120, 144, 136), (gx, inner.top), (gx, inner.bottom), 1)
            if kind == "dome":
                pygame.draw.arc(surface, (236, 242, 236), body.inflate(0, 8), math.pi, math.tau, 2)

        for mx, my, mast_h in self.exterior_masts:
            pygame.draw.line(surface, (168, 174, 178), (mx, my), (mx, my - mast_h), 2)
            pygame.draw.line(surface, (168, 174, 178), (mx, my - mast_h), (mx - 8, my - mast_h + 12), 1)
            pygame.draw.line(surface, (168, 174, 178), (mx, my - mast_h), (mx + 8, my - mast_h + 12), 1)
            pygame.draw.circle(surface, (255, 184, 112), (mx, my - mast_h), 3)

        rx, ry = self.exterior_rover
        pygame.draw.rect(surface, (90, 100, 106), (rx - 13, ry - 6, 26, 10), border_radius=3)
        pygame.draw.rect(surface, (188, 194, 198), (rx - 7, ry - 12, 14, 6), border_radius=2)
        for wx in (-9, 9):
            pygame.draw.circle(surface, (38, 42, 46), (rx + wx, ry + 4), 4)
            pygame.draw.circle(surface, (136, 144, 148), (rx + wx, ry + 4), 2)

    def _draw_viewport_frame(self, surface):
        """Draw the upper greenhouse canopy frame framing the Martian exterior."""
        w, h = self.window_size
        viewport_h = h // 3

        arch = pygame.Surface((w, viewport_h + 34), pygame.SRCALPHA)
        dome_rect = pygame.Rect(18, -viewport_h // 2, w - 36, viewport_h * 2)
        pygame.draw.arc(arch, (54, 62, 70, 255), dome_rect, math.pi * 1.02, math.pi * 1.98, 18)
        pygame.draw.arc(arch, (164, 178, 170, 255), dome_rect.inflate(-12, -12), math.pi * 1.02, math.pi * 1.98, 4)
        for inset in (48, 104, 160):
            rect = dome_rect.inflate(-inset, -inset // 2)
            pygame.draw.arc(arch, (190, 212, 204, 42), rect, math.pi * 1.03, math.pi * 1.97, 2)
        surface.blit(arch, (0, 0))

        sill = pygame.Rect(0, viewport_h - 4, w, 14)
        pygame.draw.rect(surface, (74, 84, 82), sill)
        pygame.draw.line(surface, (188, 198, 194), (0, viewport_h - 2), (w, viewport_h - 2), 2)
        for x in range(24, w - 24, 48):
            pygame.draw.circle(surface, (60, 66, 70), (x, viewport_h + 2), 3)

    def _track_bounds_screen(self, track, pad_x=110, pad_y=90):
        pts = np.array([self.convert_pos_vr(p) for p in track.centerline], dtype=float)
        min_x = max(26, int(np.min(pts[:, 0]) - pad_x))
        max_x = min(self.window_size[0] - 26, int(np.max(pts[:, 0]) + pad_x))
        viewport_h = self.window_size[1] // 3
        min_y = max(viewport_h + 24, int(np.min(pts[:, 1]) - pad_y))
        max_y = min(self.window_size[1] - 44, int(np.max(pts[:, 1]) + pad_y))
        return pygame.Rect(min_x, min_y, max(40, max_x - min_x), max(40, max_y - min_y))

    def _draw_hull_bay(self, surface, track):
        """Draw a greenhouse repair gantry around the breach zone."""
        bay = self._track_bounds_screen(track, pad_x=125, pad_y=105)
        outer = bay.inflate(80, 72)
        aperture = bay.inflate(34, 24)
        overlay = pygame.Surface(self.window_size, pygame.SRCALPHA)
        pygame.draw.rect(overlay, (20, 28, 32, 20), aperture, border_radius=18)
        surface.blit(overlay, (0, 0))

        pylons = [
            pygame.Rect(outer.left + 8, outer.top + 28, 24, outer.height - 56),
            pygame.Rect(outer.right - 32, outer.top + 28, 24, outer.height - 56),
        ]
        for pylon in pylons:
            pygame.draw.rect(surface, (64, 72, 74), pylon, border_radius=8)
            pygame.draw.rect(surface, (188, 196, 194), pylon, 2, border_radius=8)
            for yy in range(pylon.top + 18, pylon.bottom - 12, 34):
                pygame.draw.circle(surface, self.HULL_ACCENT, (pylon.centerx, yy), 4)

        top_beam = pygame.Rect(outer.left + 18, outer.top + 6, outer.width - 36, 18)
        bottom_rail = pygame.Rect(outer.left + 28, outer.bottom - 22, outer.width - 56, 16)
        for rail, color in ((top_beam, (76, 84, 86)), (bottom_rail, (82, 90, 92))):
            pygame.draw.rect(surface, color, rail, border_radius=8)
            pygame.draw.rect(surface, (198, 204, 202), rail, 2, border_radius=8)

        cart = pygame.Rect(aperture.centerx - 58, top_beam.bottom - 2, 116, 18)
        pygame.draw.rect(surface, (104, 114, 116), cart, border_radius=8)
        pygame.draw.rect(surface, (214, 220, 218), cart, 2, border_radius=8)
        pygame.draw.line(surface, (200, 210, 212), (cart.centerx, cart.bottom), (cart.centerx, cart.bottom + 36), 2)

        grid = pygame.Surface(self.window_size, pygame.SRCALPHA)
        for x in range(aperture.left + 30, aperture.right - 18, 56):
            pygame.draw.line(grid, (192, 222, 218, 18), (x, aperture.top + 12), (x, aperture.bottom - 12), 1)
        for y in range(aperture.top + 26, aperture.bottom - 18, 48):
            pygame.draw.line(grid, (192, 222, 218, 18), (aperture.left + 12, y), (aperture.right - 12, y), 1)
        surface.blit(grid, (0, 0))

        stripe = pygame.Rect(outer.left + 42, top_beam.top - 2, min(180, outer.width - 84), 14)
        pygame.draw.rect(surface, self.HULL_STRIPE, stripe, border_radius=3)
        for x in range(stripe.left, stripe.right, 14):
            pygame.draw.line(surface, self.HULL_STRIPE_DK, (x, stripe.bottom), (x + 10, stripe.top), 4)

        mono = pygame.font.SysFont("Consolas", 11, bold=True)
        surface.blit(mono.render("SEALING FRAME", True, self.HUD_CYAN), (outer.left + 34, outer.bottom - 24))
        surface.blit(mono.render("GH-DOME 02", True, (214, 220, 224)), (outer.right - 98, outer.bottom - 24))
        surface.blit(mono.render("BREACH CONTAINMENT ACTIVE", True, (255, 176, 104)), (outer.left + 34, outer.top + 32))

    def _draw_wall_surface(self, surface):
        """Draw the greenhouse shell, glazing and crop beds behind the crack."""
        w, h = self.window_size
        viewport_h = h // 3

        for y in range(viewport_h, h):
            t = (y - viewport_h) / max(h - viewport_h, 1)
            col = (
                int(186 - 34 * t),
                int(206 - 28 * t),
                int(198 - 18 * t),
            )
            pygame.draw.line(surface, col, (0, y), (w, y))

        floor_poly = [
            (0, h),
            (w, h),
            (int(w * 0.82), int(h * 0.78)),
            (int(w * 0.18), int(h * 0.78)),
        ]
        pygame.draw.polygon(surface, (132, 128, 110), floor_poly)
        pygame.draw.polygon(surface, (182, 174, 148), floor_poly, 2)

        shell = pygame.Surface(self.window_size, pygame.SRCALPHA)
        for idx, inset in enumerate((24, 78, 132, 186)):
            rect = pygame.Rect(inset, viewport_h - 110 + idx * 16, w - inset * 2, h - viewport_h + 210 - idx * 16)
            pygame.draw.arc(shell, (198, 232, 220, 34 - idx * 5), rect, math.pi * 1.02, math.pi * 1.98, max(1, 4 - idx))
        for x in range(70, w - 70, 96):
            pygame.draw.line(shell, (222, 242, 234, 16), (x, viewport_h + 10), (x + 22, h - 40), 2)
        surface.blit(shell, (0, 0))

        for rect in self.wall_panels[::2]:
            glaze = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            glaze.fill((212, 236, 228, 36))
            surface.blit(glaze, rect.topleft)
            pygame.draw.rect(surface, (92, 116, 108), rect, 1, border_radius=6)

        self._draw_greenhouse_modules(surface)

        for x, y in self.wall_rivets[::3]:
            pygame.draw.circle(surface, (96, 92, 86), (x, y), 3)
            pygame.draw.circle(surface, (190, 180, 166), (x - 1, y - 1), 1)

        reflection = pygame.Surface(self.window_size, pygame.SRCALPHA)
        for y in range(viewport_h + 18, h, 5):
            alpha = max(0, int(20 * (1.0 - (y - viewport_h) / max(h - viewport_h, 1))))
            pygame.draw.line(reflection, (255, 255, 255, alpha), (24, y), (w - 24, y))
        for sx, sy, length, alpha in self.wall_streaks[:26]:
            pygame.draw.line(reflection, (230, 236, 240, alpha // 2), (sx, sy), (sx + 10, sy + length), 1)
        surface.blit(reflection, (0, 0))

        warning_font = pygame.font.SysFont("Consolas", 10, bold=True)
        for plate in self.warning_plates[:3]:
            pygame.draw.rect(surface, self.HULL_STRIPE, plate, border_radius=2)
            surface.blit(warning_font.render("SEAL ZONE", True, (240, 236, 224)), (plate.x + 8, plate.y + 3))

        vignette = pygame.Surface(self.window_size, pygame.SRCALPHA)
        pygame.draw.rect(vignette, (0, 0, 0, 26), (0, viewport_h, w, h - viewport_h))
        surface.blit(vignette, (0, 0))

    def _draw_greenhouse_modules(self, surface):
        """Draw hydroponic racks, grow lights and support hardware behind the damaged shell."""
        viewport_h = self.window_size[1] // 3
        for p0, p1 in self.feed_pipes:
            pygame.draw.line(surface, (118, 146, 138), p0, p1, 6)
            pygame.draw.line(surface, (226, 236, 230), (p0[0], p0[1] - 1), (p1[0], p1[1] - 1), 2)

        for tank in self.service_tanks:
            pygame.draw.rect(surface, (198, 204, 196), tank, border_radius=14)
            pygame.draw.rect(surface, (118, 132, 128), tank, 2, border_radius=14)
            pygame.draw.ellipse(surface, (154, 166, 160), tank.inflate(-12, -16), 2)

        side_walls = [pygame.Rect(36, viewport_h + 60, 156, 328), pygame.Rect(self.window_size[0] - 192, viewport_h + 60, 156, 328)]
        glow = pygame.Surface(self.window_size, pygame.SRCALPHA)
        for wall in side_walls:
            pygame.draw.rect(surface, (70, 88, 80), wall, border_radius=12)
            pygame.draw.rect(surface, (188, 202, 194), wall, 2, border_radius=12)
            for row in range(7):
                y = wall.top + 24 + row * 40
                pygame.draw.line(surface, (255, 232, 184), (wall.left + 12, y - 10), (wall.right - 12, y - 10), 2)
                for col in range(5):
                    cx = wall.left + 20 + col * 26 + (row % 2) * 2
                    pygame.draw.ellipse(surface, (96, 182, 92), (cx - 12, y - 4, 24, 12))
                    pygame.draw.ellipse(surface, (132, 210, 118), (cx - 8, y - 8, 18, 10))
            pygame.draw.rect(glow, (124, 220, 130, 18), wall.inflate(10, 10), border_radius=16)
        surface.blit(glow, (0, 0))

        catwalk = pygame.Rect(self.window_size[0] // 2 - 118, self.window_size[1] - 116, 236, 44)
        pygame.draw.rect(surface, (106, 110, 94), catwalk, border_radius=8)
        pygame.draw.rect(surface, (192, 188, 164), catwalk, 2, border_radius=8)
        for x in range(catwalk.left + 18, catwalk.right - 12, 22):
            pygame.draw.line(surface, (156, 150, 126), (x, catwalk.top + 6), (x, catwalk.bottom - 6), 2)

        bin_glow = pygame.Surface(self.window_size, pygame.SRCALPHA)
        for bin_rect in self.planter_bins:
            pygame.draw.rect(surface, (88, 98, 74), bin_rect, border_radius=6)
            pygame.draw.rect(surface, (162, 170, 124), bin_rect, 2, border_radius=6)
            soil = bin_rect.inflate(-8, -10)
            pygame.draw.rect(surface, (52, 62, 38), soil, border_radius=4)
            for x in range(soil.left + 8, soil.right - 8, 12):
                stem_h = 12 + ((x + soil.y) % 14)
                pygame.draw.line(surface, (88, 176, 84), (x, soil.bottom - 4), (x, soil.bottom - stem_h), 2)
                pygame.draw.ellipse(surface, (118, 202, 104), (x - 6, soil.bottom - stem_h - 4, 12, 8))
            pygame.draw.rect(bin_glow, (128, 214, 126, 14), bin_rect.inflate(10, 8), border_radius=10)
        surface.blit(bin_glow, (0, 0))

    def _spawn_leak_particles(self, track):
        """Emit particles from exposed crack sections to suggest atmosphere loss."""
        if len(self.leak_particles) >= self.max_leak_particles:
            return

        count = min(4, self.max_leak_particles - len(self.leak_particles))
        for _ in range(count):
            idx = random.randint(6, max(6, track.n_pts - 7))
            pt = track.centerline[idx]
            normal = track.normals[idx]
            sx, sy = self.convert_pos_vr(pt)
            outward = normal if random.random() > 0.5 else -normal
            speed = random.uniform(16.0, 52.0)
            vx = outward[0] * speed * 0.65 + random.uniform(-12.0, 12.0)
            vy = -abs(outward[1] * speed) - random.uniform(8.0, 30.0)
            color = self.LEAK_COLOR if random.random() > 0.18 else self.LEAK_HOT
            self.leak_particles.append(
                LeakParticle(
                    sx,
                    sy,
                    vx,
                    vy,
                    random.uniform(0.45, 1.35),
                    random.randint(1, 3),
                    color,
                )
            )

    def _update_leak_particles(self, dt):
        alive = []
        for particle in self.leak_particles:
            particle.x += particle.vx * dt
            particle.y += particle.vy * dt
            particle.vx *= 0.995
            particle.vy -= 3.8 * dt
            particle.life -= dt
            if particle.life > 0:
                alive.append(particle)
        self.leak_particles = alive

    def _draw_leak_particles(self, surface):
        for particle in self.leak_particles:
            alpha = max(0, min(255, int(255 * particle.life / max(particle.max_life, 1e-6))))
            radius = particle.size * 2
            sprite = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(sprite, (*particle.color, alpha), (radius, radius), radius)
            surface.blit(sprite, (int(particle.x) - radius, int(particle.y) - radius))

    def _draw_dust_overlay(self, surface, dt):
        """Floating internal dust enhances scale and depth around the work area."""
        dust_layer = pygame.Surface(self.window_size, pygame.SRCALPHA)
        w, h = self.window_size
        for mote in self.dust_motes:
            mote[0] = (mote[0] + mote[3] * dt) % (w + 20) - 10
            mote[1] += mote[4] * dt
            if mote[1] < h // 3 + 12:
                mote[1] = h - 12
            elif mote[1] > h - 10:
                mote[1] = h // 3 + 18
            alpha = int(38 + 42 * (0.5 + 0.5 * math.sin(self.anim_time + mote[5])))
            pygame.draw.circle(
                dust_layer,
                (220, 208, 190, alpha),
                (int(mote[0]), int(mote[1])),
                int(mote[2]),
            )
        surface.blit(dust_layer, (0, 0))

    def _draw_mars_track(self, surface, track, highlight_wall=None):
        """Render the path as a fractured breach in the greenhouse shell."""
        center_pts = [self.convert_pos_vr(p) for p in track.centerline]
        left_pts = [self.convert_pos_vr(p) for p in track.wall_left]
        right_pts = [self.convert_pos_vr(p) for p in track.wall_right]
        half_width_px = np.maximum(2, np.round(track.half_widths * self.window_scale).astype(int))
        lip_left = []
        lip_right = []
        fracture_left = []
        fracture_right = []
        scorch_left = []
        scorch_right = []
        for i, (pt, n, width) in enumerate(zip(track.centerline, track.normals, track.half_widths)):
            wiggle = 0.55 * math.sin(i * 0.41 + 0.7) + 0.32 * math.sin(i * 0.17 + 1.9)
            lip_offset = 0.00055 + 0.00022 * wiggle
            fracture_offset = max(width * 0.24, 0.00085) + 0.00030 * wiggle
            scorch_offset = width + 0.0018 + 0.00055 * wiggle
            lip_left.append(self.convert_pos_vr(pt + n * lip_offset))
            lip_right.append(self.convert_pos_vr(pt - n * lip_offset))
            fracture_left.append(self.convert_pos_vr(pt + n * fracture_offset))
            fracture_right.append(self.convert_pos_vr(pt - n * fracture_offset))
            scorch_left.append(self.convert_pos_vr(pt + n * scorch_offset))
            scorch_right.append(self.convert_pos_vr(pt - n * scorch_offset))

        scorch = pygame.Surface(self.window_size, pygame.SRCALPHA)
        scorch_poly = [(int(p[0]), int(p[1])) for p in scorch_left] + [(int(p[0]), int(p[1])) for p in reversed(scorch_right)]
        if len(scorch_poly) > 5:
            pygame.draw.polygon(scorch, (96, 64, 48, 38), scorch_poly)
            pygame.draw.polygon(scorch, (164, 110, 82, 16), scorch_poly, 2)
        surface.blit(scorch, (0, 0))

        shadow = pygame.Surface(self.window_size, pygame.SRCALPHA)
        for i in range(len(center_pts) - 1):
            p1 = (int(center_pts[i][0] + 3), int(center_pts[i][1] + 3))
            p2 = (int(center_pts[i + 1][0] + 3), int(center_pts[i + 1][1] + 3))
            seg_half_width = 0.5 * (half_width_px[i] + half_width_px[i + 1])
            pygame.draw.line(shadow, (12, 8, 8, 64), p1, p2,
                             max(4, int(seg_half_width * 1.8)))
        surface.blit(shadow, (0, 0))

        dust = pygame.Surface(self.window_size, pygame.SRCALPHA)
        if len(center_pts) > 1:
            for i in range(len(center_pts) - 1):
                p1 = (int(center_pts[i][0]), int(center_pts[i][1]))
                p2 = (int(center_pts[i + 1][0]), int(center_pts[i + 1][1]))
                seg_half_width = 0.5 * (half_width_px[i] + half_width_px[i + 1])
                haze_width = max(4, int(seg_half_width * 2.8))
                pygame.draw.line(dust, (164, 124, 92, 18), p1, p2, haze_width)
        surface.blit(dust, (0, 0))

        if len(fracture_left) > 1:
            pygame.draw.lines(surface, (82, 54, 40), False, [(int(p[0]), int(p[1])) for p in fracture_left], 2)
            pygame.draw.lines(surface, (218, 224, 214), False, [(int(p[0]), int(p[1])) for p in lip_left], 2)
        if len(fracture_right) > 1:
            pygame.draw.lines(surface, (82, 54, 40), False, [(int(p[0]), int(p[1])) for p in fracture_right], 2)
            pygame.draw.lines(surface, (218, 224, 214), False, [(int(p[0]), int(p[1])) for p in lip_right], 2)

        for i in range(len(center_pts) - 1):
            p1 = (int(center_pts[i][0]), int(center_pts[i][1]))
            p2 = (int(center_pts[i + 1][0]), int(center_pts[i + 1][1]))
            seg_half_width = 0.5 * (half_width_px[i] + half_width_px[i + 1])
            fissure_w = max(2, int(seg_half_width * 0.62))
            pygame.draw.line(surface, (18, 10, 10), p1, p2, fissure_w + 2)
            pygame.draw.line(surface, (34, 16, 14), p1, p2, fissure_w)
            if i % 7 == 0:
                hot_alpha = 55 + int(25 * (0.5 + 0.5 * math.sin(self.anim_time * 2.6 + i * 0.3)))
                ember = pygame.Surface(self.window_size, pygame.SRCALPHA)
                pygame.draw.line(ember, (214, 112, 78, hot_alpha), p1, p2, 1)
                surface.blit(ember, (0, 0))

            if i % 18 == 0 and i < len(track.normals):
                base = np.array(center_pts[i], dtype=float)
                tangent = np.array(track.centerline[min(i + 1, track.n_pts - 1)]) - np.array(track.centerline[max(i - 1, 0)])
                if np.linalg.norm(tangent) > 1e-8:
                    tangent = tangent / np.linalg.norm(tangent)
                    tangent_s = np.array([-tangent[0], tangent[1]])
                else:
                    tangent_s = np.array([0.0, 1.0])
                normal = np.array(track.normals[i])
                branch_dir = normal * (1 if (i // 26) % 2 == 0 else -1) + 0.35 * tangent_s
                if np.linalg.norm(branch_dir) > 1e-8:
                    branch_dir = branch_dir / np.linalg.norm(branch_dir)
                branch_len = 9 + (i % 4) * 5
                end = base + branch_dir * branch_len
                pygame.draw.line(surface, (72, 42, 34), base.astype(int), end.astype(int), 2)
                pygame.draw.line(surface, (202, 194, 182), base.astype(int), end.astype(int), 1)
                if i % 36 == 0:
                    spur = end + branch_dir * (4 + (i % 3) * 2)
                    pygame.draw.line(surface, (68, 38, 30), end.astype(int), spur.astype(int), 1)

        left_color = (255, 62, 62) if highlight_wall == 'left' else (98, 68, 54)
        right_color = (255, 62, 62) if highlight_wall == 'right' else (98, 68, 54)
        if highlight_wall in ('left', 'right'):
            highlight_surf = pygame.Surface(self.window_size, pygame.SRCALPHA)
            wall_pts = fracture_left if highlight_wall == 'left' else fracture_right
            pts = [(int(p[0]), int(p[1])) for p in wall_pts]
            if len(pts) > 1:
                pygame.draw.lines(highlight_surf, (255, 84, 84, 76), False, pts, 8)
                pygame.draw.lines(highlight_surf, (255, 136, 136, 120), False, pts, 3)
            surface.blit(highlight_surf, (0, 0))

        if len(fracture_left) > 1:
            pts = [(int(p[0]), int(p[1])) for p in fracture_left]
            pygame.draw.lines(surface, left_color, False, pts, 2)
        if len(fracture_right) > 1:
            pts = [(int(p[0]), int(p[1])) for p in fracture_right]
            pygame.draw.lines(surface, right_color, False, pts, 2)

        start_s = self.convert_pos_vr(track.start)
        end_s = self.convert_pos_vr(track.end)

        for r in [14, 10, 6]:
            alpha = int(80 * (14 - r) / 8)
            s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (0, 255, 100, alpha + 60), (r, r), r)
            surface.blit(s, (int(start_s[0]) - r, int(start_s[1]) - r))
        pygame.draw.circle(surface, (0, 255, 120), (int(start_s[0]), int(start_s[1])), 5)

        for r in [14, 10, 6]:
            alpha = int(80 * (14 - r) / 8)
            s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (255, 60, 60, alpha + 60), (r, r), r)
            surface.blit(s, (int(end_s[0]) - r, int(end_s[1]) - r))
        pygame.draw.circle(surface, (255, 80, 80), (int(end_s[0]), int(end_s[1])), 5)

        label_font = pygame.font.SysFont("Consolas", 11, bold=True)
        surface.blit(label_font.render("START", True, (0, 255, 120)),
                     (int(start_s[0]) + 14, int(start_s[1]) - 5))
        surface.blit(label_font.render("END", True, (255, 80, 80)),
                     (int(end_s[0]) + 14, int(end_s[1]) - 5))

    def _draw_work_lamp(self, surface, cursor_phys):
        """Local task lighting centered on the operator tool."""
        if cursor_phys is None:
            return
        cursor = self.convert_pos_vr(cursor_phys)
        x, y = int(cursor[0]), int(cursor[1])
        lamp = pygame.Surface(self.window_size, pygame.SRCALPHA)
        for radius, alpha in ((168, 8), (118, 14), (72, 24)):
            pygame.draw.circle(lamp, (232, 242, 255, alpha), (x, y), radius)
        # Slight elongated reflection on the metal bay
        pygame.draw.ellipse(lamp, (214, 234, 255, 18), (x - 82, y + 18, 164, 62))
        pygame.draw.ellipse(lamp, (255, 190, 138, 20), (x - 56, y + 12, 112, 34))
        surface.blit(lamp, (0, 0))

    def _draw_mission_hud(self, surface, state_name, n_demos, crack_name, progress, elapsed):
        """Draw a greenhouse mission HUD overlay on the VR panel."""
        w = self.window_size[0]
        hud_font = pygame.font.SysFont("Consolas", 13)
        title_font = pygame.font.SysFont("Consolas", 16, bold=True)

        bar_h = 28
        bar_surf = pygame.Surface((w, bar_h), pygame.SRCALPHA)
        bar_surf.fill((0, 0, 0, 180))
        surface.blit(bar_surf, (0, 0))

        title = title_font.render("MARS GREENHOUSE — PRESSURE SHELL SEALING", True, self.HUD_ORANGE)
        surface.blit(title, (10, 5))

        status_text = f"[{state_name}]  Crack: {crack_name}  Demos: {n_demos}"
        status = hud_font.render(status_text, True, (180, 180, 180))
        surface.blit(status, (w - status.get_width() - 10, 7))

        # Side diagnostics stack
        diag = pygame.Surface((180, 82), pygame.SRCALPHA)
        diag.fill((0, 0, 0, 120))
        surface.blit(diag, (w - 194, 48))
        metrics = [
            ("Habitat pressure", f"{int(100 * progress):02d}%"),
            ("Leak severity", f"{int(100 * (1.0 - progress)):02d}%"),
            ("Crop bay climate", "STABLE" if progress > 0.35 else "AT RISK"),
        ]
        for i, (label, value) in enumerate(metrics):
            ly = 56 + i * 22
            surface.blit(hud_font.render(label, True, (142, 148, 154)), (w - 184, ly))
            color = self.HUD_CYAN if i < 2 else ((130, 255, 160) if progress > 0.35 else (255, 108, 82))
            surface.blit(hud_font.render(value, True, color), (w - 74, ly))

        bottom_y = self.window_size[1] - 30
        bot_surf = pygame.Surface((w, 30), pygame.SRCALPHA)
        bot_surf.fill((0, 0, 0, 160))
        surface.blit(bot_surf, (0, bottom_y))

        info_parts = [
            f"Progress: {progress * 100:.0f}%",
            f"Mission: {progress * 100:.0f}%",
            f"Time: {elapsed:.1f}s",
        ]
        info_text = hud_font.render("   |   ".join(info_parts), True, (200, 200, 200))
        surface.blit(info_text, (10, bottom_y + 7))

    def _draw_seal_progress_bar(self, surface, seal_pct, x, y, w_bar=200, h_bar=16):
        """Draw the Mars mission progress bar."""
        pygame.draw.rect(surface, (24, 27, 33), (x - 4, y - 4, w_bar + 8, h_bar + 8), border_radius=5)
        pygame.draw.rect(surface, (30, 30, 40), (x, y, w_bar, h_bar), border_radius=3)

        fill_w = int(w_bar * seal_pct)
        if fill_w > 0:
            r = int(40 * (1 - seal_pct))
            g = int(100 + 155 * seal_pct)
            b = int(200 * (1 - seal_pct * 0.5))
            pygame.draw.rect(surface, (r, g, b), (x, y, fill_w, h_bar), border_radius=3)

        pygame.draw.rect(surface, (100, 100, 120), (x, y, w_bar, h_bar), 1, border_radius=3)

        bar_font = pygame.font.SysFont("Consolas", 11)
        label = bar_font.render(f"SEAL: {seal_pct * 100:.0f}%", True, (220, 220, 220))
        surface.blit(label, (x + w_bar // 2 - label.get_width() // 2, y + 1))

    def draw_mars_vr_scene(self, track, state_name, n_demos, track_name,
                           progress, elapsed, highlight_wall=None, cursor_phys=None):
        """Render the Mars greenhouse crack environment on the VR panel."""
        dt = 1.0 / max(self.FPS, 1)
        self.anim_time += dt
        self._draw_space_background(self.screenVR)
        self._draw_viewport_frame(self.screenVR)
        self._draw_wall_surface(self.screenVR)
        self._draw_porthole(self.screenVR)
        self._draw_hull_bay(self.screenVR, track)
        self._draw_mars_track(self.screenVR, track, highlight_wall=highlight_wall)
        self._spawn_leak_particles(track)
        self._update_leak_particles(dt)
        self._draw_leak_particles(self.screenVR)
        self._draw_dust_overlay(self.screenVR, dt)
        self._draw_work_lamp(self.screenVR, cursor_phys)
        self._draw_seal_progress_bar(
            self.screenVR, progress, self.window_size[0] - 220, self.window_size[1] // 3 + 40
        )
        self._draw_mission_hud(
            self.screenVR, state_name, n_demos, track_name, progress, elapsed
        )

    def erase_screen(self):
        self.screenHaptics.fill(self.cWhite) #erase the haptics surface
        self.screenVR.fill((0, 0, 0)) #erase the VR surface
        self.debug_text = ""
    
    def render(self,pA0,pB0,pA,pB,pE,f,pM):
        ###################Render the Haptic Surface###################
        #set new position of items indicating the endpoint location
        self.haptic.center = pE #the hhandle image and effort square will also use this position for drawing
        self.effort_cursor.center = self.haptic.center

        if self.device_connected:
            self.effort_color = (255,255,255)

        #pygame.draw.rect(self.screenHaptics, self.effort_color, self.haptic,border_radius=4)
        pygame.draw.rect(self.screenHaptics, self.effort_color, self.effort_cursor,border_radius=8)

        ######### Robot visualization ###################
        if self.show_linkages:
            pantographColor = (150,150,150)
            pygame.draw.lines(self.screenHaptics, pantographColor, False,[pA0,pA],15)
            pygame.draw.lines(self.screenHaptics, pantographColor, False,[pB0,pB],15)
            pygame.draw.lines(self.screenHaptics, pantographColor, False,[pA,pE],15)
            pygame.draw.lines(self.screenHaptics, pantographColor, False,[pB,pE],15)
            
            for p in ( pA0,pB0,pA,pB,pE):
                pygame.draw.circle(self.screenHaptics, (0, 0, 0),p, 15)
                pygame.draw.circle(self.screenHaptics, (200, 200, 200),p, 6)
        
        ### Hand visualisation
        self.screenHaptics.blit(self.hhandle,self.effort_cursor)
        
        #pygame.draw.line(self.screenHaptics, (0, 0, 0), (self.haptic.center),(self.haptic.center+2*k*(xm-xh)))
        
        ###################Render the VR surface###################
        if not self.hide_haptic_vr:
            pygame.draw.rect(self.screenVR, self.colorHaptic, self.haptic, border_radius=8)
        
        if not self.device_connected:
            pygame.draw.lines(self.screenHaptics, (0,0,0), False,[self.effort_cursor.center,pM],2)
        ##Fuse it back together
        self.window.blit(self.screenHaptics, (0,0))
        self.window.blit(self.screenVR, (self.window_size[0],0))

        ##Print status in  overlay
        if self.show_debug:    
            self.debug_text += "FPS = " + str(round(self.clock.get_fps()))+" "
            self.debug_text += "fe: "+str(np.round(f[0],1))+","+str(np.round(f[1],1))+"] "
            self.debug_text += "xh: ["+str(np.round(pE[0],1))+","+str(np.round(pE[1],1))+"]"
            self.text = self.font.render(self.debug_text, True, (0, 0, 0), (255, 255, 255))
            self.window.blit(self.text, self.textRect)

        pygame.display.flip()    
        ##Slow down the loop to match FPS
        self.clock.tick(self.FPS)

    def close(self):
        pygame.display.quit()
        pygame.quit()
