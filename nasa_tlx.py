# -*- coding: utf-8 -*-
"""
PA3 — NASA-TLX Subjective Workload Questionnaire
-------------------------------------------------
Pygame-based NASA-TLX implementation that pops up after each trial.
Returns a dict with the six raw TLX subscales (0-100) and the raw
overall score (unweighted average).
"""

import pygame
import numpy as np

TLX_SCALES = [
    ("Mental Demand",
     "How mentally demanding was the task?",
     "Very Low", "Very High"),
    ("Physical Demand",
     "How physically demanding was the task?",
     "Very Low", "Very High"),
    ("Temporal Demand",
     "How hurried or rushed was the pace of the task?",
     "Very Low", "Very High"),
    ("Performance",
     "How successful were you in accomplishing what you were asked to do?",
     "Perfect", "Failure"),
    ("Effort",
     "How hard did you have to work to accomplish your level of performance?",
     "Very Low", "Very High"),
    ("Frustration",
     "How insecure, discouraged, irritated, stressed, and annoyed were you?",
     "Very Low", "Very High"),
]


def run_nasa_tlx(screen_size=(700, 500)):
    """
    Show a NASA-TLX dialog and return the ratings.

    Returns
    -------
    dict  with keys: mental, physical, temporal, performance, effort,
                     frustration, overall
    """
    pygame.init()
    win = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("NASA-TLX Questionnaire")
    font_title = pygame.font.SysFont("Arial", 22, bold=True)
    font_label = pygame.font.SysFont("Arial", 16)
    font_small = pygame.font.SysFont("Arial", 13)
    font_btn   = pygame.font.SysFont("Arial", 18, bold=True)

    W, H = screen_size
    values = [50] * 6  # default midpoint
    dragging = -1      # which slider is being dragged

    slider_x = 180
    slider_w = W - 260
    slider_y0 = 80
    slider_dy = 62

    running = True
    result = None

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
                result = None
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                mx, my = ev.pos
                # check sliders
                for i in range(6):
                    sy = slider_y0 + i * slider_dy
                    if slider_x - 10 <= mx <= slider_x + slider_w + 10 and sy - 10 <= my <= sy + 20:
                        dragging = i
                        values[i] = int(np.clip((mx - slider_x) / slider_w * 100, 0, 100))
                # check submit button
                btn_rect = pygame.Rect(W // 2 - 80, H - 65, 160, 42)
                if btn_rect.collidepoint(mx, my):
                    keys = ["mental", "physical", "temporal",
                            "performance", "effort", "frustration"]
                    result = {k: v for k, v in zip(keys, values)}
                    result["overall"] = float(np.mean(values))
                    running = False
            elif ev.type == pygame.MOUSEBUTTONUP:
                dragging = -1
            elif ev.type == pygame.MOUSEMOTION and dragging >= 0:
                mx = ev.pos[0]
                values[dragging] = int(np.clip((mx - slider_x) / slider_w * 100, 0, 100))

        # ── draw ────────────────────────────────────────────
        win.fill((245, 245, 250))

        # title
        title_surf = font_title.render("NASA-TLX — Rate your experience", True, (30, 30, 80))
        win.blit(title_surf, (W // 2 - title_surf.get_width() // 2, 20))

        for i, (name, desc, lo, hi) in enumerate(TLX_SCALES):
            sy = slider_y0 + i * slider_dy
            # label
            lbl = font_label.render(name, True, (20, 20, 20))
            win.blit(lbl, (10, sy - 5))
            # track
            pygame.draw.rect(win, (200, 200, 210),
                             (slider_x, sy + 5, slider_w, 6), border_radius=3)
            # filled portion
            fill_w = int(values[i] / 100 * slider_w)
            pygame.draw.rect(win, (60, 120, 220),
                             (slider_x, sy + 5, fill_w, 6), border_radius=3)
            # thumb
            tx = slider_x + fill_w
            pygame.draw.circle(win, (40, 90, 200), (tx, sy + 8), 10)
            pygame.draw.circle(win, (255, 255, 255), (tx, sy + 8), 6)
            # value
            val_surf = font_small.render(str(values[i]), True, (60, 60, 60))
            win.blit(val_surf, (slider_x + slider_w + 15, sy))
            # lo/hi labels
            lo_surf = font_small.render(lo, True, (130, 130, 130))
            hi_surf = font_small.render(hi, True, (130, 130, 130))
            win.blit(lo_surf, (slider_x, sy + 18))
            win.blit(hi_surf, (slider_x + slider_w - hi_surf.get_width(), sy + 18))

        # submit button
        btn_rect = pygame.Rect(W // 2 - 80, H - 65, 160, 42)
        pygame.draw.rect(win, (40, 100, 200), btn_rect, border_radius=8)
        btn_text = font_btn.render("Submit", True, (255, 255, 255))
        win.blit(btn_text, (btn_rect.centerx - btn_text.get_width() // 2,
                            btn_rect.centery - btn_text.get_height() // 2))

        pygame.display.flip()

    pygame.display.quit()
    return result
