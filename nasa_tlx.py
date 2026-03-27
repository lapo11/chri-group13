# -*- coding: utf-8 -*-
"""
nasa_tlx.py — NASA Task Load Index questionnaire (pygame-based)
================================================================
Presents the 6 NASA-TLX subscales as slider-based questions.
Returns a dict with raw scores (0–100) and the overall score.
"""

import pygame
import sys


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


def run_nasa_tlx():
    """
    Run the NASA-TLX questionnaire in a pygame window.
    Returns dict with scale names as keys and 0–100 values, plus 'overall'.
    """
    pygame.init()
    W, H = 700, 550
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("NASA-TLX Questionnaire")
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont("Arial", 22, bold=True)
    font_desc  = pygame.font.SysFont("Arial", 15)
    font_label = pygame.font.SysFont("Arial", 13)
    font_btn   = pygame.font.SysFont("Arial", 18, bold=True)

    values = [50] * len(TLX_SCALES)
    dragging = -1
    submitted = False

    # Layout
    slider_x = 120
    slider_w = 460
    start_y = 80
    row_h = 70

    while not submitted:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                return None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                for i in range(len(TLX_SCALES)):
                    sy = start_y + i * row_h + 30
                    if slider_x - 10 <= mx <= slider_x + slider_w + 10 and sy - 15 <= my <= sy + 15:
                        dragging = i
                # Check submit button
                btn_rect = pygame.Rect(W // 2 - 80, start_y + len(TLX_SCALES) * row_h + 20, 160, 44)
                if btn_rect.collidepoint(mx, my):
                    submitted = True
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = -1
            elif event.type == pygame.MOUSEMOTION and dragging >= 0:
                mx = event.pos[0]
                frac = (mx - slider_x) / slider_w
                values[dragging] = int(max(0, min(100, frac * 100)))

        screen.fill((25, 25, 35))

        # Title
        title = font_title.render("NASA Task Load Index", True, (220, 220, 220))
        screen.blit(title, (W // 2 - title.get_width() // 2, 20))

        for i, (name, desc, lo, hi) in enumerate(TLX_SCALES):
            y = start_y + i * row_h

            # Label
            lbl = font_desc.render(f"{name}: {desc}", True, (200, 200, 200))
            screen.blit(lbl, (slider_x, y))

            # Slider track
            sy = y + 30
            pygame.draw.line(screen, (80, 80, 80), (slider_x, sy), (slider_x + slider_w, sy), 3)

            # Slider handle
            hx = slider_x + int(values[i] / 100.0 * slider_w)
            color = (230, 120, 50) if dragging == i else (180, 180, 200)
            pygame.draw.circle(screen, color, (hx, sy), 10)

            # Low/high labels
            lo_surf = font_label.render(lo, True, (140, 140, 140))
            hi_surf = font_label.render(hi, True, (140, 140, 140))
            screen.blit(lo_surf, (slider_x - lo_surf.get_width() - 8, sy - 8))
            screen.blit(hi_surf, (slider_x + slider_w + 8, sy - 8))

            # Value
            val_surf = font_label.render(str(values[i]), True, (255, 200, 100))
            screen.blit(val_surf, (hx - val_surf.get_width() // 2, sy - 22))

        # Submit button
        btn_rect = pygame.Rect(W // 2 - 80, start_y + len(TLX_SCALES) * row_h + 20, 160, 44)
        pygame.draw.rect(screen, (50, 140, 80), btn_rect, border_radius=8)
        btn_text = font_btn.render("Submit", True, (255, 255, 255))
        screen.blit(btn_text, (btn_rect.centerx - btn_text.get_width() // 2,
                               btn_rect.centery - btn_text.get_height() // 2))

        pygame.display.flip()
        clock.tick(30)

    pygame.display.quit()

    result = {}
    for i, (name, _, _, _) in enumerate(TLX_SCALES):
        result[name] = values[i]
    result['overall'] = sum(values) / len(values)
    return result
