# -*- coding: utf-8 -*-
"""
PA3 — Tube Target
------------------
Generates a curved tube defined by:
  - centerline: a smooth parametric curve (Nx2 in physical coords, meters)
  - half_width: tube radius in meters
  - wall_left / wall_right: offset curves at ±half_width from centerline

The task is to navigate from the start to the end of the tube without
touching the walls.

Physical coordinate frame (Haply): +x right, +y down on screen.
All shapes are centered around cy ≈ 0.06 m (workspace center).
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────

# Narrower than the original template, but compressed to stay reachable with
# the physical Haply workspace.
DEFAULT_HALF_WIDTH = 0.007
DEFAULT_CX = 0.0
DEFAULT_CY = 0.061
REACHABILITY_SCALE = 0.90


def _bbox_center(points):
    pts = np.asarray(points, dtype=float)
    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    return 0.5 * (mins + maxs)


def _compress_about_center(centerline, sx=1.0, sy=1.0):
    """Scale a curve about its centroid to keep it inside the haptic workspace."""
    pts = np.asarray(centerline, dtype=float)
    pivot = pts.mean(axis=0)
    scaled = pts.copy()
    scaled[:, 0] = pivot[0] + sx * (scaled[:, 0] - pivot[0])
    scaled[:, 1] = pivot[1] + sy * (scaled[:, 1] - pivot[1])
    return scaled

def _normals(centerline):
    """Compute unit normals (pointing left) at each point of the centerline."""
    # Tangent via central differences
    tangents = np.zeros_like(centerline)
    tangents[1:-1] = centerline[2:] - centerline[:-2]
    tangents[0] = centerline[1] - centerline[0]
    tangents[-1] = centerline[-1] - centerline[-2]
    # Normalise
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    lengths = np.clip(lengths, 1e-12, None)
    tangents = tangents / lengths
    # Rotate 90° CCW to get left-pointing normal: (tx, ty) → (-ty, tx)
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    return normals


def _smooth(pts, sigma=3):
    """Simple Gaussian smoothing on a 2D curve."""
    from scipy.ndimage import gaussian_filter1d
    return np.column_stack([
        gaussian_filter1d(pts[:, 0], sigma),
        gaussian_filter1d(pts[:, 1], sigma),
    ])


def _normalize_target_layout(centerline, widths=None, target_x=DEFAULT_CX, target_y=DEFAULT_CY):
    """
    Recenter the crack by its visual bounding box and keep its traversal order
    consistent: start should be the upper endpoint, falling back to the leftmost
    endpoint when both endpoints share nearly the same height.
    """
    pts = np.asarray(centerline, dtype=float).copy()
    pts += np.array([target_x, target_y], dtype=float) - _bbox_center(pts)

    width_profile = None if widths is None else np.asarray(widths, dtype=float).copy()
    start = pts[0]
    end = pts[-1]
    if (end[1] < start[1] - 1e-4) or (abs(end[1] - start[1]) <= 1e-4 and end[0] < start[0]):
        pts = pts[::-1].copy()
        if width_profile is not None:
            width_profile = width_profile[::-1].copy()

    if width_profile is None:
        return pts
    return pts, width_profile


class Tube:
    """A curved tube target."""

    def __init__(self, centerline, half_width):
        """
        Parameters
        ----------
        centerline : ndarray (N, 2) — physical coords (meters)
        half_width : float or ndarray (N,) — tube radius in meters
        """
        self.centerline = np.asarray(centerline, dtype=float)
        self.n_pts = len(self.centerline)
        if np.isscalar(half_width):
            self.half_widths = np.full(self.n_pts, float(half_width), dtype=float)
        else:
            widths = np.asarray(half_width, dtype=float).reshape(-1)
            if len(widths) != self.n_pts:
                raise ValueError("half_width profile must match centerline length")
            self.half_widths = widths.copy()
        self.half_width = float(np.mean(self.half_widths))
        self.half_width_min = float(np.min(self.half_widths))
        self.half_width_max = float(np.max(self.half_widths))
        self.normals = _normals(self.centerline)
        offsets = self.normals * self.half_widths[:, None]
        self.wall_left = self.centerline + offsets
        self.wall_right = self.centerline - offsets
        self.start = self.centerline[0].copy()
        self.end = self.centerline[-1].copy()
        if self.n_pts > 1:
            self._segments = self.centerline[1:] - self.centerline[:-1]
            self._segment_len_sq = np.sum(self._segments * self._segments, axis=1)
            self._segment_len_sq = np.clip(self._segment_len_sq, 1e-12, None)
        else:
            self._segments = np.zeros((0, 2), dtype=float)
            self._segment_len_sq = np.zeros(0, dtype=float)

    def width_at_idx(self, idx):
        idx = int(np.clip(idx, 0, self.n_pts - 1))
        return float(self.half_widths[idx])

    def closest_centerline_query(self, pos, i_start=0, i_end=None):
        """
        Find the closest point on the piecewise-linear centerline to `pos`.

        Returns
        -------
        idx              : int
        proj             : ndarray (2,)
        normal           : ndarray (2,)
        signed_d         : float
        abs_d            : float
        local_half_width : float
        """
        pos = np.asarray(pos, dtype=float)
        if self.n_pts == 1:
            proj = self.centerline[0]
            normal = self.normals[0]
            signed_d = float(np.dot(pos - proj, normal))
            return 0, proj, normal, signed_d, abs(signed_d), self.width_at_idx(0)

        if i_end is None:
            i_end = self.n_pts - 1
        i_start = int(np.clip(i_start, 0, self.n_pts - 2))
        i_end = int(np.clip(i_end, i_start + 1, self.n_pts - 1))

        best = None
        for seg_idx in range(i_start, i_end):
            a = self.centerline[seg_idx]
            ab = self._segments[seg_idx]
            t = float(np.dot(pos - a, ab) / self._segment_len_sq[seg_idx])
            t = float(np.clip(t, 0.0, 1.0))
            proj = a + t * ab
            delta = pos - proj
            dist_sq = float(np.dot(delta, delta))
            if best is not None and dist_sq >= best[0]:
                continue

            normal = (1.0 - t) * self.normals[seg_idx] + t * self.normals[seg_idx + 1]
            n_norm = float(np.linalg.norm(normal))
            if n_norm > 1e-12:
                normal = normal / n_norm
            else:
                normal = self.normals[seg_idx]

            local_half_width = float(
                (1.0 - t) * self.half_widths[seg_idx] + t * self.half_widths[seg_idx + 1]
            )
            signed_d = float(np.dot(delta, normal))
            nearest_idx = seg_idx if t < 0.5 else seg_idx + 1
            best = (dist_sq, nearest_idx, proj, normal, signed_d, abs(signed_d), local_half_width)

        _, idx, proj, normal, signed_d, abs_d, local_half_width = best
        return idx, proj, normal, signed_d, abs_d, local_half_width

    def closest_centerline_point(self, pos):
        """
        Find the closest point on the centerline to `pos`.

        Returns
        -------
        idx      : int — index of closest centerline point
        proj     : ndarray (2,) — closest point on centerline
        normal   : ndarray (2,) — normal at that point
        signed_d : float — signed distance from centerline
                   (positive = left of centerline, negative = right)
        abs_d    : float — absolute distance from centerline
        """
        idx, proj, normal, signed_d, abs_d, _ = self.closest_centerline_query(pos)
        return idx, proj, normal, signed_d, abs_d

    def is_inside(self, pos):
        """Check if pos is inside the tube."""
        _, _, _, _, abs_d, local_half_width = self.closest_centerline_query(pos)
        return abs_d < local_half_width

    def wall_penetration(self, pos):
        """
        Returns how far outside the tube the position is.
        0 if inside, >0 if penetrating a wall.
        Also returns which wall: 'left', 'right', or None.
        """
        idx, _, _, signed_d, abs_d, local_half_width = self.closest_centerline_query(pos)
        if abs_d <= local_half_width:
            return 0.0, None
        penetration = abs_d - local_half_width
        wall = 'left' if signed_d > 0 else 'right'
        return penetration, wall

    def progress(self, pos):
        """Return progress along the tube as fraction [0, 1]."""
        idx, _, _, _, _ = self.closest_centerline_point(pos)
        return idx / max(self.n_pts - 1, 1)

def _make_width_profile(
    n,
    base_width=DEFAULT_HALF_WIDTH,
    phase=0.0,
    waviness=0.16,
    narrow_sections=(),
    bulges=(),
):
    t = np.linspace(0.0, 1.0, int(n))
    profile = (
        1.0
        + waviness * np.sin(2.0 * np.pi * t + phase)
        + 0.08 * np.sin(5.0 * np.pi * t + 0.45 * phase)
    )
    for center, amp, spread in narrow_sections:
        profile -= float(amp) * np.exp(-0.5 * ((t - center) / spread) ** 2)
    for center, amp, spread in bulges:
        profile += float(amp) * np.exp(-0.5 * ((t - center) / spread) ** 2)
    widths = float(base_width) * profile
    return np.clip(widths, 0.0048, 0.0108)


def _apply_crack_irregularity(centerline, amplitude=0.0026, phase=0.0, spikes=(), sigma=2.0):
    """Perturb the centerline along its local normals to feel more like a fracture."""
    pts = np.asarray(centerline, dtype=float)
    t = np.linspace(0.0, 1.0, len(pts))
    normals = _normals(pts)
    offset = amplitude * (
        0.95 * np.sin(3.0 * np.pi * t + phase)
        + 0.55 * np.sin(7.0 * np.pi * t + 0.6 * phase)
        + 0.22 * np.sin(13.0 * np.pi * t + 1.4 * phase)
    )
    for center, magnitude, spread in spikes:
        offset += float(magnitude) * np.exp(-0.5 * ((t - center) / spread) ** 2)
    perturbed = pts + normals * offset[:, None]
    return _smooth(perturbed, sigma=sigma)


def _build_crack_tube(
    waypoints,
    half_width=DEFAULT_HALF_WIDTH,
    n=420,
    cx=DEFAULT_CX,
    cy=DEFAULT_CY,
    sx=0.88,
    sy=0.78,
    sigma=7,
    width_profile_kwargs=None,
    offset_profile_kwargs=None,
):
    waypoints = np.asarray(waypoints, dtype=float).copy()
    waypoints[:, 0] += cx
    waypoints[:, 1] += cy
    t_wp = np.linspace(0.0, 1.0, len(waypoints))
    t_fine = np.linspace(0.0, 1.0, int(n))
    x = np.interp(t_fine, t_wp, waypoints[:, 0])
    y = np.interp(t_fine, t_wp, waypoints[:, 1])
    centerline = _smooth(np.column_stack([x, y]), sigma=sigma)
    centerline = _compress_about_center(
        centerline,
        sx=sx * REACHABILITY_SCALE,
        sy=sy * REACHABILITY_SCALE,
    )
    if offset_profile_kwargs:
        centerline = _apply_crack_irregularity(centerline, **offset_profile_kwargs)
    profile_kwargs = dict(width_profile_kwargs or {})
    widths = _make_width_profile(len(centerline), base_width=half_width, **profile_kwargs)
    centerline, widths = _normalize_target_layout(centerline, widths, target_x=cx, target_y=cy)
    return Tube(centerline, widths)


def fault_arc_tube(half_width=DEFAULT_HALF_WIDTH, n=420, cx=DEFAULT_CX, cy=DEFAULT_CY):
    """Long, irregular crack with asymmetric bends and varying width."""
    waypoints = np.array([
        [0.028, -0.054],
        [-0.010, -0.045],
        [-0.036, -0.032],
        [-0.024, -0.014],
        [0.010, -0.002],
        [0.034,  0.014],
        [0.006,  0.026],
        [-0.030, 0.038],
        [-0.014, 0.052],
        [0.012,  0.062],
    ])
    return _build_crack_tube(
        waypoints,
        half_width=half_width,
        n=n,
        cx=cx,
        cy=cy,
        sx=0.90,
        sy=0.80,
        sigma=5,
        width_profile_kwargs={
            "phase": 0.3,
            "waviness": 0.22,
            "narrow_sections": ((0.20, 0.24, 0.05), (0.47, 0.15, 0.05), (0.74, 0.27, 0.06)),
            "bulges": ((0.58, 0.18, 0.07),),
        },
        offset_profile_kwargs={
            "amplitude": 0.0032,
            "phase": 0.4,
            "spikes": ((0.16, -0.0036, 0.040), (0.54, 0.0032, 0.050), (0.83, -0.0028, 0.035)),
            "sigma": 1.6,
        },
    )


def dogleg_breach_tube(half_width=DEFAULT_HALF_WIDTH, n=420, cx=DEFAULT_CX, cy=DEFAULT_CY):
    """Dogleg-shaped crack with a tighter throat and wider exit pocket."""
    waypoints = np.array([
        [0.034, -0.052],
        [0.020, -0.042],
        [-0.004, -0.032],
        [-0.032, -0.020],
        [-0.038, -0.004],
        [-0.012,  0.008],
        [0.020,  0.018],
        [0.040,  0.032],
        [0.018,  0.048],
        [-0.020, 0.062],
    ])
    return _build_crack_tube(
        waypoints,
        half_width=half_width,
        n=n,
        cx=cx,
        cy=cy,
        sx=0.92,
        sy=0.82,
        sigma=5,
        width_profile_kwargs={
            "phase": 1.15,
            "waviness": 0.21,
            "narrow_sections": ((0.28, 0.18, 0.05), (0.50, 0.33, 0.040), (0.67, 0.18, 0.05)),
            "bulges": ((0.10, 0.10, 0.04), (0.84, 0.22, 0.07)),
        },
        offset_profile_kwargs={
            "amplitude": 0.0030,
            "phase": 1.3,
            "spikes": ((0.25, 0.0034, 0.035), (0.49, -0.0042, 0.032), (0.76, 0.0031, 0.040)),
            "sigma": 1.7,
        },
    )


def rift_crack_tube(half_width=DEFAULT_HALF_WIDTH, n=420, cx=DEFAULT_CX, cy=DEFAULT_CY):
    """Offset fracture with multiple irregular kinks and variable diameter."""
    waypoints = np.array([
        [-0.020, -0.054],
        [0.010,  -0.046],
        [0.038,  -0.034],
        [0.028,  -0.018],
        [-0.006, -0.008],
        [-0.034, 0.006],
        [-0.026, 0.022],
        [0.008,  0.034],
        [0.032,  0.046],
        [0.010,  0.062],
    ])
    return _build_crack_tube(
        waypoints,
        half_width=half_width,
        n=n,
        cx=cx,
        cy=cy,
        sx=0.94,
        sy=0.84,
        sigma=4,
        width_profile_kwargs={
            "phase": 2.2,
            "waviness": 0.23,
            "narrow_sections": ((0.18, 0.15, 0.04), (0.42, 0.22, 0.05), (0.72, 0.26, 0.05)),
            "bulges": ((0.08, 0.16, 0.04), (0.56, 0.12, 0.06), (0.88, 0.14, 0.05)),
        },
        offset_profile_kwargs={
            "amplitude": 0.0034,
            "phase": 2.35,
            "spikes": ((0.19, -0.0030, 0.035), (0.37, 0.0040, 0.040), (0.63, -0.0038, 0.040), (0.85, 0.0028, 0.035)),
            "sigma": 1.5,
        },
    )


def pinch_fault_tube(half_width=DEFAULT_HALF_WIDTH, n=420, cx=DEFAULT_CX, cy=DEFAULT_CY):
    """Crack with two sharp doglegs and a pronounced pinch section."""
    waypoints = np.array([
        [0.016, -0.056],
        [-0.020, -0.044],
        [-0.040, -0.030],
        [-0.010, -0.018],
        [0.030,  -0.006],
        [0.040,   0.010],
        [0.012,   0.022],
        [-0.028,  0.032],
        [-0.038,  0.046],
        [-0.010,  0.060],
    ])
    return _build_crack_tube(
        waypoints,
        half_width=half_width,
        n=n,
        cx=cx,
        cy=cy,
        sx=0.92,
        sy=0.82,
        sigma=4,
        width_profile_kwargs={
            "phase": 2.9,
            "waviness": 0.24,
            "narrow_sections": ((0.32, 0.20, 0.045), (0.56, 0.36, 0.035), (0.78, 0.18, 0.04)),
            "bulges": ((0.12, 0.14, 0.05), (0.92, 0.18, 0.05)),
        },
        offset_profile_kwargs={
            "amplitude": 0.0036,
            "phase": 2.8,
            "spikes": ((0.27, 0.0034, 0.030), (0.52, -0.0046, 0.030), (0.70, 0.0030, 0.035)),
            "sigma": 1.4,
        },
    )


def offset_ravage_tube(half_width=DEFAULT_HALF_WIDTH, n=420, cx=DEFAULT_CX, cy=DEFAULT_CY):
    """Highly irregular breach with sweeping offset and multiple throat changes."""
    waypoints = np.array([
        [-0.030, -0.054],
        [0.000,  -0.046],
        [0.030,  -0.038],
        [0.042,  -0.022],
        [0.018,  -0.008],
        [-0.020,  0.004],
        [-0.042,  0.018],
        [-0.012,  0.030],
        [0.022,   0.040],
        [0.038,   0.056],
    ])
    return _build_crack_tube(
        waypoints,
        half_width=half_width,
        n=n,
        cx=cx,
        cy=cy,
        sx=0.96,
        sy=0.84,
        sigma=4,
        width_profile_kwargs={
            "phase": 0.9,
            "waviness": 0.25,
            "narrow_sections": ((0.24, 0.16, 0.04), (0.47, 0.24, 0.04), (0.68, 0.20, 0.04)),
            "bulges": ((0.10, 0.13, 0.04), (0.84, 0.20, 0.06)),
        },
        offset_profile_kwargs={
            "amplitude": 0.0038,
            "phase": 0.85,
            "spikes": ((0.18, -0.0032, 0.030), (0.41, 0.0042, 0.035), (0.63, -0.0041, 0.032), (0.88, 0.0032, 0.030)),
            "sigma": 1.4,
        },
    )


def _rotate_centerline(centerline, angle_deg):
    """Rotate centerline by angle_deg around its own center."""
    pts = np.asarray(centerline, dtype=float)
    pivot = pts.mean(axis=0)
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s], [s, c]])
    return (pts - pivot) @ R.T + pivot


def rotate_tube(tube, angle_deg):
    """Return a rotated copy of an existing Tube."""
    angle_deg = float(angle_deg) % 360.0
    target_x, target_y = _bbox_center(tube.centerline)
    if abs(angle_deg) < 1e-9:
        centerline, widths = _normalize_target_layout(
            np.asarray(tube.centerline, dtype=float).copy(),
            tube.half_widths.copy(),
            target_x=target_x,
            target_y=target_y,
        )
        return Tube(centerline, widths)
    rotated_cl = _rotate_centerline(tube.centerline, angle_deg)
    centerline, widths = _normalize_target_layout(
        rotated_cl,
        tube.half_widths.copy(),
        target_x=target_x,
        target_y=target_y,
    )
    return Tube(centerline, widths)


def s_curve_tube_rotated(angle_deg, half_width=DEFAULT_HALF_WIDTH, n=400, cx=DEFAULT_CX, cy=DEFAULT_CY):
    base = fault_arc_tube(half_width=half_width, n=n, cx=cx, cy=cy)
    rotated_cl = _rotate_centerline(base.centerline, angle_deg)
    centerline, widths = _normalize_target_layout(
        rotated_cl,
        base.half_widths.copy(),
        target_x=_bbox_center(base.centerline)[0],
        target_y=_bbox_center(base.centerline)[1],
    )
    return Tube(centerline, widths)


# ─── Catalogue ──────────────────────────────────────────────────────────────
TUBES = {
    "fault_arc": fault_arc_tube,
    "dogleg_breach": dogleg_breach_tube,
    "rift_crack": rift_crack_tube,
    "pinch_fault": pinch_fault_tube,
    "offset_ravage": offset_ravage_tube,
    # Backward-compatible aliases for older saved names / scripts.
    "s_curve": fault_arc_tube,
    "spiral": dogleg_breach_tube,
    "zigzag": rift_crack_tube,
}

BASE_TUBE_NAMES = ["fault_arc", "dogleg_breach", "rift_crack", "pinch_fault", "offset_ravage"]
ROTATION_ANGLES_DEG = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
TUBE_NAMES = BASE_TUBE_NAMES.copy()

def get_tube(name="fault_arc", **kwargs):
    """Return a Tube object by name."""
    return TUBES[name](**kwargs)


def get_rotated_tube(base_name="fault_arc", angle_deg=0, **kwargs):
    """Return a base tube shape rotated by the requested angle."""
    base = get_tube(base_name, **kwargs)
    return rotate_tube(base, angle_deg)
