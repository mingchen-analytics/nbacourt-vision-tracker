"""Canonical NBA court geometry in meters.

Coordinate system (see docs/COORDINATE_SYSTEM.md):
  * Origin: center of the court, on the floor plane.
  * +X: along the sidelines, toward the basket on the RIGHT of the main broadcast camera.
  * +Y: along the baselines, toward the FAR sideline (away from the main camera).
  * +Z: up. Floor is z = 0. Right-handed.

Dimensions follow the NBA rulebook (Rule 1). Imperial values are kept as the source of
truth and converted with 1 ft = 0.3048 m exactly.

Line-width convention: painted lines are 2 in (~5 cm) wide. All geometry here is the
*nominal* line position used by the rulebook dimension (boundary lines: outer edge of the
playing area; arcs: measured from the basket center). Treat +/-5 cm as a systematic
uncertainty floor for any landmark-based calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

FT = 0.3048  # meters per foot (exact)
INCH = FT / 12.0


@dataclass(frozen=True)
class CourtSpec:
    length_ft: float = 94.0
    width_ft: float = 50.0
    hoop_from_baseline_ft: float = 5.25  # 63 in, baseline to rim center
    backboard_from_baseline_ft: float = 4.0
    backboard_width_ft: float = 6.0
    rim_height_ft: float = 10.0
    rim_radius_ft: float = 9.0 / 12.0  # 18 in inner diameter
    lane_width_ft: float = 16.0
    lane_length_ft: float = 19.0  # baseline to far edge of free-throw line
    ft_circle_radius_ft: float = 6.0
    center_circle_radius_ft: float = 6.0
    center_inner_circle_radius_ft: float = 2.0
    three_pt_radius_ft: float = 23.75
    three_pt_corner_ft: float = 22.0  # lateral distance from basket center to corner line
    restricted_radius_ft: float = 4.0

    # ---- derived quantities in meters -------------------------------------------------
    @property
    def half_length(self) -> float:
        return self.length_ft * FT / 2.0

    @property
    def half_width(self) -> float:
        return self.width_ft * FT / 2.0

    @property
    def hoop_x(self) -> float:
        """|x| of the rim center."""
        return self.half_length - self.hoop_from_baseline_ft * FT

    @property
    def rim_height(self) -> float:
        return self.rim_height_ft * FT

    @property
    def backboard_x(self) -> float:
        return self.half_length - self.backboard_from_baseline_ft * FT

    @property
    def lane_half_width(self) -> float:
        return self.lane_width_ft * FT / 2.0

    @property
    def ft_line_x(self) -> float:
        """|x| of the free-throw line."""
        return self.half_length - self.lane_length_ft * FT

    @property
    def three_pt_radius(self) -> float:
        return self.three_pt_radius_ft * FT

    @property
    def three_pt_corner_y(self) -> float:
        return self.three_pt_corner_ft * FT

    @property
    def three_pt_break_x(self) -> float:
        """|x| where the straight corner three-point line meets the arc.

        Computed from the arc radius and corner offset (~14.2 ft from the baseline),
        rather than the rounded '14 ft' often quoted, so that the polyline is continuous.
        """
        dx = math.sqrt(self.three_pt_radius**2 - self.three_pt_corner_y**2)
        return self.hoop_x - dx

    @property
    def three_pt_top_x(self) -> float:
        return self.hoop_x - self.three_pt_radius

    @property
    def ft_circle_radius(self) -> float:
        return self.ft_circle_radius_ft * FT

    @property
    def center_circle_radius(self) -> float:
        return self.center_circle_radius_ft * FT

    @property
    def restricted_radius(self) -> float:
        return self.restricted_radius_ft * FT


NBA_COURT = CourtSpec()


@dataclass(frozen=True)
class Landmark:
    landmark_id: int
    name: str
    xyz: tuple[float, float, float]
    description: str = ""
    on_floor: bool = field(default=True)

    @property
    def xy(self) -> tuple[float, float]:
        return (self.xyz[0], self.xyz[1])


def _side_name(s: int) -> str:
    return "L" if s < 0 else "R"


def court_landmarks(spec: CourtSpec = NBA_COURT) -> list[Landmark]:
    """Named point landmarks with known court coordinates.

    'near' = y < 0 (main-camera side), 'far' = y > 0. 'L'/'R' = x < 0 / x > 0.
    IDs are stable and must never be renumbered (they are stored in output files).
    """
    hl, hw = spec.half_length, spec.half_width
    out: list[Landmark] = []

    def add(name: str, x: float, y: float, z: float = 0.0, desc: str = "") -> None:
        out.append(Landmark(len(out), name, (float(x), float(y), float(z)), desc, z == 0.0))

    # Boundary and midcourt
    add("L_near_corner", -hl, -hw, desc="baseline x near sideline")
    add("L_far_corner", -hl, hw, desc="baseline x far sideline")
    add("R_near_corner", hl, -hw, desc="baseline x near sideline")
    add("R_far_corner", hl, hw, desc="baseline x far sideline")
    add("mid_near", 0.0, -hw, desc="midcourt line x near sideline")
    add("mid_far", 0.0, hw, desc="midcourt line x far sideline")
    r = spec.center_circle_radius
    add("center_circle_near", 0.0, -r, desc="center circle x midcourt line, near")
    add("center_circle_far", 0.0, r, desc="center circle x midcourt line, far")
    add("center_circle_L", -r, 0.0, desc="center circle, leftmost point")
    add("center_circle_R", r, 0.0, desc="center circle, rightmost point")

    for s in (-1, 1):
        n = _side_name(s)
        lw = spec.lane_half_width
        add(f"{n}_lane_baseline_near", s * hl, -lw, desc="lane edge x baseline")
        add(f"{n}_lane_baseline_far", s * hl, lw, desc="lane edge x baseline")
        add(f"{n}_lane_ft_near", s * spec.ft_line_x, -lw, desc="lane edge x free-throw line")
        add(f"{n}_lane_ft_far", s * spec.ft_line_x, lw, desc="lane edge x free-throw line")
        add(
            f"{n}_ft_circle_top",
            s * (spec.ft_line_x - spec.ft_circle_radius),
            0.0,
            desc="free-throw circle apex toward midcourt",
        )
        c = spec.three_pt_corner_y
        add(f"{n}_3pt_corner_baseline_near", s * hl, -c, desc="corner 3pt line x baseline")
        add(f"{n}_3pt_corner_baseline_far", s * hl, c, desc="corner 3pt line x baseline")
        add(f"{n}_3pt_break_near", s * spec.three_pt_break_x, -c, desc="corner line meets arc")
        add(f"{n}_3pt_break_far", s * spec.three_pt_break_x, c, desc="corner line meets arc")
        add(f"{n}_3pt_top", s * spec.three_pt_top_x, 0.0, desc="3pt arc apex")
        add(f"{n}_rim_center", s * spec.hoop_x, 0.0, spec.rim_height, desc="rim center (3D)")
    return out


def landmark_by_name(spec: CourtSpec = NBA_COURT) -> dict[str, Landmark]:
    return {lm.name: lm for lm in court_landmarks(spec)}


def hoop_positions(spec: CourtSpec = NBA_COURT) -> dict[str, np.ndarray]:
    """3D rim centers, keyed 'left' / 'right'."""
    return {
        "left": np.array([-spec.hoop_x, 0.0, spec.rim_height]),
        "right": np.array([spec.hoop_x, 0.0, spec.rim_height]),
    }


def _arc(cx: float, cy: float, r: float, a0: float, a1: float, step_m: float) -> np.ndarray:
    n = max(8, int(math.ceil(abs(a1 - a0) * r / step_m)) + 1)
    a = np.linspace(a0, a1, n)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)


def court_polylines(spec: CourtSpec = NBA_COURT, step_m: float = 0.1) -> dict[str, np.ndarray]:
    """Floor line markings as named (N, 2) polylines in court meters (z = 0).

    Curves are sampled every `step_m` meters. Straight lines are 2-point polylines;
    use `densify` for uniform sampling when computing alignment residuals.
    """
    hl, hw = spec.half_length, spec.half_width
    lines: dict[str, np.ndarray] = {
        "boundary": np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw], [-hl, -hw]]),
        "midcourt": np.array([[0.0, -hw], [0.0, hw]]),
        "center_circle": _arc(0.0, 0.0, spec.center_circle_radius, 0, 2 * math.pi, step_m),
    }
    for s in (-1, 1):
        n = _side_name(s)
        lw, fx = spec.lane_half_width, spec.ft_line_x
        lines[f"{n}_lane"] = np.array(
            [[s * hl, -lw], [s * fx, -lw], [s * fx, lw], [s * hl, lw]]
        )
        lines[f"{n}_ft_circle"] = _arc(s * fx, 0.0, spec.ft_circle_radius, 0, 2 * math.pi, step_m)
        c = spec.three_pt_corner_y
        bx = spec.three_pt_break_x
        hx = spec.hoop_x
        phi = math.asin(c / spec.three_pt_radius)
        # Arc opens toward midcourt; angles around the hoop center, ordered near (-y) -> far.
        if s > 0:
            arc = _arc(hx, 0.0, spec.three_pt_radius, math.pi + phi, math.pi - phi, step_m)
        else:
            arc = _arc(-hx, 0.0, spec.three_pt_radius, -phi, phi, step_m)
        three = np.vstack([[[s * hl, -c], [s * bx, -c]], arc, [[s * bx, c], [s * hl, c]]])
        lines[f"{n}_3pt"] = three
        ra = spec.restricted_radius
        if s > 0:
            lines[f"{n}_restricted"] = _arc(hx, 0.0, ra, math.pi / 2, 3 * math.pi / 2, step_m)
        else:
            lines[f"{n}_restricted"] = _arc(-hx, 0.0, ra, -math.pi / 2, math.pi / 2, step_m)
    return lines


def densify(polyline: np.ndarray, step_m: float = 0.1) -> np.ndarray:
    """Resample a polyline so consecutive points are at most `step_m` apart."""
    pts = [polyline[0]]
    for a, b in zip(polyline[:-1], polyline[1:]):
        d = float(np.linalg.norm(b - a))
        k = max(1, int(math.ceil(d / step_m)))
        t = np.linspace(0, 1, k + 1)[1:, None]
        pts.extend(a + t * (b - a))
    return np.asarray(pts)


def is_on_court(xy: np.ndarray, margin_m: float = 0.0, spec: CourtSpec = NBA_COURT) -> np.ndarray:
    """Boolean mask: points within the playing area expanded by `margin_m`."""
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
    return (np.abs(xy[:, 0]) <= spec.half_length + margin_m) & (
        np.abs(xy[:, 1]) <= spec.half_width + margin_m
    )
