"""Broadcast scoreboard reading without an OCR model: glyph templates per broadcast.

Each scoreboard field (scores, period, game clock, shot clock) is a fixed region of a given
broadcast graphics package. Inside a field the text is the MINORITY class of an Otsu
threshold (bright-on-dark or dark-on-bright both work). Connected components tall enough to be
characters are cut out and normalised to a small binary glyph.

Reading uses labelled glyph templates:
  1. glyphs from sampled frames are clustered (normalised correlation), per field;
  2. a person labels each cluster ONCE per graphics package (`glyph_labels.json`, written
     with the help of a contact sheet of cluster examples);
  3. every glyph is matched to the labelled templates. A glyph is accepted only if its best
     match is good enough and clearly better than the best match with a different label;
     otherwise the whole field reading is None (unreadable). Values are never guessed.

This keeps the module dependency-free (OpenCV + NumPy) and replaceable by a real OCR model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

GLYPH_W, GLYPH_H = 16, 24


@dataclass
class ScoreboardConfig:
    fields: dict[str, tuple[int, int, int, int]]  # name -> x1, y1, x2, y2 (source pixels)
    teams: dict
    presence_roi: tuple[int, int, int, int] | None = None
    presence_reference_frame: int | None = None
    image_size: tuple[int, int] = (1280, 720)

    @classmethod
    def load(cls, path: str | Path) -> "ScoreboardConfig":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({k: tuple(v) for k, v in d["fields"].items()}, d.get("teams", {}),
                   tuple(d["presence_roi"]) if d.get("presence_roi") else None,
                   d.get("presence_reference_frame"), tuple(d.get("image_size", (1280, 720))))


def crop(img: np.ndarray, roi) -> np.ndarray:
    x1, y1, x2, y2 = roi
    return img[y1:y2, x1:x2]


def field_glyphs(bgr_crop: np.ndarray, min_height_frac: float = 0.45) -> list[dict]:
    """Character glyphs in one field crop, left to right.

    Returns dicts with `glyph` (GLYPH_H x GLYPH_W float32 in [0, 1]), `box` (x, y, w, h in
    crop pixels) and `aspect` (w / h). Components shorter than `min_height_frac` of the
    tallest component (colons, superscripts, specks) are ignored."""
    g = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)  # small text: upsample first
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (th > 0).mean() > 0.5:  # text = minority class
        th = 255 - th
    n, lab, st, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    comps = [(i, *st[i, :4]) for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 12]
    if not comps:
        return []
    H, W = th.shape
    # drop components touching the crop border (background pieces, box edges)
    comps = [c for c in comps if c[1] > 0 and c[2] > 0 and c[1] + c[3] < W and c[2] + c[4] < H] or comps
    hmax = max(c[4] for c in comps)
    out = []
    for i, x, y, w, h in sorted(comps, key=lambda c: c[1]):
        if h < min_height_frac * hmax:
            continue
        m = (lab[y:y + h, x:x + w] == i).astype(np.float32)
        out.append(dict(glyph=cv2.resize(m, (GLYPH_W, GLYPH_H), interpolation=cv2.INTER_AREA),
                        box=(x / 3, y / 3, w / 3, h / 3), aspect=w / max(h, 1)))
    return out


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def cluster_glyphs(glyphs: list[np.ndarray], aspects: list[float], thr: float = 0.88,
                   max_aspect_ratio: float = 1.35) -> np.ndarray:
    """Greedy leader clustering: a glyph joins the first cluster whose leader correlates >= thr
    and has a similar aspect ratio; otherwise it starts a new cluster. Returns cluster ids."""
    leaders: list[tuple[np.ndarray, float]] = []
    ids = np.empty(len(glyphs), int)
    for k, (g, a) in enumerate(zip(glyphs, aspects)):
        best, bi = -1.0, -1
        for ci, (lg, la) in enumerate(leaders):
            if max(a, la) / max(min(a, la), 1e-6) > max_aspect_ratio:
                continue
            s = ncc(g, lg)
            if s > best:
                best, bi = s, ci
        if best >= thr:
            ids[k] = bi
        else:
            leaders.append((g, a))
            ids[k] = len(leaders) - 1
    return ids


@dataclass
class GlyphReader:
    """Labelled templates per field: field -> list of (label, glyph, aspect)."""
    templates: dict[str, list[tuple[str, np.ndarray, float]]]
    min_score: float = 0.80
    min_margin: float = 0.05

    @classmethod
    def load(cls, npz_path: str | Path, labels_path: str | Path) -> "GlyphReader":
        """npz: per field `<field>__leaders` (k, H, W) and `<field>__aspects` (k,); labels json:
        {field: {cluster_id: char}}. Clusters labelled "?" or "" are not used as templates."""
        z = np.load(npz_path)
        labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))["labels"]
        t: dict = {}
        for f, lm in labels.items():
            L, A = z[f"{f}__leaders"], z[f"{f}__aspects"]
            t[f] = [(c, L[int(i)], float(A[int(i)])) for i, c in lm.items() if c not in ("", "?")]
        return cls(t)

    def read_glyph(self, field: str, g: np.ndarray, aspect: float) -> tuple[str | None, float]:
        scores: dict[str, float] = {}
        for lab, tg, ta in self.templates.get(field, []):
            if max(aspect, ta) / max(min(aspect, ta), 1e-6) > 1.5:
                continue
            s = ncc(g, tg)
            scores[lab] = max(scores.get(lab, -1.0), s)
        if not scores:
            return None, 0.0
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        best, s1 = ranked[0]
        s2 = ranked[1][1] if len(ranked) > 1 else -1.0
        if s1 < self.min_score or s1 - s2 < self.min_margin:
            return None, s1
        return best, s1

    def read_field(self, field: str, bgr_crop: np.ndarray) -> tuple[str | None, float]:
        """Field text (characters joined) or None if any glyph is unreadable; confidence = the
        weakest glyph score."""
        gl = field_glyphs(bgr_crop)
        if not gl:
            return None, 0.0
        chars, conf = [], 1.0
        for d in gl:
            c, s = self.read_glyph(field, d["glyph"], d["aspect"])
            if c is None:
                return None, s
            chars.append(c)
            conf = min(conf, s)
        return "".join(chars), conf


def parse_clock(text: str | None) -> float | None:
    """Game-clock text from glyphs without the colon/point: "936" -> 9:36 = 576 s, "1200" ->
    12:00. Values with 2 digits are read as seconds (the broadcast may switch to "SS.t" under a
    minute; such tenths readings are not parsed here and return None)."""
    if not text or not text.isdigit():
        return None
    if len(text) == 3:
        m, s = int(text[0]), int(text[1:])
    elif len(text) == 4:
        m, s = int(text[:2]), int(text[2:])
    else:
        return None
    if s >= 60 or m > 12:
        return None
    return float(60 * m + s)


def parse_int(text: str | None, lo: int = 0, hi: int = 300) -> int | None:
    if not text or not text.isdigit():
        return None
    v = int(text)
    return v if lo <= v <= hi else None
