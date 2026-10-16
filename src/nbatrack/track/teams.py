"""Unsupervised team / referee assignment from jersey color (per clip).

Per detection: a hue-saturation-value histogram of the torso region (upper-middle part of
the box, which avoids floor, shorts and most of the arms). Per track: the mean histogram.
Tracks are clustered into k = 3 groups (team A, team B, referees) with k-means
(cv2.kmeans, several restarts). The referee cluster is the one with the fewest tracks
(ties: lowest saturation). Track fragmentation can break this rule, so role labels must be
verified visually. Team labels "A"/"B" are arbitrary per
clip -- they are not linked to a real franchise.

This is a simple, CPU-cheap baseline; its accuracy must be checked visually / with
annotation. Confidence = margin between the nearest and second-nearest cluster centers.
"""

from __future__ import annotations

import cv2
import numpy as np

H_BINS, S_BINS, V_BINS = 12, 4, 4


def torso_histogram(bgr: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w < 6 or h < 16:
        return None
    tx1, tx2 = int(x1 + 0.25 * w), int(x2 - 0.25 * w)
    ty1, ty2 = int(y1 + 0.18 * h), int(y1 + 0.50 * h)
    H, W = bgr.shape[:2]
    tx1, ty1 = max(0, tx1), max(0, ty1)
    tx2, ty2 = min(W, tx2), min(H, ty2)
    if tx2 - tx1 < 3 or ty2 - ty1 < 3:
        return None
    hsv = cv2.cvtColor(bgr[ty1:ty2, tx1:tx2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [H_BINS, S_BINS, V_BINS], [0, 180, 0, 256, 0, 256])
    hist = hist.ravel().astype(np.float32)
    return hist / max(hist.sum(), 1e-6)


def torso_lab(bgr: np.ndarray, box: np.ndarray) -> tuple[float, float, float]:
    """Median CIE-Lab color (L 0-100, a, b centered at 0) of the central torso region
    (x 30-70%, y 18-45% of the box), which avoids arms, floor and shorts."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    H, W = bgr.shape[:2]
    tx1, tx2 = max(0, int(x1 + 0.30 * w)), min(W, int(x1 + 0.70 * w))
    ty1, ty2 = max(0, int(y1 + 0.18 * h)), min(H, int(y1 + 0.45 * h))
    if tx2 - tx1 < 2 or ty2 - ty1 < 2:
        return (np.nan, np.nan, np.nan)
    lab = cv2.cvtColor(bgr[ty1:ty2, tx1:tx2], cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    m = np.median(lab, axis=0)
    return (float(m[0] * 100 / 255), float(m[1] - 128), float(m[2] - 128))


# NBA referees wear gray shirts: mid lightness, low chroma. This is a domain prior and
# fails if a team wears gray; such clips must be checked.
REF_L, REF_L_TOL, REF_CHROMA_TOL = 55.0, 20.0, 12.0


def cluster_tracks_lab(feats: dict, k: int = 3, seed: int = 0,
                       return_centers: bool = False):
    """k-means (k=3) on per-track median torso Lab; the cluster closest to referee gray
    becomes REF, the other two are teams A / B (arbitrary per clip)."""
    ids = sorted(feats)
    if len(ids) < k:
        res = {i: ("UNK", 0.0) for i in ids}
        return (res, None) if return_centers else res
    X = np.stack([feats[i] for i in ids]).astype(np.float32)
    cv2.setRNGSeed(seed)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-3)
    _, lab, centers = cv2.kmeans(X, k, None, crit, 10, cv2.KMEANS_PP_CENTERS)
    lab = lab.ravel()
    score = [abs(c[0] - REF_L) / REF_L_TOL + np.hypot(c[1], c[2]) / REF_CHROMA_TOL for c in centers]
    ref = int(np.argmin(score))
    teams = [c for c in range(k) if c != ref]
    names = {ref: "REF", teams[0]: "A", teams[1]: "B"}
    d = np.linalg.norm(X[:, None, :] - centers[None], axis=2)
    srt = np.sort(d, axis=1)
    conf = np.clip((srt[:, 1] - srt[:, 0]) / (srt[:, 1] + 1e-9), 0, 1)
    res = {i: (names[int(l)], float(c)) for i, l, c in zip(ids, lab, conf)}
    return (res, centers) if return_centers else res


def mean_saturation(hist: np.ndarray) -> float:
    h = hist.reshape(H_BINS, S_BINS, V_BINS)
    s_centers = (np.arange(S_BINS) + 0.5) / S_BINS
    return float((h.sum(axis=(0, 2)) * s_centers).sum())


def cluster_tracks(track_hists: dict[int, np.ndarray], k: int = 3,
                   seed: int = 0) -> dict[int, tuple[str, float]]:
    """Return track_id -> (label in {"A", "B", "REF"}, confidence 0..1)."""
    ids = sorted(track_hists)
    if len(ids) < k:
        return {i: ("UNK", 0.0) for i in ids}
    X = np.sqrt(np.stack([track_hists[i] for i in ids]).astype(np.float32))  # Hellinger
    cv2.setRNGSeed(seed)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    _, lab, centers = cv2.kmeans(X, k, None, crit, 10, cv2.KMEANS_PP_CENTERS)
    lab = lab.ravel()
    # Referee cluster: the one with the fewest tracks (3 referees vs 5 + 5 players); ties
    # broken by lower saturation. (Saturation alone fails when a team wears black or
    # white -- observed on the first real clip, POR black vs GSW white.)
    sats = [mean_saturation(c ** 2) for c in centers]
    counts = np.bincount(lab.ravel(), minlength=k)
    ref = int(min(range(k), key=lambda c: (counts[c], sats[c]))) if k == 3 else -1
    teams = [c for c in range(k) if c != ref]
    names = {ref: "REF", teams[0]: "A", teams[1]: "B"} if k == 3 else {0: "A", 1: "B"}
    d = np.linalg.norm(X[:, None, :] - centers[None], axis=2)
    srt = np.sort(d, axis=1)
    conf = np.clip((srt[:, 1] - srt[:, 0]) / (srt[:, 1] + 1e-9), 0, 1)
    return {i: (names[int(l)], float(c)) for i, l, c in zip(ids, lab, conf)}
