"""Video Indexing Engine CLI (MVP). Independent of the player-tracking pipeline.

  extract   decode once: shot statistics + scoreboard glyphs of sampled frames
  glyphs    cluster glyphs per field and write a labelling sheet + label template
            (label each cluster ONCE per broadcast graphics package)
  index     read the scoreboard with the labelled templates, build the game timeline and
            candidate possessions, write events (JSON + Parquet)

  python tools/index_video.py extract data/raw/clip6_long_115s.mp4 --scoreboard data/annotations/clip6/scoreboard.json --out outputs/index/clip6
  python tools/index_video.py glyphs --out outputs/index/clip6 --labels data/annotations/clip6/glyph_labels.json
  python tools/index_video.py index --out outputs/index/clip6 --scoreboard ... --labels ... \
        [--segments data/annotations/clip6/segments.json] [--tracking-run outputs/real/clip6_long_115s]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbatrack.index.extract import Extraction, extract  # noqa: E402
from nbatrack.index.scoreboard import ScoreboardConfig, cluster_glyphs  # noqa: E402


def cmd_extract(a):
    cfg = ScoreboardConfig.load(a.scoreboard)
    ex = extract(a.video, cfg, sample_every=a.sample_every, max_frames=a.max_frames)
    ex.save(Path(a.out))
    print(json.dumps(ex.info, indent=1))


def cmd_glyphs(a):
    out = Path(a.out)
    ex = Extraction.load(out)
    leaders, sheets, template = {}, [], {"note": "label each cluster with its character; '?' = not a character / ambiguous",
                                         "annotator": "", "labels": {}}
    for field, g in ex.glyph_meta.groupby("field", sort=False):
        idx = g.index.to_numpy()
        ids = cluster_glyphs([ex.glyphs[i] for i in idx], g.aspect.tolist(), thr=a.thr)
        k = int(ids.max()) + 1
        L = np.stack([ex.glyphs[idx[ids == c]].mean(0) for c in range(k)])  # cluster mean = template
        A = np.array([float(np.median(g.aspect.to_numpy()[ids == c])) for c in range(k)])
        cnt = np.array([(ids == c).sum() for c in range(k)])
        leaders[f"{field}__leaders"], leaders[f"{field}__aspects"], leaders[f"{field}__counts"] = L, A, cnt
        template["labels"][field] = {str(c): "" for c in range(k)}
        # sheet: per cluster, up to 8 examples (aspect-correct, upscaled) with id and count
        rows = []
        for c in range(k):
            ex_idx = idx[ids == c][:8]
            tiles = []
            for i in ex_idx:
                w = max(8, int(round(48 * g.loc[i, "aspect"])))
                tiles.append(cv2.copyMakeBorder(cv2.resize((ex.glyphs[i] * 255).astype(np.uint8), (w, 48)), 2, 2, 2, 2,
                                                cv2.BORDER_CONSTANT, value=90))
            strip = np.hstack(tiles)
            lab = np.full((52, 150), 40, np.uint8)
            cv2.putText(lab, f"{field[:10]} #{c} n={cnt[c]}", (3, 32), 0, 0.42, 255, 1)
            row = np.hstack([lab, strip])
            rows.append(row)
        W = max(r.shape[1] for r in rows)
        sheets.append(np.vstack([cv2.copyMakeBorder(r, 0, 2, 0, W - r.shape[1], cv2.BORDER_CONSTANT, value=0) for r in rows]))
        print(field, "clusters", k, "glyphs", len(idx))
    np.savez(out / "glyph_clusters.npz", **leaders)
    W = max(s.shape[1] for s in sheets)
    sheet = np.vstack([cv2.copyMakeBorder(s, 0, 12, 0, W - s.shape[1], cv2.BORDER_CONSTANT, value=255) for s in sheets])
    cv2.imwrite(str(out / "glyph_clusters_sheet.png"), sheet)
    lp = Path(a.labels)
    if not lp.exists():
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text(json.dumps(template, indent=1), encoding="utf-8")
        print("label template written:", lp)
    else:
        print("labels file exists (not overwritten):", lp)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract")
    p.add_argument("video")
    p.add_argument("--scoreboard", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sample-every", type=int, default=6)
    p.add_argument("--max-frames", type=int)
    p.set_defaults(fn=cmd_extract)
    p = sub.add_parser("glyphs")
    p.add_argument("--out", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--thr", type=float, default=0.88)
    p.set_defaults(fn=cmd_glyphs)
    p = sub.add_parser("index")
    p.add_argument("--out", required=True)
    p.add_argument("--scoreboard", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--segments")
    p.add_argument("--tracking-run")
    p.add_argument("--game-id", default="")
    p.set_defaults(fn=lambda a: __import__("index_build").cmd_index(a))
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
