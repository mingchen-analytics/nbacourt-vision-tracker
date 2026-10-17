"""Scoreboard readings -> game timeline -> candidate events.

Principles (camera segments, possessions and clock events are different things):
  * A value is only "observed" when the reader accepted every glyph; otherwise it is None.
  * A score / clock value counts only after it is stable over `min_stable` consecutive readable
    samples (scoreboard animations produce transient garbage).
  * Game clock runs <-> displayed value changes about once per second. A value that stays for
    clearly longer than a second means the clock was stopped (dead ball). The stop moment is only
    known to within ~1 s and is reported as such.
  * Candidate possession boundaries come ONLY from scoreboard evidence:
      score change          strong evidence that the scoring team's possession ended
      shot-clock reset to 24 medium evidence of a change of possession (also happens after some
                            fouls / kicked balls)
      shot-clock reset to 14 weak: usually the SAME team keeps the ball (offensive rebound, foul)
    Evidence within `merge_s` seconds is merged into one boundary. Camera cuts, flashes and
    replays are never possession boundaries; they only mark views / non-live time.
  * Offense team: inferred from score changes only (the team that scored had the ball; the
    other team inbounds next). Everything else stays unknown.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nbatrack.index.scoreboard import GlyphReader, parse_clock, parse_int


# ----------------------------------------------------------------------------- readings
def read_samples(samples: pd.DataFrame, glyph_meta: pd.DataFrame, glyphs: np.ndarray, reader: GlyphReader,
                 presence_min: float = 0.6) -> pd.DataFrame:
    """One row per sample with parsed field values (None = unreadable / not visible)."""
    texts: dict[tuple[int, str], tuple[str | None, float]] = {}
    for (sid, field), g in glyph_meta.groupby(["sample_id", "field"], sort=False):
        chars, conf = [], 1.0
        for i in g.sort_values("pos").index:
            c, s = reader.read_glyph(field, glyphs[i], float(glyph_meta.at[i, "aspect"]))
            if c is None:
                chars = None
                conf = s
                break
            chars.append(c)
            conf = min(conf, s)
        texts[(int(sid), field)] = ("".join(chars) if chars is not None else None, conf)
    rows = []
    for r in samples.itertuples():
        vis = r.presence is None or (r.presence == r.presence and r.presence >= presence_min)
        t = {f: texts.get((r.sample_id, f), (None, 0.0)) for f in ("away_score", "home_score", "period", "game_clock", "shot_clock")}
        per = t["period"][0]
        rows.append(dict(
            sample_id=r.sample_id, frame_index=r.frame_index, timestamp_sec=r.timestamp_sec, presence=r.presence,
            scoreboard_visible=bool(vis),
            away_score=parse_int(t["away_score"][0], 0, 250) if vis else None,
            home_score=parse_int(t["home_score"][0], 0, 250) if vis else None,
            period=(int(per[0]) if per and per[0].isdigit() else None) if vis else None,
            game_clock_s=parse_clock(t["game_clock"][0]) if vis else None,
            shot_clock=parse_int(t["shot_clock"][0], 0, 24) if vis else None,
            game_clock_text=t["game_clock"][0], shot_clock_text=t["shot_clock"][0],
            min_conf=min(v[1] for v in t.values())))
    return pd.DataFrame(rows)


def stable_values(ts: np.ndarray, vals: list, min_stable: int = 3) -> list[tuple[float, float, object]]:
    """Runs of an identical non-None value lasting >= min_stable readable samples:
    (t_first, t_last, value). None samples do not break a run."""
    runs = []
    cur, t0, t1, n = None, None, None, 0
    for t, v in zip(ts, vals):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if v == cur:
            t1, n = t, n + 1
        else:
            if cur is not None and n >= min_stable:
                runs.append((t0, t1, cur))
            cur, t0, t1, n = v, t, t, 1
    if cur is not None and n >= min_stable:
        runs.append((t0, t1, cur))
    return runs


# ----------------------------------------------------------------------------- timeline
@dataclass
class Timeline:
    readings: pd.DataFrame
    clock_runs: list  # (t0, t1, game_clock_s)
    shot_runs: list
    away_runs: list
    home_runs: list
    stopped: list  # (t_start, t_end, game_clock_s): clock stopped (dead ball), +-1 s at the start
    score_changes: list  # dict(t, team, delta, before, after)
    shot_resets: list  # dict(t, before, after)
    clock_checks: dict  # consistency statistics


def build_timeline(rd: pd.DataFrame, min_stable: int = 3, stop_min_s: float = 2.0) -> Timeline:
    ts = rd.timestamp_sec.to_numpy()
    clock_runs = stable_values(ts, rd.game_clock_s.tolist(), min_stable)
    shot_runs = stable_values(ts, rd.shot_clock.tolist(), min_stable)
    away_runs = stable_values(ts, rd.away_score.tolist(), min_stable)
    home_runs = stable_values(ts, rd.home_score.tolist(), min_stable)
    # clock consistency: consecutive stable runs should step down by exactly 1 s
    steps = [b[2] - a[2] for a, b in zip(clock_runs, clock_runs[1:])]
    n_ok = sum(1 for s in steps if s == -1)
    stopped = []
    for t0, t1, v in clock_runs:
        span = t1 - t0
        if span >= stop_min_s:  # displayed >= 2 s -> clock was stopped for about span - 1 s
            stopped.append((t0 + 1.0, t1, v))
    # running-segment slope: game seconds per video second over runs that step by -1
    slopes = []
    for a, b in zip(clock_runs, clock_runs[1:]):
        if b[2] - a[2] == -1 and (b[0] - a[0]) > 0:
            slopes.append(1.0 / (b[0] - a[0]))
    checks = dict(n_runs=len(clock_runs), n_steps=len(steps), steps_minus_one=n_ok,
                  steps_other=[int(s) for s in steps if s != -1],
                  step_consistency=n_ok / max(1, len(steps)),
                  median_seconds_per_tick=float(np.median([b[0] - a[0] for a, b in zip(clock_runs, clock_runs[1:])
                                                           if b[2] - a[2] == -1 and b[0] - a[0] < 1.5])) if n_ok else None)
    score_changes = []
    for team, runs in (("away", away_runs), ("home", home_runs)):
        for a, b in zip(runs, runs[1:]):
            d = b[2] - a[2]
            score_changes.append(dict(t=b[0], t_prev_last=a[1], team=team, delta=d, before=a[2], after=b[2],
                                      plausible=1 <= d <= 4))
    shot_resets = []
    for a, b in zip(shot_runs, shot_runs[1:]):
        if b[2] > a[2]:
            shot_resets.append(dict(t=b[0], t_prev_last=a[1], before=a[2], after=b[2]))
    return Timeline(rd, clock_runs, shot_runs, away_runs, home_runs, stopped,
                    sorted(score_changes, key=lambda x: x["t"]), shot_resets, checks)


def value_at(runs: list, t: float, tol: float = 0.6):
    """Observed stable value at time t (within tol of a run), else None."""
    for t0, t1, v in runs:
        if t0 - tol <= t <= t1 + tol:
            return v
    return None


def clock_at(tl: Timeline, t: float) -> tuple[float | None, str | None]:
    """Game clock at video time t: observed if inside a stable run; interpolated between two
    consecutive runs one tick apart (clock running); else None."""
    v = value_at(tl.clock_runs, t, tol=0.0)
    if v is not None:
        return v, "ocr_observed"
    for a, b in zip(tl.clock_runs, tl.clock_runs[1:]):
        if a[1] < t < b[0] and a[2] - b[2] == 1 and b[0] - a[1] < 1.0:
            return float(b[2]) + 1 - (t - a[1]) / (b[0] - a[1]), "interpolated"
    return None, None
