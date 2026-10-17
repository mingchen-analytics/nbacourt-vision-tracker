"""Timeline -> events (schema 0.1): camera segments, non-live intervals, score changes,
shot-clock resets and candidate possessions.

Possession boundaries (see timeline.py for the rules):
  * a shot-clock reset to 24 followed within `make_window_s` by a score change = a made
    basket: one boundary at the RESET time (the scoreboard graphic updates 1-3 s after the make),
    confidence 0.9
  * a score change without a preceding reset (e.g. free throws): boundary at the score change,
    confidence 0.75
  * a reset to 24 without a score change: likely change of possession (turnover / defensive
    rebound / foul), confidence 0.6
  * a reset to 14: weak evidence; reported as a shot_clock_reset event, never as a boundary.
Candidate possessions are the intervals between consecutive boundaries (and the video ends).
Offense team is inferred only from score changes (scorer had the ball; the other team inbounds
next) and from alternation between boundaries of confidence >= 0.6; conflicts -> unknown.
"""

from __future__ import annotations

import pandas as pd

from nbatrack.index.schema import new_event
from nbatrack.index.timeline import Timeline, clock_at, value_at


def _ctx(tl: Timeline, t0: float, t1: float, fps_frames: pd.DataFrame, segs: pd.DataFrame, period_runs: list,
         tracking: dict | None) -> dict:
    fr = fps_frames
    f0 = int(fr.frame_index.iloc[(fr.timestamp_sec - t0).abs().argmin()])
    f1 = int(fr.frame_index.iloc[(fr.timestamp_sec - t1).abs().argmin()])
    g0, s0 = clock_at(tl, t0)
    g1, s1 = clock_at(tl, t1)
    seg = segs[(segs.start_frame <= f0) & (segs.end_frame >= f0)]
    per = value_at(period_runs, t0)
    sc = value_at(tl.shot_runs, t0)
    a0, h0 = value_at(tl.away_runs, t0), value_at(tl.home_runs, t0)
    a1, h1 = value_at(tl.away_runs, t1), value_at(tl.home_runs, t1)
    return dict(
        video_start_seconds=round(float(t0), 4), video_end_seconds=round(float(t1), 4), frame_start=f0, frame_end=f1,
        period=int(per) if per is not None else None, period_source="ocr_observed" if per is not None else None,
        game_clock_start=g0, game_clock_end=g1,
        game_clock_source=s0 if s0 == s1 or s1 is None else ("interpolated" if "interpolated" in (s0, s1) else s0),
        shot_clock_start=int(sc) if sc is not None else None, shot_clock_source="ocr_observed" if sc is not None else None,
        score_away_start=None if a0 is None else int(a0), score_home_start=None if h0 is None else int(h0),
        score_away_end=None if a1 is None else int(a1), score_home_end=None if h1 is None else int(h1),
        score_source="ocr_observed" if None not in (a0, h0) else None,
        camera_segment_id=int(seg.camera_segment_id.iloc[0]) if len(seg) else None,
        tracking_data_reference=dict(tracking, frame_start=f0, frame_end=f1) if tracking else None)


def build_events(tl: Timeline, frames: pd.DataFrame, segs: pd.DataFrame, teams: dict, video_id: str,
                 game_id: str = "", period_runs: list | None = None, tracking: dict | None = None,
                 make_window_s: float = 4.0) -> tuple[list[dict], list[dict]]:
    """Returns (events, boundaries)."""
    period_runs = period_runs or []
    away, home = teams["away"]["id"], teams["home"]["id"]
    tid = {"away": away, "home": home}
    other = {away: home, home: away}
    t_end = float(frames.timestamp_sec.iloc[-1])
    ev: list[dict] = []

    def add(kind, t0, t1, **kw):
        e = new_event(event_id=f"{video_id}:{kind}:{len(ev):04d}", game_id=game_id, video_id=video_id, event_type=kind,
                      **_ctx(tl, t0, t1, frames, segs, period_runs, tracking))
        e.update(kw)
        ev.append(e)
        return e

    # camera segments (views, NOT possessions)
    for s in segs.itertuples():
        add("camera_segment", s.start_sec, s.end_sec, labels=[s.view_type], boundary_source=["camera_cut"],
            event_confidence=1.0 if s.view_source == "reviewed" else 0.8,
            review_status="confirmed" if s.view_source == "reviewed" else "auto",
            evidence=[dict(kind="view", view_type=s.view_type, source=s.view_source,
                           median_floor_frac=getattr(s, "median_floor_frac", None))])
    # non-live: clock stopped
    for t0, t1, v in tl.stopped:
        add("non_live", t0, t1, labels=["clock_stopped"], boundary_source=["game_clock_stopped"], event_confidence=0.8,
            evidence=[dict(kind="game_clock_stopped", game_clock_s=v, start_uncertainty_s=1.0)])
    # non-live: scoreboard hidden for >= 1 s (replays / graphics usually hide it)
    rd = tl.readings
    hid = rd[~rd.scoreboard_visible]
    if len(hid):
        grp = (hid.sample_id.diff() != 1).cumsum()
        for _, g in hid.groupby(grp):
            if g.timestamp_sec.iloc[-1] - g.timestamp_sec.iloc[0] >= 1.0:
                add("non_live", float(g.timestamp_sec.iloc[0]), float(g.timestamp_sec.iloc[-1]), labels=["scoreboard_hidden"],
                    boundary_source=["scoreboard_presence"], event_confidence=0.5, review_status="needs_review",
                    evidence=[dict(kind="scoreboard_hidden", note="replay, graphic or break likely; verify")])
    # point events
    for s in tl.score_changes:
        e = add("score_change", s["t"], s["t"], labels=[f"+{int(s['delta'])}"], boundary_source=["score_change"],
                event_confidence=0.9 if s["plausible"] else 0.3,
                review_status="auto" if s["plausible"] else "needs_review",
                offense_team_id=tid[s["team"]], defense_team_id=other[tid[s["team"]]], team_source="inferred",
                evidence=[dict(kind="score_change", team=tid[s["team"]], before=int(s["before"]), after=int(s["after"]),
                               last_seen_before_s=round(s["t_prev_last"], 3),
                               note="scoreboard graphic; the basket itself happened up to ~3 s earlier")])
    for r in tl.shot_resets:
        add("shot_clock_reset", r["t"], r["t"], labels=[f"to_{int(r['after'])}"], boundary_source=["shot_clock_reset"],
            event_confidence=0.6 if r["after"] == 24 else 0.3,
            evidence=[dict(kind="shot_clock_reset", before=int(r["before"]), after=int(r["after"]),
                           last_seen_before_s=round(r["t_prev_last"], 3))])

    # ---- possession boundaries
    bounds = []
    used_scores = set()
    for r in tl.shot_resets:
        if r["after"] != 24:
            continue
        sc = [k for k, s in enumerate(tl.score_changes) if k not in used_scores and s["plausible"]
              and 0 <= s["t"] - r["t"] <= make_window_s]
        if sc:
            s = tl.score_changes[sc[0]]
            used_scores.add(sc[0])
            bounds.append(dict(t=r["t"], conf=0.9, sources=["shot_clock_reset", "score_change"], scorer=tid[s["team"]],
                               evidence=[f"shot clock {r['before']}->24", f"{tid[s['team']]} +{int(s['delta'])} shown at {s['t']:.2f}s"]))
        else:
            bounds.append(dict(t=r["t"], conf=0.6, sources=["shot_clock_reset"], scorer=None,
                               evidence=[f"shot clock {r['before']}->24, no score change"]))
    for k, s in enumerate(tl.score_changes):
        if k in used_scores or not s["plausible"]:
            continue
        bounds.append(dict(t=s["t"], conf=0.75, sources=["score_change"], scorer=tid[s["team"]],
                           evidence=[f"{tid[s['team']]} +{int(s['delta'])} (no shot-clock reset before it; e.g. free throw)"]))
    bounds.sort(key=lambda b: b["t"])
    # merge near-duplicates (< 1 s apart)
    merged = []
    for b in bounds:
        if merged and b["t"] - merged[-1]["t"] < 1.0:
            m = merged[-1]
            m["conf"] = max(m["conf"], b["conf"])
            m["sources"] = sorted(set(m["sources"]) | set(b["sources"]))
            m["evidence"] += b["evidence"]
            m["scorer"] = m["scorer"] or b["scorer"]
        else:
            merged.append(dict(b))
    bounds = merged

    # ---- candidate possessions between boundaries
    edges = [dict(t=float(frames.timestamp_sec.iloc[0]), conf=0.0, sources=["video_start"], scorer=None, evidence=[])] \
        + bounds + [dict(t=t_end, conf=0.0, sources=["video_end"], scorer=None, evidence=[])]
    poss = []
    for a, b in zip(edges, edges[1:]):
        if b["t"] - a["t"] < 0.5:
            continue
        cand = []
        if b["scorer"]:
            cand.append((b["scorer"], "possession ended with this team's score"))
        if a["scorer"]:
            cand.append((other[a["scorer"]], "previous possession ended with the other team's score"))
        teams_c = {c[0] for c in cand}
        off = teams_c.pop() if len(teams_c) == 1 else None
        poss.append(dict(a=a, b=b, off=off, why=[c[1] for c in cand], conflict=len({c[0] for c in cand}) > 1))
    # alternation for still-unknown possessions between known neighbours (boundaries conf >= 0.6)
    for k, p in enumerate(poss):
        if p["off"] is None and not p["conflict"]:
            prev = poss[k - 1] if k > 0 else None
            nxt = poss[k + 1] if k + 1 < len(poss) else None
            g = []
            if prev and prev["off"] and p["a"]["conf"] >= 0.6:
                g.append(other[prev["off"]])
            if nxt and nxt["off"] and p["b"]["conf"] >= 0.6 and nxt["why"] and "previous" not in nxt["why"][0]:
                g.append(other[nxt["off"]])
            if len(set(g)) == 1:
                p["off"], p["alt"] = g[0], True
    for p in poss:
        a, b = p["a"], p["b"]
        conf = round(min(a["conf"] if a["conf"] else 0.5, b["conf"] if b["conf"] else 0.5), 2)
        truncated = "video_start" in a["sources"] or "video_end" in b["sources"]
        needs = conf < 0.7 or p["off"] is None or p["conflict"] or truncated
        labels = ["truncated"] if truncated else []
        add("candidate_possession", a["t"], b["t"], labels=labels,
            boundary_source=sorted(set(a["sources"]) | set(b["sources"])),
            event_confidence=conf, review_status="needs_review" if needs else "auto",
            offense_team_id=p["off"], defense_team_id=other[p["off"]] if p["off"] else None,
            team_source="inferred" if p["off"] else None,
            evidence=[dict(kind="start_boundary", t=round(a["t"], 3), confidence=a["conf"], sources=a["sources"], details=a["evidence"]),
                      dict(kind="end_boundary", t=round(b["t"], 3), confidence=b["conf"], sources=b["sources"], details=b["evidence"]),
                      dict(kind="offense_inference", team=p["off"], reasons=p["why"] + (["alternation"] if p.get("alt") else []),
                           conflict=p["conflict"])])
    ev.sort(key=lambda e: (e["video_start_seconds"], e["event_type"]))
    return ev, bounds
