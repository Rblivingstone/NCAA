"""Server-side live tracking: every game is followed from its scheduled tip-off, whether or not anyone is watching.

The worker's ``ncaa watch`` loop picks up each game at its scheduled start time (from the season schedule,
cross-checked with ESPN's scoreboard for delays and status), polls ESPN's game summary until it is final,
and keeps a win-probability history per game on disk:

    artifacts/<season>/live/<game id>.json          status, latest state, history points
    artifacts/<season>/live/<game id>.summary.json.gz  the latest raw ESPN summary (plays so far)

Every poll adds a point. Gaps (the first poll after tip-off, a worker restart, a feed outage) are filled by
reconstructing the game minute by minute from the play-by-play, so the history is complete whenever it is
first looked at, and stays on the game page after the final. When a game goes final its history is finished
(with the beliefs from before the game), and then the game is folded into the team ratings.

Each history point is {"elapsed", "ours" (model P(home wins)), "espn" (ESPN's P(home)), "score", "q"
(per-stat shift probabilities vs the pre-game forecast)}, the format the game page draws.
"""
from __future__ import annotations

import gzip
import json
import os
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

from . import data, model
from .live import last_play_elapsed, parse_summary

POLL_BEFORE = pd.Timedelta(minutes=10)   # start polling this long before the scheduled tip
GIVE_UP = pd.Timedelta(hours=8)          # stop chasing a game this long after its scheduled tip
GRID = 1.0                               # minutes between reconstructed history points
STALE = 150                              # seconds after which a live track is refreshed on demand
BACKFILL_PER_ROUND = 4                   # finished games reconstructed per round (live games always come first)


# --- storage ------------------------------------------------------------------

def track_dir(season: int):
    return model.artifact_dir(season) / "live"


def _paths(season: int, game_id: int):
    d = track_dir(season)
    return d / f"{int(game_id)}.json", d / f"{int(game_id)}.summary.json.gz"


def _write(path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)                # readers never see a half-written file


def load_track(season: int, game_id: int) -> dict | None:
    path, _ = _paths(season, game_id)
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return None


def load_summary(season: int, game_id: int) -> dict | None:
    _, path = _paths(season, game_id)
    try:
        return json.loads(gzip.decompress(path.read_bytes()))
    except (FileNotFoundError, ValueError, OSError):
        return None


def is_stale(track: dict | None, now: float | None = None) -> bool:
    return track is None or (track.get("status") != "post" and (now or time.time()) - track.get("updated", 0) > STALE)


@contextmanager
def _game_lock(season: int, game_id: int):
    """One writer per game at a time (the worker and a web request may both refresh it)."""
    from .locks import file_lock

    path, _ = _paths(season, game_id)
    with file_lock(path.with_name(f".{int(game_id)}.lock")):
        yield


# --- forecasting at one moment --------------------------------------------------

def _model_stamp(season: int) -> float:
    d = model.artifact_dir(season)
    return max([p.stat().st_mtime for p in (d / "posterior.npz", d / "team_state.npz") if p.exists()] or [0.0])


class Forecaster:
    """The season's current model (re-read when it changes) and the in-game forecast at any moment."""

    def __init__(self, season: int):
        self.season, self._stamp, self.post, self._pre = season, None, None, {}

    def refresh(self):
        stamp = _model_stamp(self.season)
        if stamp != self._stamp:
            from .season import current_posterior

            self.post, _ = current_posterior(self.season)
            self._stamp, self._pre = stamp, {}
        return self.post

    def point(self, state, elapsed: float | None = None) -> dict | None:
        from .ingame import forecast, pregame, shifts

        if self.post is None:
            return None
        post = model.with_teams(self.post, state.teams, data.d1_set(self.season) or None)  # no history: group average
        idx = {t: i for i, t in enumerate(post.teams)}
        a, b, home = idx[state.teams[0]], idx[state.teams[1]], 0 if state.neutral else 1
        key = (a, b, home)
        if key not in self._pre:
            self._pre[key] = pregame(post, a, b, home, n=3000, seed=0, samples=True)
        f = forecast(post, a, b, home, state, n=1500, seed=1)
        q = shifts(f["samples"], self._pre[key]["samples"])
        return {"elapsed": round(float(state.elapsed if elapsed is None else elapsed), 3),
                "ours": float(f["p_win"][0]),
                "espn": None if state.espn_home_wp is None else float(state.espn_home_wp),
                "score": [int(state.box.get(t, {}).get("pts", 0)) for t in state.teams],
                "q": {k: [float(x) for x in v] for k, v in q.items()}}


# --- one game -------------------------------------------------------------------

def update_track(season: int, game_id: int, raw: dict, fc: Forecaster) -> dict:
    """Fold the latest ESPN summary of a game into its stored history (backfilling any gaps from the plays)."""
    with _game_lock(season, game_id):
        track = load_track(season, game_id) or {"game_id": int(game_id), "points": []}
        state = parse_summary(raw)
        track.update(teams=state.teams, neutral=state.neutral, status=state.status, detail=state.detail,
                     updated=time.time())
        if state.status != "pre" and fc.refresh() is not None:
            pts = {round(p["elapsed"], 3): p for p in track["points"]}
            last = last_play_elapsed(raw)
            # Reconstruct missing minutes from the play-by-play (never past the latest play: beyond it the
            # cut-off would read as the end of the game).
            end = max(last, 2 * 20.0) if state.status == "post" else min(last, state.elapsed)
            for m in np.arange(0.0, end, GRID):
                if not any(abs(e - m) < GRID / 2 for e in pts):
                    p = fc.point(parse_summary(raw, until=float(m)))
                    if p:
                        pts[p["elapsed"]] = p
            if state.status == "in" and (not pts or state.elapsed > max(pts) + 1e-6):
                p = fc.point(state)
                if p:
                    pts[p["elapsed"]] = p
            if state.status == "post":
                final_at = max(last, 2 * 20.0)
                p = fc.point(state, elapsed=final_at)
                if p:
                    pts[p["elapsed"]] = p
            track["points"] = [pts[k] for k in sorted(pts)]
        _write(_paths(season, game_id)[1], gzip.compress(json.dumps(raw).encode()))
        _write(_paths(season, game_id)[0], json.dumps(track).encode())
        return track


def track_game(season: int, game_id: int, feed, fc: Forecaster | None = None, log=print) -> dict:
    """Poll one game now; when it is final, finish its history and fold it into the ratings."""
    from .season import apply_live_final

    fc = fc or Forecaster(season)
    raw = feed.summary(game_id)
    track = update_track(season, game_id, raw, fc)
    if track["status"] == "post" and not track.get("applied"):
        try:
            apply_live_final(season, parse_summary(raw), log=log)
            from .season import fill_espn

            fill_espn(season, raws={int(game_id): raw}, log=log)    # ESPN's pre-game forecast, for the ledger
        except Exception as e:           # the nightly update retries; never lose the history over it
            log(f"could not update ratings for {game_id}: {e}")
        finally:
            track["applied"] = True
            with _game_lock(season, game_id):
                _write(_paths(season, game_id)[0], json.dumps(track).encode())
    return track


# --- the loop --------------------------------------------------------------------

def games_to_poll(season: int, scoreboard: list[dict], now: pd.Timestamp | None = None) -> tuple[list, list]:
    """(live games, finished games still to record), by schedule start time and the scoreboard's status."""
    now = now or pd.Timestamp.now(tz="UTC")
    status = {int(g["game_id"]): g.get("status") for g in scoreboard}
    try:
        sched = data.load_schedule(season)
        due = sched[(sched.date - POLL_BEFORE <= now) & (now <= sched.date + GIVE_UP)]
        ids = set(int(g) for g in due.game_id)
    except Exception:
        ids = set()
    ids |= {g for g, s in status.items() if s in ("in", "post")}
    live, finished = [], []
    for gid in sorted(ids):
        tr = load_track(season, gid)
        if tr and tr.get("status") == "post" and tr.get("applied"):
            continue                     # done
        s = status.get(gid)
        if s == "pre":
            continue                     # not started yet (delayed), per the scoreboard
        (finished if s == "post" else live).append(gid)
    return live, finished


def run(season: int, feed, every: int = 60, log=print, rounds: int | None = None) -> None:
    fc = Forecaster(season)
    done = 0
    while rounds is None or done < rounds:
        started = time.time()
        today = pd.Timestamp.now(tz="America/New_York")
        board = []
        for day in (today - pd.Timedelta(days=1), today):
            try:
                board += feed.scoreboard(day.strftime("%Y%m%d"))
            except Exception as e:       # fall back to the schedule's start times
                log(f"scoreboard unavailable: {e}")
        live, finished = games_to_poll(season, board)
        for gid in live + finished[:BACKFILL_PER_ROUND]:
            try:
                tr = track_game(season, gid, feed, fc, log=log)
                if tr["status"] == "post":
                    log(f"final: {' vs '.join(tr.get('teams', []))} ({gid}), {len(tr['points'])} history points")
            except Exception as e:       # one bad game never stops the others
                log(f"could not track game {gid}: {e}")
        done += 1
        if rounds is None or done < rounds:
            time.sleep(max(every - (time.time() - started), 5))
