"""Keeping a season up to date: nightly refits and an out-of-sample game ledger.

``update(season)`` is meant to run once a day (cron, a scheduled task):

1. Fetch the latest box scores and schedule (``data.fetch``).
2. Score every newly completed game against the posterior from *before* it
   was played: the pre-game win probability and predicted margin, how it
   turned out, and how unusual the whole box score was
   (``evaluate.game_unusualness``).  Rows are appended to
   ``artifacts/<season>/ledger.csv``, so every score is out of sample.
3. Refit.  The previous season's posterior is the preseason prior
   (``model.fit(prior=...)``); before any games are played the model is just
   that prior (``model.preseason_posterior``).  A mean-field VI refit takes a
   few minutes; NUTS (``refit="nuts"``) takes about an hour.

Upcoming games get pre-game forecasts on demand from the latest posterior.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

from . import data, model
from .evaluate import game_unusualness
from .simulate import estimate_lead_beta

LEDGER_COLUMNS = ["game_id", "date", "home", "away", "neutral", "kind", "home_pts", "away_pts",
                  "p_home", "pred_margin", "margin_percentile", "unusualness", "drivers", "model",
                  "espn_p_home", "espn_checked", "margin_lo", "margin_hi", "line_margin", "line_checked",
                  "calibration_logged", "calibration", "restated",
                  "p_home_logged", "pred_margin_logged", "margin_lo_logged", "margin_hi_logged"]
# Forecast columns that change with the simulator's calibration (see restate_ledger).
CALIBRATED_COLUMNS = ("p_home", "pred_margin", "margin_lo", "margin_hi")


NO_RESULT_AFTER = pd.Timedelta(hours=6)    # a game this long past tip-off with no result was probably not played


def ledger_swapped(lr, home: str, away: str) -> bool:
    """Whether a ledger row lists the game's teams the other way round from the schedule's home/away.
    (Older rows name a non-D1 school as the pooled "Non-D1", so one matching name is enough.)"""
    return (lr.home == away or lr.away == home) and not (lr.home == home or lr.away == away)


def schedule_with_results(season: int) -> pd.DataFrame:
    """The season schedule with every result already known filled in.

    The schedule file is refreshed by the nightly update, but games go final, are scored and fold into the
    ratings within minutes, through the live feed or the game archive. So a game the schedule still lists as
    unplayed is marked completed, with its score, when the ledger has scored it or the live tracker has a
    final for it. ``result_source`` says where each result came from; ``no_result`` marks games well past
    tip-off with no result anywhere (postponed or cancelled, most likely)."""
    from .tracker import load_track

    sched = data.load_schedule(season)
    if not len(sched):
        return sched
    sched = sched.copy()
    sched["result_source"] = np.where(sched.completed, "schedule", "")
    led = load_ledger(season)
    by_id = {int(r.game_id): r for r in led.itertuples()} if len(led) else {}
    for i in sched.index[~sched.completed]:
        r = sched.loc[i]
        gid = int(r.game_id)
        score = None
        lr = by_id.get(gid)
        if lr is not None:
            swapped = ledger_swapped(lr, r.home, r.away)
            score = (int(lr.away_pts), int(lr.home_pts)) if swapped else (int(lr.home_pts), int(lr.away_pts))
            src = "ledger"
        else:
            tr = load_track(season, gid)
            if tr and tr.get("status") == "post" and tr.get("points"):
                s = tr["points"][-1]["score"]
                if tr.get("teams") and tr["teams"][0] == r.away:
                    s = s[::-1]
                score, src = (int(s[0]), int(s[1])), "live tracker"
        if score is not None:
            sched.loc[i, ["completed", "home_score", "away_score", "result_source"]] = [True, score[0], score[1], src]
    now = pd.Timestamp.now(tz="UTC")
    sched["no_result"] = ~sched.completed & (sched.date + NO_RESULT_AFTER < now)
    return sched


def ledger_path(season: int):
    return model.artifact_dir(season) / "ledger.csv"


def load_ledger(season: int) -> pd.DataFrame:
    path = ledger_path(season)
    if not path.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    led = pd.read_csv(path)
    # Rows come from several sources (live feed, game archive, nightly box scores) whose timestamps are written
    # in slightly different formats; parse them all, as UTC.
    led["date"] = pd.to_datetime(led["date"], utc=True, format="mixed")
    if "espn_p_home" not in led:                 # ledgers written before ESPN's forecast was recorded
        led["espn_p_home"] = np.nan
    if "espn_checked" not in led:
        led["espn_checked"] = False
    led["espn_checked"] = led.espn_checked.fillna(False).astype(bool)
    for c in ("margin_lo", "margin_hi", "line_margin"):
        if c not in led:
            led[c] = np.nan
    led["line_checked"] = (led.line_checked if "line_checked" in led else pd.Series(False, index=led.index)).fillna(False).astype(bool)
    for c in ("calibration_logged", "calibration") + tuple(f"{c}_logged" for c in CALIBRATED_COLUMNS):
        if c not in led:
            led[c] = np.nan
    led["calibration_logged"] = led.calibration_logged.fillna(1.0)   # rows from before calibration: the raw model
    led["calibration"] = led.calibration.fillna(led.calibration_logged)
    led["restated"] = (led.restated if "restated" in led else pd.Series(False, index=led.index)).fillna(False).astype(bool)
    return led


def as_logged(led: pd.DataFrame) -> pd.DataFrame:
    """The ledger with every forecast as it was first written, before any restatement."""
    led = led.copy()
    for c in CALIBRATED_COLUMNS:
        led[c] = led[f"{c}_logged"].where(led.restated, led[c])
    return led


def calibration_map(post: model.Posterior, k_from: float, k_to: float, n_games: int = 300, n: int = 1000,
                    seed: int = 0, teams=None) -> dict:
    """How forecasts made under skill calibration ``k_from`` translate to ``k_to``, fitted on simulated
    matchups of this season's teams: log-odds of the first team's win chance, and the predicted margin, each a
    linear function of the old value and whether that team is at home (+1), away (-1) or on a neutral floor."""
    from .simulate import calibration, simulate_games

    rng = np.random.default_rng(seed)
    teams = set(teams or post.teams)
    d1 = [i for i, t in enumerate(post.teams) if t in teams]
    a = rng.choice(d1, n_games); b = rng.choice(d1, n_games)
    keep = a != b; a, b = a[keep], b[keep]
    h = rng.choice([-1, 0, 1], len(a))
    k = rng.integers(post.n_draws, size=len(a) * n)
    out = {}
    for name, cal in (("from", k_from), ("to", k_to)):
        with calibration(cal, 1.0):
            r = simulate_games(post, k, np.repeat(a, n), np.repeat(b, n), np.random.default_rng(seed + 1),
                               home=np.repeat(h, n), full_box=False)
        p = np.clip(r["a_wins"].reshape(-1, n).mean(1), 1 / n, 1 - 1 / n)
        out[name] = (np.log(p / (1 - p)), (r["pts"][:, 0] - r["pts"][:, 1]).reshape(-1, n).mean(1))
    X = np.c_[out["from"][0], h]
    lc = np.linalg.lstsq(X, out["to"][0], rcond=None)[0]
    Xm = np.c_[out["from"][1], h]
    mc = np.linalg.lstsq(Xm, out["to"][1], rcond=None)[0]
    return {"logit": lc.tolist(), "margin": mc.tolist(), "k_from": k_from, "k_to": k_to}


def restate_ledger(season: int, log=print) -> int:
    """Put every forecast in the ledger on the simulator's current calibration.

    A forecast is written once, before its game is scored, and never recomputed: that is what keeps the ledger
    out of sample. When the calibration changes, forecasts written under the old one are restated: calibration
    changes no team's rating, only how a matchup becomes a win chance and margin, so each logged forecast is
    translated with a map fitted on simulated matchups (``calibration_map``). The original numbers stay in the
    ``*_logged`` columns, and ``restated`` marks the rows. Returns the number of rows restated."""
    from .simulate import CAL

    path = ledger_path(season)
    led = load_ledger(season)
    k_to = float(CAL["skill_scale"])
    todo = led[(led.calibration - k_to).abs() > 1e-9]
    if not len(todo):
        return 0
    post, _ = current_posterior(season)
    if post is None:
        return 0
    sched = schedule_with_results(season).set_index("game_id")
    for c in CALIBRATED_COLUMNS:                                       # keep the first-written numbers
        first = ~led.restated
        led.loc[first, f"{c}_logged"] = led.loc[first, c]
    maps = {k0: calibration_map(post, float(k0), k_to, teams=data.d1_set(season)) for k0 in todo.calibration_logged.unique()}
    for i in todo.index:
        row = led.loc[i]
        k0 = float(row.calibration_logged)
        if abs(k0 - k_to) < 1e-9:                                      # back to the calibration it was made under
            for c in CALIBRATED_COLUMNS:
                led.loc[i, c] = row[f"{c}_logged"]
            led.loc[i, ["calibration", "restated"]] = [k_to, False]
            continue
        m = maps[k0]
        h = 0
        if not bool(row.neutral) and int(row.game_id) in sched.index:
            g = sched.loc[int(row.game_id)]
            g = g.iloc[0] if isinstance(g, pd.DataFrame) else g
            h = -1 if ledger_swapped(row, g.home, g.away) else 1
        p0 = float(row.p_home_logged)
        if p0 == p0:
            lo = np.log(np.clip(p0, 1e-4, 1 - 1e-4) / (1 - np.clip(p0, 1e-4, 1 - 1e-4)))
            led.loc[i, "p_home"] = float(1 / (1 + np.exp(-(m["logit"][0] * lo + m["logit"][1] * h))))
        mg = float(row.pred_margin_logged)
        if mg == mg:
            new = m["margin"][0] * mg + m["margin"][1] * h
            led.loc[i, "pred_margin"] = new
            for c in ("margin_lo", "margin_hi"):                       # the same spread around the new centre
                v = float(row[f"{c}_logged"])
                if v == v:
                    led.loc[i, c] = v + (new - mg)
        led.loc[i, ["calibration", "restated"]] = [k_to, True]
    path.parent.mkdir(parents=True, exist_ok=True)
    led.to_csv(path, index=False)
    log(f"restated {len(todo)} ledger forecasts on the current calibration (skill x{k_to})")
    return len(todo)


def _espn_for_row(row, raw: dict) -> float:
    """ESPN's pre-game P(first-listed ledger team wins), or NaN."""
    from .live import espn_pregame

    e = espn_pregame(raw)
    if e is None:
        return np.nan
    if row.home == e["home"] or row.away == e["away"]:
        return e["p_home"]
    if row.home == e["away"] or row.away == e["home"]:
        return 1 - e["p_home"]
    return np.nan                                  # names don't line up (e.g. a pooled non-D1 opponent)


def _line_for_row(row, raw: dict) -> float:
    """The betting line's expected margin for the first-listed ledger team, or NaN."""
    from .live import betting_line

    ln = betting_line(raw)
    if ln is None:
        return np.nan
    if row.home == ln["home"] or row.away == ln["away"]:
        return ln["home_margin"]
    if row.home == ln["away"] or row.away == ln["home"]:
        return -ln["home_margin"]
    return np.nan


def fill_espn(season: int, feed=None, raws: dict | None = None, log=print) -> int:
    """Record ESPN's pre-game win probability next to ours for every ledger game that doesn't have it.

    Summaries come from ``raws`` (already in hand), the tracker's stored copies, or, with a ``feed``, a
    fetch (the archive has every finished game). Each game is looked up once; games ESPN published no
    forecast for stay blank."""
    from .tracker import load_summary

    with season_lock(season):
        led = load_ledger(season)
        todo = led.index[~led.espn_checked | ~led.line_checked]
        filled = 0
        for i in todo:
            row = led.loc[i]
            gid = int(row.game_id)
            raw = (raws or {}).get(gid) or load_summary(season, gid)
            if raw is None and feed is not None:
                try:
                    raw = feed.summary(gid)
                except Exception:
                    continue                       # try again next time
            if raw is None:
                continue
            led.loc[i, "espn_p_home"] = _espn_for_row(row, raw)
            led.loc[i, "espn_checked"] = True
            led.loc[i, "line_margin"] = _line_for_row(row, raw)        # the betting line ESPN shows, if any
            led.loc[i, "line_checked"] = True
            filled += int(not np.isnan(led.loc[i, "espn_p_home"]))
        if len(todo):
            led.to_csv(ledger_path(season), index=False)
    if filled:
        log(f"recorded ESPN's pre-game forecast for {filled} games")
    return filled


def base_posterior(season: int) -> tuple[model.Posterior | None, str]:
    """This season's latest full fit, else a preseason posterior from last season's, else None."""
    if (model.artifact_dir(season) / "posterior.npz").exists():
        return model.load_posterior(season), f"{season} fit"
    if (model.artifact_dir(season - 1) / "posterior.npz").exists():
        prev = model.load_posterior(season - 1)
        teams = sorted(set(data.d1_teams(season)) | ({"Non-D1"} & set(prev.teams))) or None
        return model.preseason_posterior(prev, teams), f"{season} preseason (from {season - 1})"
    return None, "none"


def _prior_dir(season: int):
    """The artifacts holding the population skill spread (this season's fit, else last season's)."""
    here = model.artifact_dir(season)
    return here if (here / "posterior.npz").exists() else model.artifact_dir(season - 1)


def current_state(season: int, create: bool = True):
    """The running per-team beliefs, started from the base posterior and covering every D1 team."""
    from .updates import init_state, load_state, population_cov, save_state

    st = load_state(season)
    if st is not None or not create:
        return st
    base, label = base_posterior(season)
    if base is None:
        return None
    st = init_state(base, label, teams=data.d1_teams(season), prior_cov=population_cov(_prior_dir(season)))
    fitted = data.load_games(season)
    if "fit" in label and len(fitted):         # a full fit has seen the games it was fit to (not the tournament)
        st.applied = [int(g) for g in fitted.game_id.unique()]
    st.base_count = len(st.applied)
    save_state(season, st)
    return st


def current_posterior(season: int) -> tuple[model.Posterior | None, str]:
    """The model as of now: the base posterior with every game since folded in by Bayesian updates."""
    from .updates import state_posterior

    base, label = base_posterior(season)
    if base is None:
        return None, "none"
    st = current_state(season, create=False)
    n = len(st.applied) - st.base_count if st is not None else 0
    if n <= 0:                              # nothing folded in since the base: use it as is
        return base, label
    return state_posterior(st, base), f"{label}, updated after {n} games"


def load_updates(season: int):
    from .updates import load_updates as _lu

    return _lu(season)


def _completed_games(season: int) -> pd.DataFrame:
    """One row per completed game, home team first (neutral games: as listed)."""
    frames = [data.load_games(season).assign(kind="regular season")]
    t = data.load_tournament_games(season)
    if len(t):
        frames.append(t.assign(kind="tournament"))
    g = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) else pd.DataFrame()
    if not len(g):
        return g
    return g[(g.home >= 0)].sort_values(["date", "game_id", "home"], ascending=[True, True, False]) \
        .drop_duplicates("game_id")


def _pair_posterior(st, base, a: str, b: str, seed: int):
    """Current beliefs about just two teams, as a Posterior (fast forecasts for one game)."""
    import dataclasses

    from .updates import state_posterior

    ia, ib = st.index(a), st.index(b)
    sub = dataclasses.replace(st, teams=[a, b], mean=st.mean[[ia, ib]], cov=st.cov[[ia, ib]])
    return state_posterior(sub, base, seed=seed)


def _canonical_pooled(team: str, st) -> str | None:
    """A non-D1 school's name in the two-sided update: the pooled pseudo-team."""
    return data.OTHER_TEAM if data.OTHER_TEAM in st.teams else (team if team in st.teams else None)


def _canonical(team: str, st) -> str | None:
    """The model's name for a team (non-D1 opponents share one pseudo-team)."""
    if team in st.teams:
        return team
    return "Non-D1" if "Non-D1" in st.teams else None


@contextmanager
def season_lock(season: int):
    """Serialise writes to a season's ratings across processes (web app, watcher, nightly job)."""
    from .locks import file_lock

    with file_lock(model.artifact_dir(season) / ".lock"):
        yield


def process_final(season: int, games: pd.DataFrame, source: str, n: int = 2000, log=print) -> int:
    """For each finished game not yet seen, in date order: score it in the ledger against the
    beliefs held *before* it, then fold it into those beliefs.  Returns how many were applied."""
    with season_lock(season):          # re-read the state inside the lock so no game is applied twice
        return _process_final(season, games, source, n, log)


def _process_final(season: int, games: pd.DataFrame, source: str, n: int, log) -> int:
    from .updates import apply_game, apply_one_sided, minutes_played, population_cov, save_state

    st = current_state(season)
    base, label = base_posterior(season)
    if st is None or base is None or not len(games):
        return 0
    prior_cov = population_cov(_prior_dir(season))
    ledger = load_ledger(season)
    seen = set(st.applied) | set(ledger.game_id.astype(int)) if len(ledger) else set(st.applied)
    new = games[~games.game_id.isin(st.applied)].sort_values("date")
    d1 = set(data.d1_teams(season))
    rows, applied = [], 0
    for k, r in enumerate(new.itertuples()):
        # The two-sided update always uses the pooled non-D1 team (a school's own rating is updated separately).
        name = lambda t: t if (t in d1 if d1 else t in st.teams) else _canonical_pooled(t, st)
        a, b = name(r.team), name(r.opponent)
        if a is None or b is None or a == b:
            continue
        for t in (a, b):
            if t not in st.teams:
                st.add_team(t, prior_cov)
        home = int(r.home)
        if int(r.game_id) not in seen:
            # Forecast with the best pre-game belief: a school's own rating once it has one.
            fa = r.team if a == data.OTHER_TEAM and r.team in st.teams else a
            fb = r.opponent if b == data.OTHER_TEAM and r.opponent in st.teams else b
            pair = _pair_posterior(st, base, fa, fb, seed=k)
            u = game_unusualness(pair, 0, 1, home, r, n=n, seed=k)
            p_first = u["p_winner"] if r.pts > r.opp_pts else 1 - u["p_winner"]
            show_a = r.team if a == data.OTHER_TEAM else a            # the school's own name, not the pool
            show_b = r.opponent if b == data.OTHER_TEAM else b
            rows.append({"game_id": int(r.game_id), "date": r.date, "home": show_a, "away": show_b, "neutral": home == 0,
                         "kind": getattr(r, "kind", "regular season"), "home_pts": int(r.pts), "away_pts": int(r.opp_pts),
                         "p_home": p_first, "pred_margin": u["pred_margin"], "margin_lo": u["margin_lo"],
                         "margin_hi": u["margin_hi"], "margin_percentile": u["margin_percentile"],
                         "unusualness": u["p_value"],
                         "drivers": "; ".join(f"{[a, b][d['team'] == 'b']} {d['stat']}" for d in u["drivers"]
                                              if d["share"] > 0.1),
                         "model": f"{label} + updates", **_calibration_tag()})
        box_a = {s_: int(getattr(r, s_)) for s_ in LIVE_BOX}
        box_b = {s_: int(getattr(r, "opp_" + s_)) for s_ in LIVE_BOX}
        # A named non-D1 school also gets its own rating, updated against the D1 team's pre-game belief.
        for school, opp, h, bs, bo in ((r.team, b, home, box_a, box_b), (r.opponent, a, -home, box_b, box_a)):
            if school not in d1 and school != data.OTHER_TEAM and opp in d1:
                try:
                    apply_one_sided(st, base, school, opp, h, bs, bo, minutes_played(getattr(r, "ot", 0)), prior_cov)
                except Exception as e:     # never block the D1 update over it
                    log(f"could not update {school}'s own rating: {e}")
        apply_game(season, st, base, int(r.game_id), r.date, a, b, home, box_a, box_b,
                   minutes_played(getattr(r, "ot", 0)), prior_cov, source)
        applied += 1
    if rows:
        ledger = pd.concat([ledger, pd.DataFrame(rows)], ignore_index=True)
        ledger_path(season).parent.mkdir(parents=True, exist_ok=True)
        ledger.to_csv(ledger_path(season), index=False)
    if applied:
        save_state(season, st)
        record_ratings(season)
        log(f"applied Bayesian updates for {applied} finished games ({source})")
    return applied


LIVE_BOX = ["pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "tov", "ast", "stl", "blk", "pf"]


def live_final_row(season: int, state) -> pd.DataFrame:
    """A finished game from the live feed, in the same shape as the nightly box-score rows."""
    sched = data.load_schedule(season)
    row = sched[sched.game_id == state.game_id]
    date = row.date.iloc[0] if len(row) else pd.Timestamp.now(tz="UTC")
    h, a = state.teams
    rec = {"game_id": state.game_id, "date": date, "team": h, "opponent": a, "home": 0 if state.neutral else 1,
           "ot": max(state.period - 2, 0), "kind": "regular season"}
    for s_ in LIVE_BOX:
        rec[s_], rec["opp_" + s_] = state.box[h][s_], state.box[a][s_]
    return pd.DataFrame([rec])


def apply_live_final(season: int, state, log=print) -> int:
    """Fold a game into the ratings the moment the live feed reports it final."""
    if state.status != "post" or not state.box:
        return 0
    return process_final(season, live_final_row(season, state), source="live feed", log=log)


def apply_archived_finals(season: int, log=print, feed=None) -> int:
    """Finished games the schedule knows about but the nightly box scores don't have yet:
    take each game's final box score from the per-game ESPN archive (which updates sooner)."""
    from .live import ReplayFeed

    sched = data.load_schedule(season)
    st = current_state(season)
    if not len(sched) or st is None:
        return 0
    done = set(st.applied)
    todo = sched[sched.completed & ~sched.game_id.isin(done)].sort_values("date")
    feed = feed or ReplayFeed()
    n = 0
    for gid in todo.game_id:
        try:
            final = feed.game(int(gid), 999)
        except Exception:            # not archived yet: the next run will pick it up
            continue
        if final.status == "post" and final.box and all(final.box[t]["fga"] for t in final.teams):
            n += process_final(season, live_final_row(season, final), source="game archive", log=lambda *_: None)
    if n:
        log(f"applied {n} finished games from the game archive")
    return n


def watch(season: int, every: int = 60, log=print, feed=None, rounds: int | None = None) -> None:
    """Follow every game from its scheduled tip-off: keep its win-probability history on disk and fold it
    into the ratings the moment it goes final (see ``ncaa.tracker``)."""
    from .live import FallbackFeed
    from .tracker import run

    run(season, feed or FallbackFeed(), every=every, log=log, rounds=rounds)


def score_new_games(season: int, post: model.Posterior, label: str, n: int = 2000) -> pd.DataFrame:
    """Append out-of-sample scores for completed games not yet in the ledger."""
    ledger = load_ledger(season)
    games = _completed_games(season)
    if not len(games):
        return ledger
    idx = {t: i for i, t in enumerate(post.teams)}
    new = games[~games.game_id.isin(ledger.game_id) & games.team.isin(idx) & games.opponent.isin(idx)]
    rows = []
    for k, r in enumerate(new.itertuples()):
        home = int(r.home)  # +1 when r.team is at home, 0 neutral
        u = game_unusualness(post, idx[r.team], idx[r.opponent], home, r, n=n, seed=k)
        # p_winner refers to whoever actually won; turn it into P(first-listed team wins).
        p_first = u["p_winner"] if r.pts > r.opp_pts else 1 - u["p_winner"]
        rows.append({"game_id": int(r.game_id), "date": r.date, "home": r.team, "away": r.opponent,
                     "neutral": home == 0, "kind": r.kind, "home_pts": int(r.pts), "away_pts": int(r.opp_pts),
                     "p_home": p_first, "pred_margin": np.nan, "margin_percentile": u["margin_percentile"],
                     "unusualness": u["p_value"],
                     "drivers": "; ".join(f"{[r.team, r.opponent][d['team'] == 'b']} {d['stat']}"
                                          for d in u["drivers"] if d["share"] > 0.1),
                     "model": label, **_calibration_tag()})
    if rows:
        ledger = pd.concat([ledger, pd.DataFrame(rows)], ignore_index=True)
        ledger_path(season).parent.mkdir(parents=True, exist_ok=True)
        ledger.to_csv(ledger_path(season), index=False)
    return ledger


def _calibration_tag() -> dict:
    """Which calibration a forecast is written under (see restate_ledger)."""
    from .simulate import CAL

    k = float(CAL["skill_scale"])
    return {"calibration_logged": k, "calibration": k, "restated": False}


def update(season: int, refit: str = "vi", fetch: bool = True, log=print) -> dict:
    """The nightly job: fetch, score new games out of sample, refit."""
    t0 = time.time()
    if fetch:
        games, bracket = data.fetch(season)
        log(f"fetched {season}: {games.game_id.nunique() if len(games) else 0} games so far"
            f"{'; tournament field set' if bracket else ''}")
    base, label = base_posterior(season)
    if base is None:
        raise RuntimeError(f"no posterior for {season} or {season - 1}: run `python -m ncaa build --season {season - 1}`")
    before = len(load_ledger(season))
    process_final(season, _completed_games(season), source="nightly box scores", log=log)
    apply_archived_finals(season, log=log)
    try:
        from .live import FallbackFeed

        fill_espn(season, feed=FallbackFeed(), log=log)
    except Exception as e:                         # never fail the nightly update over the comparison
        log(f"could not record ESPN forecasts: {e}")
    try:
        restate_ledger(season, log=log)
    except Exception as e:                                             # never block the nightly update over it
        log(f"could not restate the ledger: {e}")
    ledger = load_ledger(season)
    log(f"scored {len(ledger) - before} new games out of sample")

    games = data.load_games(season)
    status = {"season": season, "updated": pd.Timestamp.now(tz="UTC").isoformat(), "games": int(games.game_id.nunique()) if len(games) else 0,
              "scored": len(ledger), "posterior": current_posterior(season)[1]}
    if refit != "none" and len(games) and games.game_id.nunique() >= 25:
        prev = model.load_posterior(season - 1) if (model.artifact_dir(season - 1) / "posterior.npz").exists() else None
        model.fit(games, season, method=refit, prior=prev, progress=False)
        fitted = model.load_posterior(season)
        if games.game_id.nunique() >= 300:
            beta = estimate_lead_beta(games, fitted)
        else:  # too few games to estimate lead erosion: carry last season's over
            beta = prev.lead_beta if prev is not None else np.zeros(fitted.n_draws)
            beta = np.resize(beta, fitted.n_draws)
        model.add_to_posterior(season, lead_beta=beta)
        status["posterior"] = f"{season} fit"
        log(f"refit {season} with {refit}")
        # The fit has seen every game so far: restart the running beliefs from it.
        from .updates import load_state, save_state, state_path
        with season_lock(season):
            old = load_state(season)
            state_path(season).unlink(missing_ok=True)
            st = current_state(season)
            # The fit pools non-D1 opponents into one team; keep each school's own rating from before it.
            d1 = set(data.d1_teams(season))
            if old is not None and st is not None:
                kept = 0
                for t in old.teams:
                    if t not in st.teams and t not in d1 and t != data.OTHER_TEAM:
                        i = st.add_team(t, old.cov[old.index(t)])
                        st.mean[i] = old.mean[old.index(t)]
                        kept += 1
                if kept:
                    save_state(season, st)
    record_ratings(season)
    status["seconds"] = round(time.time() - t0)
    (model.artifact_dir(season)).mkdir(parents=True, exist_ok=True)
    (model.artifact_dir(season) / "status.json").write_text(json.dumps(status, indent=1))
    return status


def ratings_history_path(season: int):
    return model.artifact_dir(season) / "ratings_history.csv"


def record_ratings(season: int) -> None:
    """Append today's ratings for every team, so team pages can show how the assessment moves."""
    from .ratings import team_ratings

    post, label = current_posterior(season)
    if post is None:
        return
    today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    d1 = set(data.d1_teams(season))
    rated = [r for r in team_ratings(post) if r["team"] != data.OTHER_TEAM and (not d1 or r["team"] in d1)]
    rated.sort(key=lambda r: -r["net"])
    rows = pd.DataFrame([{"date": today, "team": r["team"], "net": r["net"], "net_lo": r["net_lo"], "net_hi": r["net_hi"],
                          "ortg": r["ortg"], "drtg": r["drtg"], "rank": k, "model": label}
                         for k, r in enumerate(rated, 1)])
    path = ratings_history_path(season)
    if path.exists():
        old = pd.read_csv(path)
        rows = pd.concat([old[old.date != today], rows], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, index=False)


def load_ratings_history(season: int) -> pd.DataFrame:
    path = ratings_history_path(season)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def upcoming(season: int, post: model.Posterior, start=None, days: int = 1, n: int = 1500, game_ids=None) -> list[dict]:
    """Pre-game forecasts for scheduled games in a date window (only ``game_ids``, when given)."""
    from .simulate import simulate_games

    sched = schedule_with_results(season)
    if not len(sched):
        return []
    start = pd.Timestamp.now(tz="UTC").normalize() if start is None else pd.Timestamp(start, tz="UTC")
    win = sched[(sched.date >= start) & (sched.date < start + pd.Timedelta(days=days))]
    if game_ids is not None:
        win = win[win.game_id.isin(set(int(g) for g in game_ids))]
    post = model.with_teams(post, set(win.home) | set(win.away), data.d1_set(season) or None)  # group average
    idx = {t: i for i, t in enumerate(post.teams)}
    out = []
    rng = np.random.default_rng(0)
    for r in win.itertuples():
        row = {"game_id": int(r.game_id), "date": r.date.isoformat(), "home": r.home, "away": r.away,
               "neutral": bool(r.neutral), "completed": bool(r.completed), "no_result": bool(r.no_result)}
        if r.completed:
            row["score"] = [int(r.home_score), int(r.away_score)]
        if r.home in idx and r.away in idx:
            res = simulate_games(post, rng.integers(post.n_draws, size=n), np.full(n, idx[r.home]),
                                 np.full(n, idx[r.away]), rng, home=0 if r.neutral else 1, full_box=False)
            m = res["pts"][:, 0] - res["pts"][:, 1]
            row |= {"p_home": float(res["a_wins"].mean()), "pred_margin": float(m.mean()),
                    "pred_score": res["pts"].mean(0).round(1).tolist()}
        out.append(row)
    return out
