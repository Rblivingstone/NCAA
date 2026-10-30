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

import numpy as np
import pandas as pd

from . import data, model
from .evaluate import game_unusualness
from .simulate import estimate_lead_beta

LEDGER_COLUMNS = ["game_id", "date", "home", "away", "neutral", "kind", "home_pts", "away_pts",
                  "p_home", "pred_margin", "margin_percentile", "unusualness", "drivers", "model"]


def ledger_path(season: int):
    return model.artifact_dir(season) / "ledger.csv"


def load_ledger(season: int) -> pd.DataFrame:
    path = ledger_path(season)
    if not path.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    return pd.read_csv(path, parse_dates=["date"])


def current_posterior(season: int) -> tuple[model.Posterior | None, str]:
    """This season's latest fit, else a preseason posterior from last season's, else None."""
    if (model.artifact_dir(season) / "posterior.npz").exists():
        return model.load_posterior(season), f"{season} fit"
    if (model.artifact_dir(season - 1) / "posterior.npz").exists():
        prev = model.load_posterior(season - 1)
        return model.preseason_posterior(prev), f"{season} preseason (from {season - 1})"
    return None, "none"


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
                     "model": label})
    if rows:
        ledger = pd.concat([ledger, pd.DataFrame(rows)], ignore_index=True)
        ledger_path(season).parent.mkdir(parents=True, exist_ok=True)
        ledger.to_csv(ledger_path(season), index=False)
    return ledger


def update(season: int, refit: str = "vi", fetch: bool = True, log=print) -> dict:
    """The nightly job: fetch, score new games out of sample, refit."""
    t0 = time.time()
    if fetch:
        games, bracket = data.fetch(season)
        log(f"fetched {season}: {games.game_id.nunique() if len(games) else 0} games so far"
            f"{'; tournament field set' if bracket else ''}")
    post, label = current_posterior(season)
    if post is None:
        raise RuntimeError(f"no posterior for {season} or {season - 1}: run `python -m ncaa build --season {season - 1}`")
    before = len(load_ledger(season))
    ledger = score_new_games(season, post, label)
    log(f"scored {len(ledger) - before} new games with the {label} posterior")

    games = data.load_games(season)
    status = {"season": season, "updated": pd.Timestamp.now(tz="UTC").isoformat(), "games": int(games.game_id.nunique()) if len(games) else 0,
              "scored": len(ledger), "posterior": label}
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
        status["posterior"] = f"{season} fit ({refit})"
        log(f"refit {season} with {refit}")
    status["seconds"] = round(time.time() - t0)
    (model.artifact_dir(season)).mkdir(parents=True, exist_ok=True)
    (model.artifact_dir(season) / "status.json").write_text(json.dumps(status, indent=1))
    return status


def upcoming(season: int, post: model.Posterior, start=None, days: int = 1, n: int = 1500) -> list[dict]:
    """Pre-game forecasts for scheduled games in a date window."""
    from .simulate import simulate_games

    sched = data.load_schedule(season)
    if not len(sched):
        return []
    start = pd.Timestamp.now(tz="UTC").normalize() if start is None else pd.Timestamp(start, tz="UTC")
    win = sched[(sched.date >= start) & (sched.date < start + pd.Timedelta(days=days))]
    idx = {t: i for i, t in enumerate(post.teams)}
    out = []
    rng = np.random.default_rng(0)
    for r in win.itertuples():
        row = {"game_id": int(r.game_id), "date": r.date.isoformat(), "home": r.home, "away": r.away,
               "neutral": bool(r.neutral), "completed": bool(r.completed)}
        if r.home in idx and r.away in idx:
            res = simulate_games(post, rng.integers(post.n_draws, size=n), np.full(n, idx[r.home]),
                                 np.full(n, idx[r.away]), rng, home=0 if r.neutral else 1, full_box=False)
            m = res["pts"][:, 0] - res["pts"][:, 1]
            row |= {"p_home": float(res["a_wins"].mean()), "pred_margin": float(m.mean()),
                    "pred_score": res["pts"].mean(0).round(1).tolist()}
        out.append(row)
    return out
