"""Calibrating the game simulator on ten tournaments, out of sample.

Tested on the backtest's tournament games, the raw model was underconfident: favourites it gave 85% won 93% of
the time. Two things caused it, and both are measurable from the games:

* the gaps between teams were compressed: actual margins ran 1.2 times the predicted ones;
* single games looked too noisy (13.3 points of spread against 12.0), but that turned out to be the seasons
  without half-time scores, which had no second-half lead erosion; with it, the spread matches.

The simulator therefore widens every matchup's skill gap by ``skill_scale`` (``ncaa.simulate.cal()``), fitted on
the margins of each tournament game as forecast on Selection Sunday. The live site uses the value fitted on all
ten tournaments; the backtest scores each season with the value fitted on the other nine. (``phi_scale``, the
game-night precision, stays at 1.)

``python -m ncaa calibrate`` (after ``python -m ncaa backtest`` has fitted the seasons) writes
``ncaa/calibration.json``. Ratings, rankings and the rating updates are untouched: only game simulation
(win chances, margins, box scores, brackets, projections) changes.
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import data, model
from .simulate import calibration, calibration_file, simulate_games

N_SIMS = 1000


def matchups(season: int) -> dict:
    """The season's played tournament games: posterior team indices, actual margin (home minus away)."""
    post = model.load_posterior(season)
    s = data.load_schedule(season)
    tg = s[s.tournament.astype(bool) & s.completed.astype(bool)]
    tg = tg[tg.home.isin(post.teams) & tg.away.isin(post.teams)]
    idx = {t: i for i, t in enumerate(post.teams)}
    return {"post": post, "a": tg.home.map(idx).to_numpy(), "b": tg.away.map(idx).to_numpy(),
            "margin": (tg.home_score - tg.away_score).to_numpy(float), "game_id": tg.game_id.to_numpy()}


def simulate(m: dict, skill: float, phi: float, n: int = N_SIMS, seed: int = 0) -> dict:
    """Mean margin, margin spread and home win chance of every matchup under one calibration."""
    rng = np.random.default_rng(seed)
    G = len(m["a"])
    a, b = np.repeat(m["a"], n), np.repeat(m["b"], n)
    draws = rng.integers(m["post"].n_draws, size=len(a))
    with calibration(skill, phi):
        r = simulate_games(m["post"], draws, a, b, rng, home=np.zeros(len(a)), full_box=False)
    mg = (r["pts"][:, 0] - r["pts"][:, 1]).reshape(G, n).astype(float)
    return {"mean": mg.mean(1), "sd": mg.std(1), "p": (r["a_wins"].reshape(G, n).mean(1) * n + 0.5) / (n + 1)}


def log_loss(p: np.ndarray, margin: np.ndarray) -> np.ndarray:
    """Per-game log loss of home win chances ``p`` given the actual margins."""
    won = margin > 0
    return -np.log(np.where(won, p, 1 - p))


def fit(seasons, log=print) -> dict:
    """The skill gap from margins, then a check by log loss with the same random numbers before and after.

    Simulated win chances from a few hundred games per matchup are too noisy to pick a calibration by log loss
    over a grid (the minimum chases the noise), so the skill gap is fitted where the signal is smooth: real
    margins against the predicted ones. The predicted margin scales almost exactly with ``skill_scale``, so
    the slope of actual on predicted (through zero) is the factor that makes it 1. Leaving each season out
    gives the factor that season is scored with. Game-night noise is left as it is: with the half-time lead
    erosion in place, simulated spread already matches the real one (12.2 against 11.9 points)."""
    t0 = time.time()
    ms = {s: matchups(s) for s in seasons}
    raw = {s: simulate(ms[s], 1.0, 1.0, seed=1) for s in seasons}
    xy = {s: float((raw[s]["mean"] * ms[s]["margin"]).sum()) for s in seasons}
    xx = {s: float((raw[s]["mean"] ** 2).sum()) for s in seasons}
    slope = lambda keep: round(sum(xy[s] for s in keep) / sum(xx[s] for s in keep), 2)
    k_all = slope(seasons)
    n = {s: len(ms[s]["a"]) for s in seasons}
    N = sum(n.values())
    loso, ll_raw, ll_all, ll_loso = {}, 0.0, 0.0, 0.0
    for s in seasons:
        k = slope([x for x in seasons if x != s])
        r_raw = simulate(ms[s], 1.0, 1.0, seed=7)                   # the same random numbers for all three
        r_loso = simulate(ms[s], k, 1.0, seed=7)
        r_all = simulate(ms[s], k_all, 1.0, seed=7)
        a, b_, c = (log_loss(r["p"], ms[s]["margin"]).sum() for r in (r_raw, r_loso, r_all))
        ll_raw += a; ll_loso += b_; ll_all += c
        loso[s] = {"skill_scale": k, "phi_scale": 1.0, "log_loss_raw": a / n[s], "log_loss": b_ / n[s]}
        log(f"{s}: skill x{k} (fitted without {s}); log loss {a / n[s]:.4f} -> {b_ / n[s]:.4f} ({time.time() - t0:.0f}s)")
    pm = np.concatenate([raw[s]["mean"] for s in seasons]); act = np.concatenate([ms[s]["margin"] for s in seasons])
    return {"skill_scale": k_all, "phi_scale": 1.0, "seasons": list(seasons), "games": N,
            "log_loss_raw": ll_raw / N, "log_loss": ll_all / N, "log_loss_out_of_sample": ll_loso / N,
            "margin_slope_raw": float((pm * act).sum() / (pm * pm).sum()),
            "sim_sd_raw": float(np.concatenate([raw[s]["sd"] for s in seasons]).mean()),
            "resid_sd": float((act - pm).std()), "leave_one_out": {str(s): v for s, v in loso.items()}}


def run(seasons=None, log=print) -> dict:
    from .backtest import SEASONS

    out = fit(seasons or SEASONS, log)
    calibration_file().write_text(json.dumps(out, indent=1))
    log(f"calibration: skill x{out['skill_scale']}, game-night precision x{out['phi_scale']}; tournament log loss "
        f"{out['log_loss_raw']:.4f} -> {out['log_loss']:.4f} (each season fitted on the others: "
        f"{out['log_loss_out_of_sample']:.4f}) -> {calibration_file()}")
    return out


def leave_one_out(season: int) -> tuple[float, float] | None:
    """The calibration fitted without ``season``, for scoring it honestly; None if there is none."""
    f = calibration_file()
    if not f.exists():
        return None
    c = json.loads(f.read_text()).get("leave_one_out", {}).get(str(season))
    return (c["skill_scale"], c["phi_scale"]) if c else None
