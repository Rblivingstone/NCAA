"""Calibrating the game simulator on ten tournaments, out of sample.

Tested on the backtest's tournament games, the raw model was underconfident: favourites it gave 85% won 93% of
the time. Two things caused it, and both are measurable from the games:

* the gaps between teams were compressed: actual margins ran 1.2 times the predicted ones;
* single games were too noisy: simulated margins spread 13.3 points around their mean, real ones 12.0.

The simulator therefore gets two settings (``ncaa.simulate.CAL``): ``skill_scale`` multiplies the matchup's
skill gap, and ``phi_scale`` multiplies the beta-binomial precision of game-night rates (more precision, less
noise). They are chosen together by log loss over a grid, each tournament's game chance coming from
simulations of that exact matchup on Selection Sunday. The live site uses the values fitted on all ten
tournaments; the backtest scores each season with the values fitted on the other nine.

``python -m ncaa calibrate`` (after ``python -m ncaa backtest`` has fitted the seasons) writes
``ncaa/calibration.json``. Ratings, rankings and the rating updates are untouched: only game simulation
(win chances, margins, box scores, brackets, projections) changes.
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import data, model
from .simulate import CALIBRATION_FILE, calibration, simulate_games

SKILL_GRID = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
PHI_GRID = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0]
N_SIMS = 400


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
    """Grid of (skill_scale, phi_scale) by log loss: all seasons together, and leaving each season out."""
    t0 = time.time()
    ms = {s: matchups(s) for s in seasons}
    grid = [(k, f) for k in SKILL_GRID for f in PHI_GRID]
    loss = {s: np.zeros(len(grid)) for s in seasons}               # summed log loss per season, per grid point
    n_games = {s: len(ms[s]["a"]) for s in seasons}
    diag = {}
    for gi, (k, f) in enumerate(grid):
        means, sds, margins = [], [], []
        for s in seasons:
            r = simulate(ms[s], k, f, seed=gi)
            loss[s][gi] = log_loss(r["p"], ms[s]["margin"]).sum()
            means.append(r["mean"]); sds.append(r["sd"]); margins.append(ms[s]["margin"])
        pm, act = np.concatenate(means), np.concatenate(margins)
        diag[(k, f)] = {"margin_slope": float((pm * act).sum() / (pm * pm).sum()),
                        "sim_sd": float(np.concatenate(sds).mean()), "resid_sd": float((act - pm).std())}
        log(f"skill {k:.1f} phi {f:>4.1f}: log loss {sum(l[gi] for l in loss.values()) / sum(n_games.values()):.4f} "
            f"| margin slope {diag[(k, f)]['margin_slope']:.2f}, spread {diag[(k, f)]['sim_sd']:.1f} vs "
            f"{diag[(k, f)]['resid_sd']:.1f} ({time.time() - t0:.0f}s)")
    total = sum(loss.values())
    best = int(np.argmin(total))
    N = sum(n_games.values())
    raw = grid.index((1.0, 1.0))
    loso = {}
    for s in seasons:
        others = total - loss[s]
        j = int(np.argmin(others))
        loso[s] = {"skill_scale": grid[j][0], "phi_scale": grid[j][1],
                   "log_loss_raw": float(loss[s][raw] / n_games[s]), "log_loss": float(loss[s][j] / n_games[s])}
    k, f = grid[best]
    return {"skill_scale": k, "phi_scale": f, "seasons": list(seasons), "games": N,
            "log_loss_raw": float(total[raw] / N), "log_loss": float(total[best] / N),
            "log_loss_out_of_sample": float(sum(loso[s]["log_loss"] * n_games[s] for s in seasons) / N),
            "margin_slope_raw": diag[(1.0, 1.0)]["margin_slope"], "margin_slope": diag[grid[best]]["margin_slope"],
            "sim_sd_raw": diag[(1.0, 1.0)]["sim_sd"], "sim_sd": diag[grid[best]]["sim_sd"],
            "resid_sd": diag[grid[best]]["resid_sd"], "leave_one_out": {str(s): v for s, v in loso.items()}}


def run(seasons=None, log=print) -> dict:
    from .backtest import SEASONS

    out = fit(seasons or SEASONS, log)
    CALIBRATION_FILE.write_text(json.dumps(out, indent=1))
    log(f"calibration: skill x{out['skill_scale']}, game-night precision x{out['phi_scale']}; tournament log loss "
        f"{out['log_loss_raw']:.4f} -> {out['log_loss']:.4f} (each season fitted on the others: "
        f"{out['log_loss_out_of_sample']:.4f}) -> {CALIBRATION_FILE}")
    return out


def leave_one_out(season: int) -> tuple[float, float] | None:
    """The calibration fitted without ``season``, for scoring it honestly; None if there is none."""
    if not CALIBRATION_FILE.exists():
        return None
    c = json.loads(CALIBRATION_FILE.read_text()).get("leave_one_out", {}).get(str(season))
    return (c["skill_scale"], c["phi_scale"]) if c else None
