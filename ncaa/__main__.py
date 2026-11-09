"""Command line pipeline.

    python -m ncaa fetch    --season 2026   # download box scores + bracket
    python -m ncaa fit      --season 2026   # NUTS posterior over team skills
    python -m ncaa simulate --season 2026   # Monte Carlo tournaments + win matrix
    python -m ncaa build    --season 2026   # all three
    python -m ncaa update   --season 2027   # nightly during a season: fetch, score new games, refit
    python -m ncaa project  --season 2027   # nightly: Road to March, the rest of the season simulated
    python -m ncaa geo      --season 2027   # team locations and tournament sites, for the bracket rules
    python -m ncaa backtest                 # ten tournaments out of sample against seeds, efficiency, BPI, Vegas
    python -m ncaa calibrate                # fit the game simulator's calibration on those tournaments
    python -m ncaa restate  --season 2027   # put logged forecasts on the current calibration (also done nightly)
    python -m ncaa watch    --season 2027   # all day: track every game from tip-off (win-prob history), update ratings at the final
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from . import data, model
from .bracket import Bracket, pairwise_win_matrix, save_simulation, simulate_tournament, summarize
from .ratings import team_ratings
from .simulate import estimate_lead_beta


DEFAULT_LEAD_BETA = -0.18     # second-half change in margin per point of half-time lead, where it can't be fitted


def cmd_fetch(a):
    games, bracket = data.fetch(a.season)
    print(f"{len(games)} team-games, {games.team.nunique() if len(games) else 0} teams")
    if bracket:
        print(f"{sum(len(v) for r in bracket['seeds'].values() for v in r.values())} tournament teams, "
              f"{len(bracket['results'])} tournament results")


def cmd_fit(a):
    games = data.load_games(a.season)
    t = time.time()
    out = model.fit(games, a.season, method=a.method, num_warmup=a.warmup, num_samples=a.samples, num_chains=a.chains,
                    seed=a.seed)
    print(f"fit in {time.time() - t:.0f}s -> {out}")
    if games.h1.notna().sum() < 100:
        # Seasons before 2023 have no half-time scores. How far leads erode is a stable fact of the game
        # (every season with the data gives -0.175 to -0.183), so those seasons use the typical value.
        post = model.load_posterior(a.season)
        model.add_to_posterior(a.season, lead_beta=np.full(post.n_draws, DEFAULT_LEAD_BETA))
        print(f"no half-time scores this season: second-half lead regression set to the usual {DEFAULT_LEAD_BETA}")
    else:
        beta = estimate_lead_beta(games, model.load_posterior(a.season))
        model.add_to_posterior(a.season, lead_beta=beta)
        print(f"second-half lead regression: {beta.mean():.3f} ± {beta.std():.3f} per point of half-time surplus")
    print((out / "diagnostics.json").read_text())


def cmd_simulate(a):
    """The bracket under the calibrated model and, for the bracket builder's comparison switch, the raw one."""
    from .simulate import cal, calibration

    post = model.load_posterior(a.season)
    br = Bracket(data.load_bracket(a.season))
    k_cal = cal()["skill_scale"]
    variants = [("calibrated", k_cal)] + ([("raw", 1.0)] if k_cal != 1.0 else [])
    for variant, k in variants:
        with calibration(k, 1.0):
            t = time.time()
            sims = simulate_tournament(br, post, a.sims, seed=a.seed)
            summary = summarize(br, sims)
            summary["bracket"] = br.to_json()
            summary["ratings"] = team_ratings(post, br)
            summary["calibration"] = {"skill_scale": k, "variant": variant}
            print(f"{variant}: {a.sims} tournaments in {time.time() - t:.0f}s")
            t = time.time()
            draw_ids, W = pairwise_win_matrix(br, post, n_draws=a.win_draws, sims_per_draw=a.win_sims, seed=a.seed + 1)
            print(f"{variant}: pairwise win matrix in {time.time() - t:.0f}s")
        save_simulation(a.season, br, sims, summary, W, draw_ids, variant=variant)
        champ = sorted(((p[-1], br.teams[i]) for i, p in enumerate(summary["reach"])), reverse=True)[:8]
        print(f"{variant} title odds:", json.dumps([(t, round(p, 3)) for p, t in champ]))


def cmd_update(a):
    from . import season

    st = season.update(a.season, refit=a.refit)
    print(json.dumps(st, indent=1))


def cmd_project(a):
    from . import projection

    out = projection.run(a.season, n_sims=a.project_sims)
    print(f"Road to March: {out['n_sims']} simulations of the rest of {a.season}, {out['games_simulated']:,} games, {out['seconds']:.0f}s")


def cmd_geo(a):
    from . import geo

    geo.build(a.season, force=a.force)


def cmd_backtest(a):
    from . import backtest

    seasons = [int(x) for x in a.seasons.split(",")] if a.seasons else None
    backtest.run(seasons, sims=a.sims, force=a.force)


def cmd_calibrate(a):
    from . import calibration

    seasons = [int(x) for x in a.seasons.split(",")] if a.seasons else None
    calibration.run(seasons)


def cmd_restate(a):
    from . import season

    n = season.restate_ledger(a.season)
    print(f"{n} ledger forecasts restated on the current calibration")


def cmd_watch(a):
    from . import season

    print(f"watching {a.season}: polling games from their scheduled tip-off every {a.every}s (Ctrl-C to stop)", flush=True)
    season.watch(a.season, every=a.every)


def main(argv=None):
    p = argparse.ArgumentParser(prog="ncaa")
    p.add_argument("command", choices=["fetch", "fit", "simulate", "build", "update", "watch", "project", "geo", "backtest", "calibrate", "restate"])
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--method", choices=["nuts", "vi"], default="nuts",
                   help="nuts: full MCMC (slow, exact); vi: variational approximation (fast)")
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--samples", type=int, default=250, help="posterior draws per chain")
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--sims", type=int, default=10000, help="simulated tournaments")
    p.add_argument("--win-draws", type=int, default=100, help="posterior draws for the pairwise win matrix")
    p.add_argument("--win-sims", type=int, default=100, help="games per pair per draw for the win matrix")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--refit", choices=["vi", "nuts", "none"], default="vi", help="update: how to refit after scoring")
    p.add_argument("--every", type=int, default=60, help="watch: seconds between polls of live games")
    p.add_argument("--project-sims", type=int, default=1000, help="project: simulations of the rest of the season")
    p.add_argument("--force", action="store_true", help="geo: rewrite sites.json even if it exists (loses hand edits); "
                   "backtest: refit and resimulate")
    p.add_argument("--seasons", default="", help="backtest: comma-separated seasons (default: the last ten tournaments)")
    p.add_argument("--league", choices=["mens", "womens"], default=None,
                   help="men's or women's college basketball (default: $NCAA_LEAGUE, else men's)")
    a = p.parse_args(argv)
    if a.league:
        from . import league
        league.set_league(a.league)
    steps = {"fetch": [cmd_fetch], "fit": [cmd_fit], "simulate": [cmd_simulate],
             "build": [cmd_fetch, cmd_fit, cmd_simulate], "update": [cmd_update], "watch": [cmd_watch],
             "project": [cmd_project], "geo": [cmd_geo], "backtest": [cmd_backtest], "calibrate": [cmd_calibrate], "restate": [cmd_restate]}[a.command]
    for step in steps:
        step(a)


if __name__ == "__main__":
    main()
