"""Command line pipeline.

    python -m ncaa fetch    --season 2026   # download box scores + bracket
    python -m ncaa fit      --season 2026   # NUTS posterior over team skills
    python -m ncaa simulate --season 2026   # Monte Carlo tournaments + win matrix
    python -m ncaa build    --season 2026   # all three
    python -m ncaa update   --season 2027   # nightly during a season: fetch, score new games, refit
"""
from __future__ import annotations

import argparse
import json
import time

from . import data, model
from .bracket import Bracket, pairwise_win_matrix, save_simulation, simulate_tournament, summarize
from .ratings import team_ratings
from .simulate import estimate_lead_beta


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
    beta = estimate_lead_beta(games, model.load_posterior(a.season))
    model.add_to_posterior(a.season, lead_beta=beta)
    print(f"second-half lead regression: {beta.mean():.3f} ± {beta.std():.3f} per point of half-time surplus")
    print((out / "diagnostics.json").read_text())


def cmd_simulate(a):
    post = model.load_posterior(a.season)
    br = Bracket(data.load_bracket(a.season))
    t = time.time()
    sims = simulate_tournament(br, post, a.sims, seed=a.seed)
    summary = summarize(br, sims)
    summary["bracket"] = br.to_json()
    summary["ratings"] = team_ratings(post, br)
    print(f"{a.sims} tournaments in {time.time() - t:.0f}s")
    t = time.time()
    draw_ids, W = pairwise_win_matrix(br, post, n_draws=a.win_draws, sims_per_draw=a.win_sims, seed=a.seed + 1)
    print(f"pairwise win matrix in {time.time() - t:.0f}s")
    save_simulation(a.season, br, sims, summary, W, draw_ids)
    champ = sorted(((p[-1], br.teams[i]) for i, p in enumerate(summary["reach"])), reverse=True)[:8]
    print("title odds:", json.dumps([(t, round(p, 3)) for p, t in champ]))


def cmd_update(a):
    from . import season

    st = season.update(a.season, refit=a.refit)
    print(json.dumps(st, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(prog="ncaa")
    p.add_argument("command", choices=["fetch", "fit", "simulate", "build", "update"])
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
    a = p.parse_args(argv)
    steps = {"fetch": [cmd_fetch], "fit": [cmd_fit], "simulate": [cmd_simulate],
             "build": [cmd_fetch, cmd_fit, cmd_simulate], "update": [cmd_update]}[a.command]
    for step in steps:
        step(a)


if __name__ == "__main__":
    main()
