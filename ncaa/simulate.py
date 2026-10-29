"""Vectorised simulation of full box scores from the posterior.

A batch of ``n`` games is described by arrays of posterior-draw indices and
team indices.  Every game draws its own night-level rates (beta noise around
the matchup rates), a possession count, and then plays the possession chain
for both teams.  Ties go to five-minute overtimes until someone wins.
"""
from __future__ import annotations

import numpy as np

from .model import FT_TRIP, PACE, S, STATS, Posterior, def_index

J = {s: j for j, s in enumerate(STATS)}
EXTRA_FTA = 1 / FT_TRIP - 2  # and-ones / three-shot fouls per trip
BOX_STATS = ["pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "tov",
             "ast", "stl", "blk", "pf", "poss"]


def matchup_eta(post: Posterior, k, o, d, home=None) -> np.ndarray:
    """Linear predictors (n x S) for offense ``o`` against defense ``d`` under draws ``k``."""
    di = def_index()
    zo = post.z[k, o, :S]
    zd = post.z[k, d][:, np.maximum(di, 0)] * (di >= 0)
    eta = post.mu[k] + zo + zd
    if home is not None:
        eta = eta + post.hfa[k] * np.asarray(home, dtype=float)[:, None]
    return eta


def _night_rates(post: Posterior, k, eta, rng) -> np.ndarray:
    """Draw this game's actual rates: Beta around the matchup mean (Poisson rate for fouls)."""
    p = 1 / (1 + np.exp(-eta))
    phi = post.phi[k]  # n x (S-1)
    out = np.empty_like(p)
    pp = p[:, :S - 1]
    out[:, :S - 1] = rng.beta(phi * pp, phi * (1 - pp))
    out[:, S - 1] = np.exp(eta[:, S - 1])
    return out


def _play(n_poss, r, rng) -> dict:
    """Run ``n_poss`` possessions for one offense with night rates ``r`` (n x S)."""
    tov = rng.binomial(n_poss, r[:, J["to"]])
    plays = n_poss - tov
    acc = {k: np.zeros_like(n_poss) for k in ("trips", "fga", "fg3a", "fg3m", "fg2m", "oreb")}
    while plays.any():
        trips = rng.binomial(plays, r[:, J["ftr"]])
        fga = plays - trips
        fg3a = rng.binomial(fga, r[:, J["three"]])
        fg3m = rng.binomial(fg3a, r[:, J["p3"]])
        fg2m = rng.binomial(fga - fg3a, r[:, J["p2"]])
        oreb = rng.binomial(fga - fg3m - fg2m, r[:, J["orb"]])
        for key, v in (("trips", trips), ("fga", fga), ("fg3a", fg3a), ("fg3m", fg3m),
                       ("fg2m", fg2m), ("oreb", oreb)):
            acc[key] += v
        plays = oreb
    fta = 2 * acc["trips"] + rng.binomial(acc["trips"], EXTRA_FTA)
    ftm = rng.binomial(fta, r[:, J["ft"]])
    fgm = acc["fg2m"] + acc["fg3m"]
    miss = acc["fga"] - fgm
    return {
        "pts": 2 * acc["fg2m"] + 3 * acc["fg3m"] + ftm,
        "fgm": fgm, "fga": acc["fga"], "fg3m": acc["fg3m"], "fg3a": acc["fg3a"],
        "ftm": ftm, "fta": fta, "oreb": acc["oreb"], "tov": tov,
        "ast": rng.binomial(fgm, r[:, J["ast"]]),
        # Credited to the defending team:
        "opp_dreb": miss - acc["oreb"],
        "opp_stl": rng.binomial(tov, r[:, J["stl"]]),
        "opp_blk": rng.binomial(miss, r[:, J["blk"]]),
        "opp_pf": rng.poisson(n_poss * r[:, J["pf"]]),
        "poss": n_poss,
    }


def _merge(a: dict, b: dict) -> dict:
    return {k: a[k] + b[k] for k in a}


def simulate_games(post: Posterior, k, a, b, rng=None, home=None, full_box=True) -> dict:
    """Simulate games between team indices ``a`` and ``b`` (posterior team order).

    ``home`` is +1 when ``a`` is at home, -1 when away, None/0 for neutral.
    Returns arrays of shape (n, 2) for each box-score stat (column 0 = ``a``),
    plus ``ot`` (overtimes played) and ``a_wins``.
    """
    rng = rng or np.random.default_rng()
    k, a, b = (np.asarray(x) for x in (k, a, b))
    n = len(k)
    h = None if home is None else np.asarray(home, dtype=float) * np.ones(n)
    ra = _night_rates(post, k, matchup_eta(post, k, a, b, h), rng)
    rb = _night_rates(post, k, matchup_eta(post, k, b, a, None if h is None else -h), rng)
    log_pace = post.mu_pace[k] + post.z[k, a, PACE] + post.z[k, b, PACE] \
        + post.sigma_pace[k] * rng.standard_normal(n)
    n_poss = np.maximum(np.round(np.exp(log_pace)).astype(np.int64), 30)

    sa, sb = _play(n_poss, ra, rng), _play(n_poss, rb, rng)
    ot = np.zeros(n, dtype=np.int64)
    tied = sa["pts"] == sb["pts"]
    while tied.any():
        idx = np.flatnonzero(tied)
        np_ot = np.maximum(np.round(n_poss[idx] * 5 / 40).astype(np.int64), 1)
        ea, eb = _play(np_ot, ra[idx], rng), _play(np_ot, rb[idx], rng)
        for key in sa:
            sa[key][idx] += ea[key]
            sb[key][idx] += eb[key]
        ot[idx] += 1
        tied = sa["pts"] == sb["pts"]

    out = {"ot": ot, "a_wins": sa["pts"] > sb["pts"]}
    keys = BOX_STATS if full_box else ["pts"]
    for key in keys:
        if key == "dreb":
            out[key] = np.stack([sb["opp_dreb"], sa["opp_dreb"]], axis=1)
        elif key in ("stl", "blk", "pf"):
            out[key] = np.stack([sb["opp_" + key], sa["opp_" + key]], axis=1)
        else:
            out[key] = np.stack([sa[key], sb[key]], axis=1)
    return out


def expected_efficiency(eta: np.ndarray) -> np.ndarray:
    """Expected points per possession given linear predictors ``eta`` (n x S), at mean rates.

    Closed form of the possession chain: each play is a trip (worth ~FT points) or a
    field goal attempt; a missed attempt is rebounded with p_orb and becomes a new play.
    """
    p = 1 / (1 + np.exp(-eta))
    to, ftr, three, p2, p3, orb, ft = (p[:, J[s]] for s in ("to", "ftr", "three", "p2", "p3", "orb", "ft"))
    fg_pts = three * 3 * p3 + (1 - three) * 2 * p2
    make = three * p3 + (1 - three) * p2
    ft_pts = (2 + EXTRA_FTA) * ft
    per_play = (ftr * ft_pts + (1 - ftr) * fg_pts) / (1 - (1 - ftr) * (1 - make) * orb)
    return (1 - to) * per_play
