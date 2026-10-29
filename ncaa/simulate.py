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


SHOOTING = [J["p2"], J["p3"]]
SEGMENTS = 8  # five-minute stretches per regulation game


def _logit(p):
    return np.log(p / (1 - p))


def _ppp_slope(r, eps=0.05) -> np.ndarray:
    """d(expected points per possession) / d(shooting logit) at night rates ``r``."""
    eta = np.zeros_like(r)
    eta[:, :S - 1] = _logit(np.clip(r[:, :S - 1], 1e-6, 1 - 1e-6))
    up, dn = eta.copy(), eta.copy()
    up[:, SHOOTING] += eps
    dn[:, SHOOTING] -= eps
    return (expected_efficiency(up) - expected_efficiency(dn)) / (2 * eps)


def _lead_adjusted(beta, surplus, n2, ra, rb):
    """Shift both teams' shooting so the expected margin over the next ``n2``
    possessions moves by ``beta * surplus``, where ``surplus`` is the current
    lead beyond what the matchup predicted."""
    sa, sb = _ppp_slope(ra), _ppp_slope(rb)
    delta = beta * surplus / (np.maximum(n2, 1) * (sa + sb))
    ra2, rb2 = ra.copy(), rb.copy()
    for j in SHOOTING:
        ra2[:, j] = 1 / (1 + np.exp(-(_logit(ra[:, j]) + delta)))
        rb2[:, j] = 1 / (1 + np.exp(-(_logit(rb[:, j]) - delta)))
    return ra2, rb2


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
    eta_a = matchup_eta(post, k, a, b, h)
    eta_b = matchup_eta(post, k, b, a, None if h is None else -h)
    ra, rb = _night_rates(post, k, eta_a, rng), _night_rates(post, k, eta_b, rng)
    log_pace = post.mu_pace[k] + post.z[k, a, PACE] + post.z[k, b, PACE] \
        + post.sigma_pace[k] * rng.standard_normal(n)
    n_poss = np.maximum(np.round(np.exp(log_pace)).astype(np.int64), 30)

    # The game is played in SEGMENTS stretches.  Before each one, part of any
    # lead beyond what the matchup predicts erodes (benches, fouling, effort),
    # at a rate matching the half-time regression estimated from real games.
    exp_diff = expected_efficiency(eta_a) - expected_efficiency(eta_b)
    kappa = 1 - (1 + post.lead_beta[k]) ** (1 / (SEGMENTS // 2))
    sa = sb = None
    for i in range(SEGMENTS):
        lo, hi = n_poss * i // SEGMENTS, n_poss * (i + 1) // SEGMENTS
        if sa is None:
            ra_i, rb_i = ra, rb
        else:
            surplus = sa["pts"] - sb["pts"] - lo * exp_diff
            ra_i, rb_i = _lead_adjusted(-kappa, surplus, hi - lo, ra, rb)
        ea, eb = _play(hi - lo, ra_i, rng), _play(hi - lo, rb_i, rng)
        sa, sb = (ea, eb) if sa is None else (_merge(sa, ea), _merge(sb, eb))
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


def estimate_lead_beta(games, post: Posterior, n_draws=None, seed=0) -> np.ndarray:
    """How much of an unexpected half-time lead survives the second half.

    Regresses each game's second-half margin on its expected second-half margin
    and its first-half surplus (actual minus expected first-half margin), using
    the posterior-mean matchup predictions.  Returns draws of the surplus
    coefficient from its (flat-prior) normal posterior.
    """
    g = games[~games.duplicated("game_id") & (games.ot == 0)].dropna(subset=["h1", "opp_h1"])
    idx = {t: i for i, t in enumerate(post.teams)}
    a, b, h = g.team.map(idx).to_numpy(), g.opponent.map(idx).to_numpy(), g.home.to_numpy()
    mean = Posterior(teams=post.teams, z=post.z.mean(0, keepdims=True), mu=post.mu.mean(0, keepdims=True),
                     hfa=post.hfa.mean(0, keepdims=True), phi=post.phi.mean(0, keepdims=True),
                     mu_pace=post.mu_pace.mean(keepdims=True), sigma_pace=post.sigma_pace.mean(keepdims=True),
                     omega=post.omega)
    k = np.zeros(len(g), dtype=int)
    pace = np.exp(mean.mu_pace[k] + mean.z[k, a, PACE] + mean.z[k, b, PACE])
    exp_half = pace / 2 * (expected_efficiency(matchup_eta(mean, k, a, b, h))
                           - expected_efficiency(matchup_eta(mean, k, b, a, -h)))
    m1 = (g.h1 - g.opp_h1).to_numpy()
    m2 = (g.pts - g.h1 - (g.opp_pts - g.opp_h1)).to_numpy()
    X = np.column_stack([np.ones(len(g)), exp_half, m1 - exp_half])
    coef, *_ = np.linalg.lstsq(X, m2, rcond=None)
    resid = m2 - X @ coef
    cov = np.linalg.inv(X.T @ X) * resid.var(ddof=3)
    rng = np.random.default_rng(seed)
    return rng.normal(coef[2], np.sqrt(cov[2, 2]), size=n_draws or post.n_draws)
