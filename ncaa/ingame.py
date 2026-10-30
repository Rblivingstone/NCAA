"""Bayesian in-game forecasting: the final box score and the winner, given the game so far.

Pre-game, a matchup is described by the posterior draws of both teams'
skills.  Once play starts, the box score so far is data, and three things
are updated before the rest of the game is simulated:

1. **Team skills.**  Each posterior draw is reweighted by the likelihood of
   the partial box score under it (the same beta-binomial counts the model
   is fit to: turnovers per possession, 3PA share, 2P%, 3P%, offensive
   rebounds, free throws, assists, steals, blocks).  A team that is clearly
   outplaying expectations shifts weight to the draws that rate it higher.

2. **This game's rates.**  Within a game every rate is Beta(phi*pi, phi*(1-pi))
   around the matchup mean (the "night" effect).  That prior is conjugate to
   the counts so far, so each team's rest-of-game rates are drawn from
   Beta(phi*pi + made, phi*(1-pi) + missed): a cold-shooting first half moves
   the rest of the game a little, by exactly as much as phi says it should.

3. **Tempo.**  The pre-game possession rate (log-normal) is combined with the
   pace observed so far, weighted by their precisions.

The rest of the game is then played possession by possession from the
current score, in five-minute stretches with the usual lead erosion, plus
overtime if tied.  Final box score = so far + rest of game.

``live_unusualness`` asks the in-game version of the earlier question: how
unusual is the box score so far, against what the pre-game model expected
over the same number of possessions?
"""
from __future__ import annotations

import numpy as np

from .evaluate import JOINT_STATS, unusualness
from .live import STATS, LiveState
from .model import FT_TRIP, PACE, S, STATS as RATE_STATS, Posterior
from .simulate import (SEGMENTS, _lead_adjusted, _merge, _play, expected_efficiency, matchup_eta,
                       simulate_games)

J = {s: j for j, s in enumerate(RATE_STATS)}

# The last two minutes of regulation are different: the trailing team fouls on
# purpose, so both teams get more possessions and the leader's end at the free
# throw line.  Measured from 6,170 games of 2026 regular-season play-by-play:
# by the lead with 2:00 left -> (leader possessions, trailer possessions,
# share of the leader's plays that are foul-shot trips) over those 2 minutes.
LATE_MINUTES = 2.0
LATE_GAME = [  # (max lead, leader poss, trailer poss, leader trip share)
    (0, 4.10, 4.08, 0.23), (3, 4.53, 4.74, 0.38), (6, 4.86, 5.35, 0.52),
    (9, 5.01, 5.68, 0.59), (12, 4.70, 5.36, 0.54), (999, 3.80, 4.18, 0.21)]


def _late_regime(lead: np.ndarray):
    """Per-sim (leader possessions, trailer possessions, leader trip share) for a full 2-minute window."""
    lead = np.abs(lead)
    out = np.empty((len(lead), 3))
    for i, row in enumerate(LATE_GAME):
        lo = -1 if i == 0 else LATE_GAME[i - 1][0]
        m = (lead > lo) & (lead <= row[0])
        out[m] = row[1:]
    return out


def possessions(b: dict) -> float:
    return b["fga"] - b["oreb"] + b["tov"] + FT_TRIP * b["fta"]


def _counts(b: dict, opp: dict) -> tuple[np.ndarray, np.ndarray]:
    """(successes, trials) for every beta-binomial rate, offense ``b`` vs defense ``opp`` (STATS order, no fouls)."""
    poss = int(round(possessions(b)))
    trips = int(round(FT_TRIP * b["fta"]))
    miss = b["fga"] - b["fgm"]
    pairs = {
        "to": (b["tov"], max(poss, b["tov"])), "ftr": (trips, b["fga"] + trips), "three": (b["fg3a"], b["fga"]),
        "p2": (b["fgm"] - b["fg3m"], b["fga"] - b["fg3a"]), "p3": (b["fg3m"], b["fg3a"]),
        "orb": (min(b["oreb"], miss), miss), "ft": (b["ftm"], b["fta"]), "ast": (min(b["ast"], b["fgm"]), b["fgm"]),
        "stl": (min(opp["stl"], b["tov"]), b["tov"]), "blk": (min(opp["blk"], miss), miss),
    }
    k = np.array([pairs[s][0] for s in RATE_STATS[:-1]], dtype=float)
    n = np.array([pairs[s][1] for s in RATE_STATS[:-1]], dtype=float)
    return k, n


def _betabinom_loglik(k, n, a, b):
    from scipy.special import betaln

    return betaln(k + a, n - k + b) - betaln(a, b)


def forecast(post: Posterior, a: int, b: int, home: float, state: LiveState, n: int = 4000,
             seed: int | None = None) -> dict:
    """Forecast the final box score and winner of a game in progress.

    ``a``/``b`` are the posterior team indices of ``state.teams[0]``/``[1]``;
    ``home`` is +1 when ``a`` is at home, 0 on a neutral floor.
    """
    rng = np.random.default_rng(seed)
    K = post.n_draws
    cur = [state.box.get(t) or dict.fromkeys(STATS, 0) for t in state.teams]
    ks = np.arange(K)
    eta = [matchup_eta(post, ks, np.full(K, a), np.full(K, b), np.full(K, home)),
           matchup_eta(post, ks, np.full(K, b), np.full(K, a), np.full(K, -home))]
    prob = [1 / (1 + np.exp(-e[:, :S - 1])) for e in eta]                     # K x (S-1)
    phi = post.phi                                                             # K x (S-1)

    # 1. Reweight posterior draws by the likelihood of the box score so far.
    counts = [_counts(cur[0], cur[1]), _counts(cur[1], cur[0])]
    loglik = np.zeros(K)
    for j in range(2):
        kk, nn = counts[j]
        loglik += _betabinom_loglik(kk, nn, phi * prob[j], phi * (1 - prob[j])).sum(1)
    # Tempo so far: Poisson count of possessions given the draw's pace.
    poss_so_far = 0.5 * (possessions(cur[0]) + possessions(cur[1]))
    elapsed, remaining = state.elapsed, state.remaining
    rate = np.exp(post.mu_pace + post.z[:, a, PACE] + post.z[:, b, PACE]) / 40.0    # possessions per minute
    if elapsed > 0:
        loglik += poss_so_far * np.log(rate * elapsed) - rate * elapsed
    w = np.exp(loglik - loglik.max())
    w /= w.sum()
    ess = float(1 / (w ** 2).sum())
    draws = rng.choice(K, size=n, p=w)

    # 2. This game's rates: conjugate beta update from the counts so far.
    r = []
    for j in range(2):
        kk, nn = counts[j]
        pi, ph = prob[j][draws], phi[draws]
        rates = np.empty((n, S))
        rates[:, :S - 1] = rng.beta(ph * pi + kk, ph * (1 - pi) + nn - kk)
        rates[:, S - 1] = np.exp(eta[j][draws, S - 1])
        r.append(rates)

    # 3. Tempo: log-normal prior on possessions per minute, updated by the pace so far.
    prior_mu = np.log(rate[draws])
    prior_var = post.sigma_pace[draws] ** 2
    if elapsed > 0 and poss_so_far > 0:
        obs_mu, obs_var = np.log(poss_so_far / elapsed), 1.0 / poss_so_far
        post_var = 1 / (1 / prior_var + 1 / obs_var)
        post_mu = post_var * (prior_mu / prior_var + obs_mu / obs_var)
    else:
        post_mu, post_var = prior_mu, prior_var
    per_min = np.exp(post_mu + np.sqrt(post_var) * rng.standard_normal(n))
    # Regulation splits into ordinary play and the last two minutes (fouling regime).
    late = min(remaining, LATE_MINUTES) if state.period <= 2 and state.status != "post" else 0.0
    normal = remaining - late
    left = rng.poisson(per_min * normal)

    # 4. Play out the rest from the current score, in ~5-minute stretches with lead erosion.
    eta_a, eta_b = eta[0][draws], eta[1][draws]
    exp_diff = expected_efficiency(eta_a) - expected_efficiency(eta_b)
    kappa = 1 - (1 + post.lead_beta[draws]) ** (1 / (SEGMENTS // 2))
    margin0 = cur[0]["pts"] - cur[1]["pts"]
    expected0 = poss_so_far * exp_diff
    segs = max(1, int(np.ceil(normal / 5.0)))
    zero = {key: np.zeros(n, dtype=np.int64) for key in
            ("pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "tov", "ast",
             "opp_dreb", "opp_stl", "opp_blk", "opp_pf", "poss")}
    sa, sb = {k_: v.copy() for k_, v in zero.items()}, {k_: v.copy() for k_, v in zero.items()}
    done = np.zeros(n, dtype=np.int64)
    for i in range(segs):
        lo, hi = left * i // segs, left * (i + 1) // segs
        surplus = margin0 + sa["pts"] - sb["pts"] - (expected0 + done * exp_diff)
        ra_i, rb_i = _lead_adjusted(-kappa, surplus, np.maximum(hi - lo, 1), r[0], r[1])
        sa, sb = _merge(sa, _play(hi - lo, ra_i, rng)), _merge(sb, _play(hi - lo, rb_i, rng))
        done += hi - lo
    if late > 0:
        # Final two minutes: possession counts and the leader's trip share depend on the lead.
        lead = margin0 + sa["pts"] - sb["pts"]
        reg = _late_regime(lead)
        frac = late / LATE_MINUTES
        a_leads = lead > 0
        mean_a = np.where(a_leads, reg[:, 0], reg[:, 1]) * frac
        mean_b = np.where(a_leads, reg[:, 1], reg[:, 0]) * frac
        if state.possession is not None and normal == 0:
            # Already inside the last two minutes: the team with the ball is half a
            # possession ahead of the other in what remains.
            has = 0 if state.possession == state.teams[0] else 1
            shift = 0.5 * np.ones(n)
            mean_a, mean_b = (mean_a + shift, np.maximum(mean_b - shift, 0)) if has == 0 else \
                (np.maximum(mean_a - shift, 0), mean_b + shift)
        n_a, n_b = rng.poisson(mean_a), rng.poisson(mean_b)
        ra_l, rb_l = r[0].copy(), r[1].copy()
        ra_l[a_leads, J["ftr"]] = reg[a_leads, 2]
        b_leads = lead < 0
        rb_l[b_leads, J["ftr"]] = reg[b_leads, 2]
        sa, sb = _merge(sa, _play(n_a, ra_l, rng)), _merge(sb, _play(n_b, rb_l, rng))
        left = left + (n_a + n_b) // 2
    final_a = cur[0]["pts"] + sa["pts"]
    final_b = cur[1]["pts"] + sb["pts"]
    ot = np.zeros(n, dtype=np.int64)
    if state.status != "post":
        tied = final_a == final_b
        while tied.any():
            idx = np.flatnonzero(tied)
            np_ot = np.maximum(np.round(per_min[idx] * 5).astype(np.int64), 1)
            ea, eb = _play(np_ot, r[0][idx], rng), _play(np_ot, r[1][idx], rng)
            for key in sa:
                sa[key][idx] += ea[key]
                sb[key][idx] += eb[key]
            final_a, final_b = cur[0]["pts"] + sa["pts"], cur[1]["pts"] + sb["pts"]
            ot[idx] += 1
            tied = final_a == final_b

    rest = {"dreb": (sb["opp_dreb"], sa["opp_dreb"]), "stl": (sb["opp_stl"], sa["opp_stl"]),
            "blk": (sb["opp_blk"], sa["opp_blk"]), "pf": (sb["opp_pf"], sa["opp_pf"])}
    final = {}
    for s in STATS:
        add = rest[s] if s in rest else (sa[s], sb[s])
        final[s] = np.stack([cur[0][s] + add[0], cur[1][s] + add[1]], axis=1)
    p_a = float((final["pts"][:, 0] > final["pts"][:, 1]).mean())
    if state.status == "post":
        p_a = float(cur[0]["pts"] > cur[1]["pts"])
    margin = final["pts"][:, 0] - final["pts"][:, 1]
    return {
        "p_win": [p_a, 1 - p_a],
        "final": {s: {"mean": final[s].mean(0).round(1).tolist(),
                      "p10": np.percentile(final[s], 10, axis=0).round(1).tolist(),
                      "p90": np.percentile(final[s], 90, axis=0).round(1).tolist()} for s in STATS},
        "margin": {"mean": float(margin.mean()), "p10": float(np.percentile(margin, 10)),
                   "p90": float(np.percentile(margin, 90))},
        "p_overtime": float((ot > 0).mean()) if state.status != "post" else 0.0,
        "possessions_left": float(left.mean()),
        "effective_draws": ess,
        "samples": final,
    }


def pregame(post: Posterior, a: int, b: int, home: float, n: int = 4000, seed: int | None = None) -> dict:
    """The pre-game forecast (no data yet), for comparison with the live one."""
    rng = np.random.default_rng(seed)
    res = simulate_games(post, rng.integers(post.n_draws, size=n), np.full(n, a), np.full(n, b), rng, home=home)
    return {"p_win": [float(res["a_wins"].mean()), float(1 - res["a_wins"].mean())],
            "final": {s: {"mean": res[s].mean(0).round(1).tolist()} for s in STATS if s in res}}


def live_unusualness(post: Posterior, a: int, b: int, home: float, state: LiveState, n: int = 3000,
                     seed: int | None = None) -> dict | None:
    """How unusual is the box score so far, against the pre-game model over the same possessions?"""
    cur = [state.box.get(t) for t in state.teams]
    if not cur[0] or not cur[1]:
        return None
    poss = int(round(0.5 * (possessions(cur[0]) + possessions(cur[1]))))
    if poss < 5:
        return None
    rng = np.random.default_rng(seed)
    res = simulate_games(post, rng.integers(post.n_draws, size=n), np.full(n, a), np.full(n, b), rng,
                         home=home, n_poss=poss, overtime=False)
    X = np.column_stack([res[s][:, j] for j in range(2) for s in JOINT_STATS]).astype(float)
    x = np.array([cur[j][s] for j in range(2) for s in JOINT_STATS], dtype=float)
    u = unusualness(X, x)
    labels = [(state.teams[j], s) for j in range(2) for s in JOINT_STATS]
    top = np.argsort(-u["contrib"])[:3]
    margin = res["pts"][:, 0] - res["pts"][:, 1]
    m = cur[0]["pts"] - cur[1]["pts"]
    return {"p_value": u["p_value"], "distance": u["distance"], "possessions": poss,
            "margin_percentile": float((margin < m).mean() + 0.5 * (margin == m).mean()),
            "drivers": [{"team": labels[i][0], "stat": labels[i][1], "share": float(u["contrib"][i])}
                        for i in top if u["contrib"][i] > 0.1]}
