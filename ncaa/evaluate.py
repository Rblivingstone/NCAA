"""How unusual was a real game, judged against the model's simulated games?

Per-stat percentiles can't simply be combined, because box-score stats move
together (a hot shooting night raises FGM, points and assists at once).  So
the whole box score is scored jointly: its Mahalanobis distance from the
cloud of simulated box scores, using the simulations' own mean and
covariance.  The summary is

    p = share of simulated games at least as far from the centre as the real one,

computed from the simulated distances themselves, so no normality is
assumed.  p near 0 means the real game was stranger than almost anything the
model produces; for a calibrated model p is uniform over many games.

Left out:
  * points and possessions: exact functions of the other stats;
  * defensive rebounds: the simulator does not model rebounds of missed free
    throws, so they run ~2.5 low and would make every real game look odd;
  * field-goal and free-throw *attempts*: in the simulator they are pinned to
    possessions, turnovers, offensive rebounds and foul trips by exact
    bookkeeping, while real box scores are looser (team rebounds, end-of-half
    heaves, estimated possessions).  With them included, 20% of the 2026
    tournament games scored p < 0.05 (vs 5% expected) almost entirely
    because of that thin ridge; without them it is 6%.  Attempts are still
    compared one at a time in the per-stat percentiles.
"""
from __future__ import annotations

import numpy as np

from .model import Posterior
from .simulate import simulate_games

JOINT_STATS = ["fgm", "fg3m", "fg3a", "ftm", "oreb", "tov", "ast", "stl", "blk", "pf"]


def sim_matrix(res: dict) -> np.ndarray:
    """Simulated games as rows of [team A stats..., team B stats...]."""
    return np.column_stack([res[s][:, j] for j in range(2) for s in JOINT_STATS]).astype(float)


def real_vector(row) -> np.ndarray:
    return np.array([getattr(row, s) for s in JOINT_STATS] + [getattr(row, "opp_" + s) for s in JOINT_STATS],
                    dtype=float)


def unusualness(X: np.ndarray, x: np.ndarray) -> dict:
    """Mahalanobis distance of ``x`` from the rows of ``X`` and the share of rows at least as far."""
    mu = X.mean(0)
    cov = np.cov(X, rowvar=False) + 1e-6 * np.eye(X.shape[1])
    P = np.linalg.pinv(cov)
    dX = np.sqrt(np.einsum("ij,jk,ik->i", X - mu, P, X - mu))
    d = float(np.sqrt((x - mu) @ P @ (x - mu)))
    # Which stats pushed it out: each one's share of the squared distance.
    z = P @ (x - mu)
    contrib = (x - mu) * z
    hist, edges = np.histogram(dX, bins=40, range=(0, max(np.quantile(dX, 0.999), d) * 1.02))
    return {"distance": d, "p_value": float((dX >= d).mean()), "dims": X.shape[1],
            "hist": hist.tolist(), "edges": edges.round(3).tolist(),
            "sim_distances": dX, "contrib": contrib / max(contrib.sum(), 1e-9)}


def game_unusualness(post: Posterior, a: int, b: int, home: int, row, n: int = 4000, seed: int = 0,
                     res: dict | None = None) -> dict:
    """Simulate a matchup (or reuse ``res``) and score a real game's whole box score against it."""
    if res is None:
        rng = np.random.default_rng(seed)
        res = simulate_games(post, rng.integers(post.n_draws, size=n), np.full(n, a), np.full(n, b), rng,
                             home=home)
    u = unusualness(sim_matrix(res), real_vector(row))
    margin = res["pts"][:, 0] - res["pts"][:, 1]
    m = row.pts - row.opp_pts
    u["p_winner"] = float(res["a_wins"].mean() if m > 0 else 1 - res["a_wins"].mean())
    u["margin_percentile"] = float((margin < m).mean() + 0.5 * (margin == m).mean())
    labels = [f"{t}:{s}" for t in ("a", "b") for s in JOINT_STATS]
    top = np.argsort(-u["contrib"])[:3]
    u["drivers"] = [{"team": labels[i].split(":")[0], "stat": labels[i].split(":")[1],
                     "share": float(u["contrib"][i])} for i in top]
    return u


def tournament_unusualness(post: Posterior, tgames, n: int = 3000) -> list[dict]:
    """Score every tournament game (each from the perspective of the alphabetically first team)."""
    idx = {t: i for i, t in enumerate(post.teams)}
    rows = tgames[tgames.team < tgames.opponent]
    out = []
    for k, row in enumerate(rows.itertuples()):
        u = game_unusualness(post, idx[row.team], idx[row.opponent], int(row.home), row, n=n, seed=k)
        out.append({"game_id": int(row.game_id), "date": row.date.strftime("%b %d"), "teams": [row.team, row.opponent],
                    "score": [int(row.pts), int(row.opp_pts)], "p_value": u["p_value"], "distance": u["distance"],
                    "p_winner": u["p_winner"], "margin_percentile": u["margin_percentile"],
                    "drivers": [{"team": [row.team, row.opponent][d["team"] == "b"], "stat": d["stat"],
                                 "share": d["share"]} for d in u["drivers"]]})
    return out
