"""Team summaries derived from the posterior latent skills."""
from __future__ import annotations

import numpy as np

from .model import PACE, S, STATS, Posterior, def_index
from .simulate import expected_efficiency


def team_ratings(post: Posterior, bracket=None) -> list[dict]:
    """Offensive/defensive efficiency against an average D1 opponent, per 100 possessions.

    ``net`` is the expected margin per 100 possessions on a neutral floor versus an
    average team, as of the end of the regular season; ``trend`` is how much it
    changed since opening night.  Posterior means with 90% intervals.
    """
    def efficiency(z):
        zdef = z[..., np.maximum(di, 0)] * (di >= 0)
        ortg = 100 * expected_efficiency((mu + z[..., :S]).reshape(-1, S)).reshape(z.shape[:2])
        drtg = 100 * expected_efficiency((mu + zdef).reshape(-1, S)).reshape(z.shape[:2])
        return ortg, drtg

    di = def_index()
    z, mu = post.z, post.mu[:, None, :]                       # draws x T x D, draws x 1 x S
    ortg, drtg = efficiency(z)
    net = ortg - drtg
    # Change over the season: net rating now minus net rating on opening night.
    o0, d0 = efficiency(post.at(1.0).z)
    trend = net - (o0 - d0)
    tempo = np.exp(post.mu_pace[:, None] + z[..., PACE])
    p = 1 / (1 + np.exp(-(mu + z[..., :S])))
    seeds = {} if bracket is None else bracket.seed
    regions = {} if bracket is None else bracket.region_of
    out = []
    for i, t in enumerate(post.teams):
        out.append({
            "team": t, "seed": seeds.get(t), "region": regions.get(t),
            "ortg": float(ortg[:, i].mean()), "drtg": float(drtg[:, i].mean()),
            "net": float(net[:, i].mean()),
            "net_lo": float(np.quantile(net[:, i], 0.05)), "net_hi": float(np.quantile(net[:, i], 0.95)),
            "tempo": float(tempo[:, i].mean()),
            "trend": float(trend[:, i].mean()),
            "trend_lo": float(np.quantile(trend[:, i], 0.05)), "trend_hi": float(np.quantile(trend[:, i], 0.95)),
            "rates": {s: float(p[:, i, j].mean()) for j, s in enumerate(STATS) if s != "pf"},
        })
    out.sort(key=lambda r: -r["net"])
    for rank, r in enumerate(out, 1):
        r["rank"] = rank
    return out
