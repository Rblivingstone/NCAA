"""Why did a rating move? A plain account of each game's Bayesian update.

When a finished game updates a team, two things are worth showing:

* **what moved**: the change in each latent skill, turned into points per 100 possessions of net rating. Each
  skill's share is the change in net rating from moving that skill alone from its pre-game value to its
  post-game value, so the shares add up (nearly exactly) to the total change;
* **why**: the box-score evidence. For each rate the model tracks (turnovers per possession, free-throw trips
  per shot, three-point share, 2- and 3-point shooting, offensive rebounding, free-throw shooting, assists,
  steals, blocks), the count the model expected given the attempts, from both teams' pre-game skills, against
  what happened. The biggest surprises, in either direction, are the reasons.

Explanations are computed when the update runs and kept in ``explanations.jsonl`` (one line per game and team).
"""
from __future__ import annotations

import json

import numpy as np

from . import model
from .model import S, STATS, def_index

DI = def_index()

# How each tracked rate reads in words: (what the count is, what it is out of, label of the rate).
RATE_WORDS = {
    "to": ("turnovers", "possessions", "turnover rate"),
    "ftr": ("free-throw trips", "shots and trips", "free-throw rate"),
    "three": ("threes attempted", "shots", "three-point share"),
    "p2": ("twos made", "twos attempted", "2-point shooting"),
    "p3": ("threes made", "threes attempted", "3-point shooting"),
    "orb": ("offensive rebounds", "misses", "offensive rebounding"),
    "ft": ("free throws made", "free throws attempted", "free-throw shooting"),
    "ast": ("assisted baskets", "baskets", "assist rate"),
    "stl": ("turnovers that were steals", "turnovers", "live-ball turnovers"),
    "blk": ("shots blocked", "misses", "shots blocked"),
}
# Whether more of this, by the offense, is good for the offense (style stats are neutral: 0).
OFFENSE_GOOD = {"to": -1, "ftr": +1, "three": 0, "p2": +1, "p3": +1, "orb": +1, "ft": +1, "ast": 0, "stl": -1, "blk": -1}


def _net(z: np.ndarray, mu: np.ndarray) -> float:
    from .simulate import expected_efficiency

    zdef = z[np.maximum(DI, 0)] * (DI >= 0)
    return float(100 * (expected_efficiency((mu + z[:S])[None])[0] - expected_efficiency((mu + zdef)[None])[0]))


def skill_shares(z0: np.ndarray, z1: np.ndarray, mu: np.ndarray) -> list[dict]:
    """Net-rating change from moving each skill alone from its pre-game to its post-game value."""
    from .teams import SKILLS, column

    base = _net(z0, mu)
    out = []
    for side, stat, hi, lo, good in SKILLS:
        c = column(side, stat)
        z = z0.copy(); z[c] = z1[c]
        d = _net(z, mu) - base
        out.append({"side": side, "stat": stat, "label": RATE_WORDS[stat][2] if side == "off" else f"defending {RATE_WORDS[stat][2]}",
                    "change": float(z1[c] - z0[c]), "points": d})
    return out


def evidence(za: np.ndarray, zb: np.ndarray, base, home: float, k: np.ndarray, n: np.ndarray) -> list[dict]:
    """For team a's offense (against b's defense) and b's offense (against a's defense): expected count of
    each rate given the attempts, at both teams' pre-game skills, against the real count."""
    mu, hfa, phi = base.mu.mean(0), base.hfa.mean(0), base.phi.mean(0)
    out = []
    for side, zo, zd, h, j in (("off", za, zb, home, 0), ("def", zb, za, -home, 1)):
        zdef = np.where(DI >= 0, zd[np.maximum(DI, 0)], 0.0)
        p = 1 / (1 + np.exp(-(mu + zo[:S] + zdef + hfa * h)))
        for s_i, stat in enumerate(STATS[:-1]):                       # fouls are not a rate
            kk, nn, pp = float(k[j, s_i]), float(n[j, s_i]), float(p[s_i])
            if nn <= 0:
                continue
            exp = nn * pp
            sd = np.sqrt(nn * pp * (1 - pp) * (1 + (nn - 1) / (phi[s_i] + 1)))   # beta-binomial spread
            good = OFFENSE_GOOD[stat] * (1 if side == "off" else -1)    # for team a
            out.append({"side": side, "stat": stat, "count": kk, "of": nn, "expected": exp, "z": float((kk - exp) / max(sd, 1e-9)),
                        "good_for_team": float(np.sign(good) * np.sign(kk - exp)) if good else 0.0})
    return out


def sentence(team: str, opponent: str, dnet: float, shares: list[dict], ev: list[dict]) -> dict:
    """Plain-language summary: the rating change, the skills that drove it, the stats behind it."""
    tops = sorted(shares, key=lambda s: -abs(s["points"]))
    drivers = [s for s in tops if abs(s["points"]) >= 0.1][:3]
    surprises = sorted([e for e in ev if e["good_for_team"] != 0], key=lambda e: -abs(e["z"]))[:3]

    def ev_text(e):
        w = RATE_WORDS[e["stat"]]
        who = team if e["side"] == "off" else opponent
        pct = lambda x: f"{100 * x / e['of']:.0f}%"
        return (f"{who} {w[0]}: {e['count']:.0f} of {e['of']:.0f} ({pct(e['count'])}), "
                f"where the model expected {e['expected']:.1f} ({pct(e['expected'])})")
    head = f"Rating {'+' if dnet >= 0 else '−'}{abs(dnet):.1f}."
    if drivers:
        up = [f"{d['label']} ({d['points']:+.1f})" for d in drivers if d["points"] > 0]
        down = [f"{d['label']} ({d['points']:+.1f})" for d in drivers if d["points"] < 0]
        parts = []
        if up:
            parts.append("Up most on " + ", ".join(up))
        if down:
            parts.append(("down on " if up else "Down most on ") + ", ".join(down))
        head += " " + "; ".join(parts) + "."
    return {"headline": head, "evidence": [ev_text(e) for e in surprises],
            "drivers": drivers, "surprises": surprises}


def explain_update(team_a: str, team_b: str, za0, za1, zb0, zb1, base, home: float, k, n, dnets: dict | None = None) -> dict:
    """Explanations for both teams of one game's update (``k``, ``n``: each team's successes and trials,
    row 0 = a's offense, row 1 = b's offense)."""
    mu = base.mu.mean(0)
    out = {}
    for t, o, z0, z1, zo, kk, nn, h in ((team_a, team_b, za0, za1, zb0, k, n, home),
                                        (team_b, team_a, zb0, zb1, za0, k[::-1], n[::-1], -home)):
        shares = skill_shares(np.asarray(z0, float), np.asarray(z1, float), mu)
        ev = evidence(np.asarray(z0, float), np.asarray(zo, float), base, h, np.asarray(kk), np.asarray(nn))
        dnet_skills = _net(np.asarray(z1, float), mu) - _net(np.asarray(z0, float), mu)
        dnet = (dnets or {}).get(t, dnet_skills)                    # the change the update log shows, when given
        s = sentence(t, o, dnet, shares, ev)
        out[t] = {"opponent": o, "dnet": dnet, "dnet_mean_skills": dnet_skills, **s}
    return out


# --- storage ------------------------------------------------------------------

def path(season: int):
    return model.artifact_dir(season) / "explanations.jsonl"


def record(season: int, game_id: int, date, expl: dict) -> None:
    """Append one game's explanations (one line per team); a later line for the same game and team wins."""
    p = path(season)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        for team, e in expl.items():
            f.write(json.dumps({"game_id": int(game_id), "date": str(date)[:10], "team": team, **e}, default=float) + "\n")


def load(season: int, team: str | None = None, game_id: int | None = None) -> dict:
    """{(game_id, team): explanation}, the latest line for each."""
    p = path(season)
    out = {}
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if (team is None or e["team"] == team) and (game_id is None or e["game_id"] == game_id):
            out[(e["game_id"], e["team"])] = e
    return out
