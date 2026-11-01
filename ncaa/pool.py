"""Brackets that maximise the chance of winning a pool.

A pool is not won by maximising expected points but by beating every other
entry, so a pick is worth most when it is right *and* few rivals made it.
That only pays where the public's picks differ from the true odds, so the
rivals must be simulated from a model of how people **pick**, not from the
game model (whose simulated brackets agree with it everywhere, and so
are best answered by the chalk bracket).

Public model (a priori: no pick data is needed).  Each simulated rival
advances team ``a`` over ``b`` with probability sigmoid(kappa * (u_a - u_b)),
where the public's perceived strength mixes our ratings with seeding,

    u = alpha * m + (1 - alpha) * s,

``m`` being each team's average head-to-head log-odds under the model and
``s`` a seed-line strength scaled to the same spread.  ``alpha`` (how well
informed the public is) and ``kappa`` (how chalky: >1 means favourites are
advanced more often than they really win) are uncertain, so every rival draws
its own pair from a range; the pool is a mixture over plausible publics.

Objective.  Given a simulated tournament, rivals' scores are independent
draws from the public's score distribution F there, so the chance of
finishing first against N rivals, splitting ties evenly, is exact:

    P(win | t) = [(F_< + f)^(N+1) - F_<^(N+1)] / ((N + 1) f),

with F_< and f the shares of rivals scoring below / exactly our score.
Averaged over simulated tournaments this is P(win the pool).

Search.  Candidates come from the bracket DP with per-pick value
points x p x (p / q)^lambda, p the model's probability that the team wins
that game and q the public's pick share: lambda = 0 is the max-expected-
points bracket, larger lambda leans towards under-picked teams.  Each
lambda is tried with the best overall champion and with each of the top
candidate champions held fixed.  Candidates are chosen on half of the
simulated tournaments and the winner is reported on the other half, so the
reported chance is not flattered by the selection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bracket import ESPN_POINTS, Bracket, Constraints, _dp, _win_probabilities

LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)


@dataclass(frozen=True)
class PublicModel:
    alpha: tuple[float, float] = (0.3, 0.8)   # weight on team quality vs seed line
    kappa: tuple[float, float] = (1.0, 2.0)   # chalkiness (1 = as confident as the model)


def strengths(bracket: Bracket, Wm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Model strength m (mean head-to-head log-odds) and seed strength s on the same scale."""
    T = len(bracket.teams)
    L = np.log(Wm / (1 - Wm))
    m = (L.sum(1) - np.diag(L)) / (T - 1)
    seeds = np.array([bracket.seed[t] for t in bracket.teams], dtype=float)
    s = -(seeds - seeds.mean())
    s *= m.std() / s.std()
    return m, s


def simulate_public(bracket: Bracket, Wm: np.ndarray, n: int, public: PublicModel = PublicModel(),
                    seed: int = 0) -> np.ndarray:
    """``n`` brackets filled in the way the public might: n x games array of picked winners."""
    rng = np.random.default_rng(seed)
    m, s = strengths(bracket, Wm)
    alpha = rng.uniform(*public.alpha, size=(n, 1))
    kappa = rng.uniform(*public.kappa, size=n)
    u = alpha * m + (1 - alpha) * s                               # n x T perceived strength
    picks = np.empty((n, len(bracket.games)), dtype=np.int16)
    rows = np.arange(n)
    for i, g in enumerate(bracket.games):
        a, b = (np.full(n, v) if kind == "team" else picks[:, bracket.gidx[v]] for kind, v in g.sources)
        p_a = 0.5 * (1 + np.tanh(kappa * (u[rows, a] - u[rows, b]) / 2))  # stable sigmoid
        picks[:, i] = np.where(rng.random(n) < p_a, a, b)
    return picks


def _game_points(bracket: Bracket, points=ESPN_POINTS) -> np.ndarray:
    return np.array([points[g.round] for g in bracket.games], dtype=np.int32)


def score_brackets(picks: np.ndarray, winners: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Pool scores: tournaments x brackets (``picks`` is brackets x games)."""
    out = np.zeros((winners.shape[0], picks.shape[0]), dtype=np.int32)
    for g in range(picks.shape[1]):
        if pts[g]:
            out += (winners[:, g, None] == picks[None, :, g]) * pts[g]
    return out


class Pool:
    """Simulated tournaments and a simulated public, ready to evaluate entries."""

    def __init__(self, bracket: Bracket, sims: dict, Wm: np.ndarray, summary: dict,
                 public: PublicModel = PublicModel(), n_public: int = 3000, seed: int = 0):
        self.bracket, self.public = bracket, public
        self.pts = _game_points(bracket)
        self.unit = int(np.gcd.reduce(self.pts[self.pts > 0]))
        self.winners = sims["winners"].astype(np.int16)
        self.public_picks = simulate_public(bracket, Wm, n_public, public, seed)
        T = len(bracket.teams)
        # q[g, t]: share of the public picking t to win game g.
        self.q = np.stack([np.bincount(self.public_picks[:, i], minlength=T) / n_public
                           for i in range(len(bracket.games))])
        self.p = _win_probabilities(bracket, summary)               # model: P(t wins game g)
        # Per tournament, the distribution of rival scores (in units of 10 points).
        opp = score_brackets(self.public_picks, self.winners, self.pts) // self.unit
        V = int(self.pts.sum() // self.unit) + 1
        K = opp.shape[0]
        hist = np.bincount((np.arange(K)[:, None] * V + opp).ravel(), minlength=K * V).reshape(K, V)
        self.f = hist / n_public                                     # P(rival score == v)
        self.below = np.cumsum(self.f, axis=1) - self.f              # P(rival score < v)
        half = K // 2
        self.select, self.report = np.arange(half), np.arange(half, K)

    def scores(self, picks: dict[str, int]) -> np.ndarray:
        """Our score in each simulated tournament."""
        vec = np.array([picks[g.id] for g in self.bracket.games])
        return score_brackets(vec[None, :], self.winners, self.pts)[:, 0]

    def win_probability(self, picks: dict[str, int], n_rivals: int, which=None) -> float:
        """P(finishing first, ties split) against ``n_rivals`` public entries."""
        rows = self.report if which is None else which
        v = self.scores(picks)[rows] // self.unit
        lo, f = self.below[rows, v], self.f[rows, v]
        N = n_rivals
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(f > 1e-12, ((lo + f) ** (N + 1) - lo ** (N + 1)) / ((N + 1) * f), lo ** N)
        return float(share.mean())

    def leverage_bracket(self, lam: float, c: Constraints = Constraints()) -> dict[str, int]:
        """DP bracket maximising sum of points x p x (p / q)^lam."""
        T = len(self.bracket.teams)
        ratio = (self.p + 1e-6) / (self.q + 0.01)

        def value(g):
            i = self.bracket.gidx[g.id]
            v = ESPN_POINTS[g.round] * self.p[i] * ratio[i] ** lam
            return np.repeat(v[:, None], T, axis=1)

        return _dp(self.bracket, value, c)[0]


def best_pool_bracket(pool: Pool, n_rivals: int, summary: dict, n_champions: int = 8) -> dict:
    """Search leverage brackets (with and without fixed champions) for the best chance to win."""
    br = pool.bracket
    title = np.array(summary["reach"])[:, -1]
    champs = [None] + [int(t) for t in np.argsort(-title)[:n_champions]]
    cands = []
    for champ in champs:
        c = Constraints(forced={} if champ is None else {"R6": champ})
        for lam in LAMBDAS:
            picks = pool.leverage_bracket(lam, c)
            if any(picks == x["picks"] for x in cands):
                continue
            cands.append({"picks": picks, "lambda": lam, "fixed_champion": champ,
                          "p_select": pool.win_probability(picks, n_rivals, pool.select)})
    best = max(cands, key=lambda x: x["p_select"])
    ep = next(x for x in cands if x["lambda"] == 0.0 and x["fixed_champion"] is None)
    return {
        "picks": best["picks"], "lambda": best["lambda"],
        "p_win": pool.win_probability(best["picks"], n_rivals),
        "p_win_max_points": pool.win_probability(ep["picks"], n_rivals),
        "max_points_picks": ep["picks"],
        "baseline": 1 / (n_rivals + 1),
        "candidates": len(cands),
    }
