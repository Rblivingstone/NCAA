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
    tilt: float = 0.0                          # how much of the seed edge the entrant discounts (upset lovers)


# Bracket-picking archetypes.  Each rival draws its own alpha and kappa from its
# archetype's range, so a pool is a mixture of strategies, not one average entrant.
ARCHETYPES = {
    "casual": ("Goes mostly by seed, moderately confident", PublicModel((0.0, 0.4), (0.8, 1.5))),
    "informed": ("Follows ratings like this model's", PublicModel((0.8, 1.0), (0.9, 1.3))),
    "chalk": ("Picks the favourite nearly every time", PublicModel((0.2, 0.6), (2.5, 4.0))),
    "upset": ("Hunts upsets: discounts seeds, picks freely", PublicModel((0.3, 0.7), (0.3, 0.7), tilt=0.5)),
}


@dataclass(frozen=True)
class PoolSpec:
    """Who is in the pool.

    ``mix`` gives the share of each archetype (normalised).  ``rho`` in [0, 1)
    is how much rivals' picks move together beyond their shared ratings: each
    rival picks team a when  sqrt(rho)*Z_g + sqrt(1-rho)*Z_ig < Phi^-1(p_ig),
    with Z_g a pool-wide shock for that game (a narrative everyone bought) and
    Z_ig the rival's own noise.  Every rival's chance of each pick is exactly
    p_ig whatever rho is; rho only changes how often they make the *same*
    combinations, i.e. how similar the field's brackets are to each other.
    """
    mix: tuple = (("casual", 0.45), ("informed", 0.2), ("chalk", 0.25), ("upset", 0.1))
    rho: float = 0.0

    def shares(self) -> dict:
        tot = sum(v for _, v in self.mix) or 1.0
        return {k: v / tot for k, v in self.mix if v > 0}


def strengths(bracket: Bracket, Wm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Model strength m (mean head-to-head log-odds) and seed strength s on the same scale."""
    T = len(bracket.teams)
    L = np.log(Wm / (1 - Wm))
    m = (L.sum(1) - np.diag(L)) / (T - 1)
    seeds = np.array([bracket.seed[t] for t in bracket.teams], dtype=float)
    s = -(seeds - seeds.mean())
    s *= m.std() / s.std()
    return m, s


def simulate_public(bracket: Bracket, Wm: np.ndarray, n: int, public: PublicModel | PoolSpec = PublicModel(),
                    seed: int = 0, climates: int = 1) -> np.ndarray:
    """``n`` brackets filled in the way the public might: n x games array of picked winners.

    With a ``PoolSpec``, rivals come from its archetype mix, and their picks are
    correlated through ``climates`` independent pool-wide shocks: rival i belongs
    to climate i % climates, and rivals in the same climate share Z_g.
    """
    from scipy.special import ndtri

    rng = np.random.default_rng(seed)
    m, s = strengths(bracket, Wm)
    if isinstance(public, PoolSpec):
        names, w = zip(*public.shares().items())
        kind = rng.choice(len(names), size=n, p=np.array(w))
        models = [ARCHETYPES[k][1] for k in names]
        rho = public.rho
    else:
        kind, models, rho = np.zeros(n, dtype=int), [public], 0.0
    alpha = np.empty((n, 1))
    kappa = np.empty(n)
    tilt = np.empty((n, 1))
    for j, pm in enumerate(models):
        sel = kind == j
        alpha[sel, 0] = rng.uniform(*pm.alpha, size=sel.sum())
        kappa[sel] = rng.uniform(*pm.kappa, size=sel.sum())
        tilt[sel, 0] = pm.tilt
    u = alpha * m + (1 - alpha) * (1 - tilt) * s                    # n x T perceived strength
    G = len(bracket.games)
    shock = rng.standard_normal((climates, G))                      # pool-wide narrative per game
    climate = np.arange(n) % climates
    picks = np.empty((n, G), dtype=np.int16)
    rows = np.arange(n)
    for i, g in enumerate(bracket.games):
        a, b = (np.full(n, v) if kind_ == "team" else picks[:, bracket.gidx[v]] for kind_, v in g.sources)
        p_a = np.clip(0.5 * (1 + np.tanh(kappa * (u[rows, a] - u[rows, b]) / 2)), 1e-9, 1 - 1e-9)
        latent = np.sqrt(rho) * shock[climate, i] + np.sqrt(1 - rho) * rng.standard_normal(n)
        picks[:, i] = np.where(latent < ndtri(p_a), a, b)
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
    """Simulated tournaments and a simulated field of rivals, ready to evaluate entries.

    Rivals are split into ``climates`` groups that share a pool-wide shock.
    Simulated tournament t is paired with climate t % climates; given the
    tournament and the climate, rivals are independent, so finishing-position
    probabilities have closed forms for any pool size.
    """

    def __init__(self, bracket: Bracket, sims: dict, Wm: np.ndarray, summary: dict,
                 public: PublicModel | PoolSpec = PublicModel(), n_public: int = 3000, seed: int = 0,
                 climates: int | None = None):
        self.bracket, self.public = bracket, public
        self.pts = _game_points(bracket)
        self.unit = int(np.gcd.reduce(self.pts[self.pts > 0]))
        self.winners = sims["winners"].astype(np.int16)
        correlated = isinstance(public, PoolSpec) and public.rho > 0
        C = climates or (20 if correlated else 1)
        if correlated and climates is None:
            # Enough rivals per climate that the per-tournament score distribution
            # is well resolved in its upper tail (what decides first place).
            n_public = max(n_public, 400 * C)
        self.climates = C
        self.public_picks = simulate_public(bracket, Wm, n_public, public, seed, climates=C)
        T = len(bracket.teams)
        # q[g, t]: share of the public picking t to win game g.
        self.q = np.stack([np.bincount(self.public_picks[:, i], minlength=T) / n_public
                           for i in range(len(bracket.games))])
        self.p = _win_probabilities(bracket, summary)               # model: P(t wins game g)
        # Per tournament, the distribution of rival scores (in units of 10 points),
        # using only the rivals of the climate paired with that tournament.
        K = self.winners.shape[0]
        V = int(self.pts.sum() // self.unit) + 1
        self.f = np.zeros((K, V))
        climate_of_rival = np.arange(n_public) % C
        for c in range(C):
            ts = np.arange(c, K, C)
            riv = self.public_picks[climate_of_rival == c]
            opp = score_brackets(riv, self.winners[ts], self.pts) // self.unit
            hist = np.bincount((np.arange(len(ts))[:, None] * V + opp).ravel(), minlength=len(ts) * V)
            self.f[ts] = hist.reshape(len(ts), V) / len(riv)
        self.below = np.cumsum(self.f, axis=1) - self.f              # P(rival score < v)
        # Alternate tournaments (and so climates) between choosing and reporting.
        idx = np.arange(K)
        self.select, self.report = idx[(idx // C) % 2 == 0], idx[(idx // C) % 2 == 1]

    def scores(self, picks: dict[str, int]) -> np.ndarray:
        """Our score in each simulated tournament."""
        vec = np.array([picks[g.id] for g in self.bracket.games])
        return score_brackets(vec[None, :], self.winners, self.pts)[:, 0]

    def _position(self, picks, which=None):
        rows = self.report if which is None else which
        v = self.scores(picks)[rows] // self.unit
        return rows, v, self.below[rows, v], self.f[rows, v]

    def win_probability(self, picks: dict[str, int], n_rivals: int, which=None) -> float:
        """P(finishing first, ties split) against ``n_rivals`` public entries."""
        _, _, lo, f = self._position(picks, which)
        return float(_tie_split(lo, f, n_rivals).mean())

    def finish(self, picks: dict[str, int], n_rivals: int, which=None) -> dict:
        """P(first), P(top 10%), P(last) and the expected finishing percentile."""
        from scipy.stats import binom

        _, _, lo, f = self._position(picks, which)
        above = 1 - lo - f                                           # P(a rival beats us)
        N = n_rivals
        allowed = max(0, int(np.ceil(0.1 * (N + 1))) - 1)            # rivals allowed above us in the top 10%
        first = _tie_split(lo, f, N)
        # Ties count as half above; never below P(first), whose tie split is exact.
        top10 = first if allowed == 0 else np.maximum(binom.cdf(allowed, N, np.clip(above + f / 2, 0, 1)), first)
        return {"p_first": float(first.mean()),
                "p_top10": float(top10.mean()),
                "p_last": float(_tie_split(above, f, N).mean()),
                "beat_share": float((lo + f / 2).mean())}

    def winning_score(self, n_rivals: int, which=None) -> dict:
        """Distribution of the best rival score: what it takes to win this pool."""
        rows = self.report if which is None else which
        F = np.cumsum(self.f[rows], axis=1)                          # P(rival <= v)
        G = np.clip(F, 0, 1) ** n_rivals                             # P(max of N rivals <= v), per tournament
        pmf = np.diff(np.concatenate([np.zeros((len(rows), 1)), G], axis=1), axis=1)
        v = np.arange(F.shape[1]) * self.unit
        mix = G.mean(0)
        q = lambda p: float(v[np.searchsorted(mix, p)])  # noqa: E731
        return {"mean": float((pmf * v).sum(1).mean()), "median": q(0.5), "p90": q(0.9),
                "p10": q(0.1), "hist": pmf.mean(0).round(5).tolist(), "unit": self.unit}

    def leverage_bracket(self, lam: float, c: Constraints = Constraints()) -> dict[str, int]:
        """DP bracket maximising sum of points x p x (p / q)^lam."""
        T = len(self.bracket.teams)
        ratio = (self.p + 1e-6) / (self.q + 0.01)

        def value(g):
            i = self.bracket.gidx[g.id]
            v = ESPN_POINTS[g.round] * self.p[i] * ratio[i] ** lam
            return np.repeat(v[:, None], T, axis=1)

        return _dp(self.bracket, value, c)[0]


def _tie_split(lo, f, N):
    """P(all N rivals below us), splitting ties evenly: [(lo+f)^(N+1) - lo^(N+1)] / ((N+1) f)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(f > 1e-12, ((lo + f) ** (N + 1) - lo ** (N + 1)) / ((N + 1) * f), lo ** N)


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
