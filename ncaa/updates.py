"""Bayesian updates of team skills the moment a game goes final.

The full model is refit nightly.  Between refits, each team carries a
Gaussian belief over its 22 latent skills, N(m_i, S_i), initialised from the
current posterior (teams the posterior has never seen start at the
population prior N(0, diag(tau) Omega diag(tau))).

When a game between a and b finishes, its whole box score updates both
teams at once (assumed-density filtering): the likelihood is the model's own
(the beta-binomial counts for turnovers, shot mix, makes, rebounds, free
throws, assists, steals and blocks; Poisson fouls; log-normal tempo), with the
league-wide parameters held at their posterior means.  The posterior over
(z_a, z_b) is approximated by a Laplace fit (Newton to the mode, covariance
from the curvature there), and the cross-covariance it creates between a and
b is dropped, as is standard for this kind of filter.  A game's form offset
is zero (form is pinned at the present), so the latest games act on the
skills directly.

Every update is logged (``updates.csv``: each team's net rating and its
uncertainty before and after), and the nightly refit resets the state.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import model
from .model import D, PACE, S, Posterior, def_index

DI = def_index()


@dataclass
class TeamState:
    teams: list[str]
    mean: np.ndarray                 # T x D
    cov: np.ndarray                  # T x D x D
    applied: list = field(default_factory=list)   # game ids already folded in
    base: str = ""                   # which posterior the state started from
    base_count: int = 0              # games already in that posterior (applied counts beyond this are updates)

    def index(self, team: str) -> int:
        return self.teams.index(team)

    def add_team(self, team: str, prior_cov: np.ndarray) -> int:
        """A team the posterior has never seen starts at the population prior."""
        self.teams.append(team)
        self.mean = np.vstack([self.mean, np.zeros((1, D))])
        self.cov = np.concatenate([self.cov, prior_cov[None]], axis=0)
        return len(self.teams) - 1


def population_cov(season_dir) -> np.ndarray:
    """diag(tau) Omega diag(tau) at the posterior means: the spread of skills across teams."""
    npz = np.load(season_dir / "posterior.npz")
    tau = npz["tau"].mean(0) if "tau" in npz else np.full(D, 0.1)
    omega = npz["omega"] if "omega" in npz else np.eye(D)
    return tau[:, None] * omega * tau[None, :]


def init_state(post: Posterior, label: str, teams: list[str] | None = None,
               prior_cov: np.ndarray | None = None) -> TeamState:
    """Gaussian beliefs per team from the posterior draws, plus any extra teams at the prior."""
    z = post.z.astype(float)
    mean = z.mean(0)
    cov = np.einsum("kti,ktj->tij", z - mean, z - mean) / max(len(z) - 1, 1) + 1e-6 * np.eye(D)
    st = TeamState(list(post.teams), mean, cov, [], label)
    if teams:
        pc = prior_cov if prior_cov is not None else np.cov(mean.T) + 1e-6 * np.eye(D)
        for t in teams:
            if t not in st.teams:
                st.add_team(t, pc)
    return st


def state_path(season: int):
    return model.artifact_dir(season) / "team_state.npz"


def save_state(season: int, st: TeamState) -> None:
    path = state_path(season)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mean=st.mean, cov=st.cov)
    (path.parent / "team_state.json").write_text(json.dumps({"teams": st.teams, "applied": st.applied,
                                                             "base": st.base, "base_count": st.base_count}))


def load_state(season: int) -> TeamState | None:
    path = state_path(season)
    if not path.exists():
        return None
    meta = json.loads((path.parent / "team_state.json").read_text())
    npz = np.load(path)
    return TeamState(meta["teams"], npz["mean"], npz["cov"], meta["applied"], meta.get("base", ""),
                     meta.get("base_count", 0))


def state_posterior(st: TeamState, base: Posterior, draws: int = 400, seed: int = 0) -> Posterior:
    """A Posterior whose team skills are draws from the current Gaussian beliefs (for ratings and forecasts)."""
    import dataclasses

    rng = np.random.default_rng(seed)
    K = min(draws, base.n_draws)
    L = np.linalg.cholesky(st.cov)                                   # T x D x D
    eps = rng.standard_normal((K, len(st.teams), D))
    z = (st.mean[None] + np.einsum("tij,ktj->kti", L, eps)).astype(np.float32)
    k = np.arange(K)
    return dataclasses.replace(base, teams=list(st.teams), z=z, mu=base.mu[k], hfa=base.hfa[k], phi=base.phi[k],
                               mu_pace=base.mu_pace[k], sigma_pace=base.sigma_pace[k],
                               lead_beta=base.lead_beta[k], form_mean=None, form_lo=None, form_hi=None,
                               form_start=None)


# --- the game likelihood ----------------------------------------------------

def _game_arrays(box_a: dict, box_b: dict, minutes: float):
    from .ingame import _counts, possessions

    (ka, na), (kb, nb) = _counts(box_a, box_b), _counts(box_b, box_a)
    pa, pb = max(possessions(box_a), 1.0), max(possessions(box_b), 1.0)
    from .model import blowout_weight

    return {"k": np.stack([ka, kb]), "n": np.stack([na, nb]),
            "weight": float(blowout_weight(box_a["pts"] - box_b["pts"])),
            "pf": np.array([box_b["pf"], box_a["pf"]], dtype=float),   # fouls committed by the defense
            "poss": np.array([round(pa), round(pb)], dtype=float),
            "log_pace": float(np.log(0.5 * (pa + pb) * 40.0 / minutes))}


def _make_objective(base: Posterior):
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    mu, hfa = jnp.asarray(base.mu.mean(0)), jnp.asarray(base.hfa.mean(0))
    phi = jnp.asarray(base.phi.mean(0))
    mu_pace, sig_pace = float(base.mu_pace.mean()), float(base.sigma_pace.mean())
    di = jnp.asarray(DI)
    lg = jax.scipy.special.gammaln

    def loglik(x, g, home):
        za, zb = x[:D], x[D:]

        def eta(zo, zd, h):
            zdef = jnp.where(di >= 0, zd[jnp.maximum(di, 0)], 0.0)
            return mu + zo[:S] + zdef + hfa * h

        etas = jnp.stack([eta(za, zb, home), eta(zb, za, -home)])           # 2 x S
        p = jnp.clip(jax.nn.sigmoid(etas[:, :S - 1]), 1e-6, 1 - 1e-6)
        a, b = phi * p, phi * (1 - p)
        k, n = g["k"], g["n"]
        ll = (lg(k + a) + lg(n - k + b) - lg(a) - lg(b) + lg(phi) - lg(n + phi)).sum()
        log_rate = jnp.log(g["poss"]) + etas[:, S - 1]
        ll += (g["pf"] * log_rate - jnp.exp(log_rate)).sum()
        m = mu_pace + za[PACE] + zb[PACE]
        ll += -0.5 * ((g["log_pace"] - m) / sig_pace) ** 2
        return ll

    def objective(x, g, home, m0, P0):
        d = x - m0
        return g["weight"] * loglik(x, g, home) - 0.5 * d @ P0 @ d       # weight: the blowout damper

    grad = jax.jit(jax.grad(objective))
    hess = jax.jit(jax.hessian(objective))
    value = jax.jit(objective)
    return value, grad, hess


_OBJECTIVES: dict = {}


def laplace_update(st: TeamState, base: Posterior, a: str, b: str, home: float, box_a: dict, box_b: dict,
                   minutes: float = 40.0, iters: int = 30) -> dict:
    """Fold one finished game into the beliefs about teams a and b (in place)."""
    import jax.numpy as jnp

    key = id(base)
    if key not in _OBJECTIVES:
        _OBJECTIVES[key] = _make_objective(base)
    value, grad, hess = _OBJECTIVES[key]
    ia, ib = st.index(a), st.index(b)
    m0 = np.concatenate([st.mean[ia], st.mean[ib]])
    S0 = np.zeros((2 * D, 2 * D))
    S0[:D, :D], S0[D:, D:] = st.cov[ia], st.cov[ib]
    P0 = np.linalg.inv(S0)
    g = {k_: jnp.asarray(v) for k_, v in _game_arrays(box_a, box_b, minutes).items()}
    args = (g, float(home), jnp.asarray(m0), jnp.asarray(P0))
    x = m0.copy()
    f = float(value(jnp.asarray(x), *args))
    for _ in range(iters):                       # damped Newton ascent to the posterior mode
        G = np.asarray(grad(jnp.asarray(x), *args))
        H = np.asarray(hess(jnp.asarray(x), *args))
        step = np.linalg.solve(-H + 1e-9 * np.eye(2 * D), G)
        t = 1.0
        while t > 1e-4:
            f_new = float(value(jnp.asarray(x + t * step), *args))
            if f_new >= f - 1e-12:
                break
            t /= 2
        x, f_old, f = x + t * step, f, f_new
        if abs(f - f_old) < 1e-9:
            break
    H = np.asarray(hess(jnp.asarray(x), *args))
    cov = np.linalg.inv(-H)
    cov = 0.5 * (cov + cov.T)
    w, V = np.linalg.eigh(cov)                                       # guard against round-off
    cov = (V * np.maximum(w, 1e-8)) @ V.T
    st.mean[ia], st.mean[ib] = x[:D], x[D:]
    st.cov[ia], st.cov[ib] = cov[:D, :D], cov[D:, D:]
    return {"mode": x, "cov": cov}


def net_rating(st: TeamState, base: Posterior, team: str, draws: int = 300) -> tuple[float, float]:
    """Posterior mean and sd of a team's net rating (pts/100 vs an average team) under the current belief."""
    from .simulate import expected_efficiency

    i = st.index(team)
    rng = np.random.default_rng(i)
    z = rng.multivariate_normal(st.mean[i], st.cov[i], size=draws)
    mu = base.mu.mean(0)
    zdef = z[:, np.maximum(DI, 0)] * (DI >= 0)
    net = 100 * (expected_efficiency(mu + z[:, :S]) - expected_efficiency(mu + zdef))
    return float(net.mean()), float(net.std())


def apply_game(season: int, st: TeamState, base: Posterior, game_id: int, date, a: str, b: str, home: float,
               box_a: dict, box_b: dict, minutes: float, prior_cov: np.ndarray, source: str) -> dict | None:
    """Update the state with a finished game and log the rating change; skips games already applied."""
    if game_id in st.applied:
        return None
    for t in (a, b):
        if t not in st.teams:
            st.add_team(t, prior_cov)
    before = {t: net_rating(st, base, t) for t in (a, b)}
    z_before = {t: st.mean[st.index(t)].copy() for t in (a, b)}
    laplace_update(st, base, a, b, home, box_a, box_b, minutes)
    after = {t: net_rating(st, base, t) for t in (a, b)}
    try:                                         # "Why did the rating move?" (never blocks the update)
        from . import explain

        g = _game_arrays(box_a, box_b, minutes)
        expl = explain.explain_update(a, b, z_before[a], st.mean[st.index(a)], z_before[b], st.mean[st.index(b)],
                                      base, home, g["k"], g["n"], dnets={t: after[t][0] - before[t][0] for t in (a, b)})
        explain.record(season, game_id, date, expl)
    except Exception:
        pass
    st.applied.append(int(game_id))
    rows = [{"game_id": int(game_id), "date": pd.Timestamp(date).isoformat(), "team": t, "opponent": o,
             "pts": box_a["pts"] if t == a else box_b["pts"], "opp_pts": box_b["pts"] if t == a else box_a["pts"],
             "net_before": before[t][0], "sd_before": before[t][1], "net_after": after[t][0], "sd_after": after[t][1],
             "source": source} for t, o in ((a, b), (b, a))]
    _write_log_rows(updates_path(season), pd.DataFrame(rows))
    return {t: {"before": before[t], "after": after[t]} for t in (a, b)}


def _write_log_rows(log, new: pd.DataFrame) -> None:
    """Add a game's rows to the update log, replacing any earlier rows for the same game and team.

    A game can legitimately be folded in twice: once from the live feed, and again after a nightly refit
    whose box-score data didn't have it yet (the refit rebuilds every rating from that data, so the game must
    be applied again). The log keeps one row per game and team, the latest, so counts built on it (games,
    records, streaks) never see the game twice."""
    import os

    if log.exists():
        old = pd.read_csv(log)
        keep = ~(old.game_id.astype(int).isin(new.game_id.astype(int)) & old.team.isin(new.team))
        out = pd.concat([old[keep], new], ignore_index=True)
    else:
        out = new
    tmp = log.with_name(f".{log.name}.{os.getpid()}.tmp")
    out.to_csv(tmp, index=False)
    os.replace(tmp, log)


def apply_one_sided(st: TeamState, base: Posterior, school: str, opp: str, home: float, box_school: dict,
                    box_opp: dict, minutes: float, prior_cov: np.ndarray, pooled: str = "Non-D1") -> None:
    """Update a non-D1 school's own belief from a game against a D1 team, holding the D1 team at its
    pre-game belief. (The same game updates the D1 team through the pooled non-D1 rating, so it is not
    counted twice for the D1 team.) A school seen for the first time starts at the pooled non-D1 rating,
    the average of the non-D1 opponents played, with the between-team spread as its uncertainty."""
    if school not in st.teams:
        i = st.add_team(school, prior_cov)
        if pooled in st.teams:
            st.mean[i] = st.mean[st.index(pooled)]
    j = st.index(opp)
    keep_m, keep_c = st.mean[j].copy(), st.cov[j].copy()
    st.cov[j] = keep_c * 1e-4 + 1e-10 * np.eye(D)          # opponent effectively fixed
    try:
        laplace_update(st, base, school, opp, home, box_school, box_opp, minutes)
    finally:
        st.mean[j], st.cov[j] = keep_m, keep_c


def updates_path(season: int):
    return model.artifact_dir(season) / "updates.csv"


def load_updates(season: int) -> pd.DataFrame:
    """The update log, one row per game and team (the latest, if a game was folded in again after a refit)."""
    path = updates_path(season)
    if not path.exists():
        return pd.DataFrame()
    up = pd.read_csv(path)
    return up.drop_duplicates(["game_id", "team"], keep="last").reset_index(drop=True) if len(up) else up


def minutes_played(ot: int) -> float:
    return 40.0 + 5.0 * max(int(ot), 0)

