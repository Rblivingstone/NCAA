"""Bayesian generative model of an NCAA basketball game.

Every team ``i`` has a latent skill vector ``z_i`` (``D`` = 22 numbers): an
offensive and a defensive component for each part of a possession, plus a
tempo component.  The skill vectors share a multivariate normal prior whose
correlation matrix is learned (LKJ prior), so strength in one area informs
the others ("good teams tend to be good at most things").

When team ``o`` has the ball against team ``d`` the rate of every event is

    logit p_s = mu_s + off_{o,s} + def_{d,s} + hfa_s * home_o

and the box score is generated possession by possession:

    possessions   log(poss / 40 min) ~ Normal(mu_pace + pace_o + pace_d, sigma_pace)
    turnovers     TOV  ~ BetaBinomial(poss, p_to)
    each play     ends in a shooting-foul trip with p_ftr, otherwise a field goal attempt
    shot type     3PA  ~ BetaBinomial(FGA, p_three)
    makes         2PM  ~ BetaBinomial(2PA, p_2),  3PM ~ BetaBinomial(3PA, p_3)
    misses        OREB ~ BetaBinomial(misses, p_orb)  (an OREB starts a new play)
    free throws   FTM  ~ BetaBinomial(FTA, p_ft)
    assists       AST  ~ BetaBinomial(FGM, p_ast)
    steals        STL  ~ BetaBinomial(TOV, p_stl)     (credited to the defense)
    blocks        BLK  ~ BetaBinomial(misses, p_blk)  (credited to the defense)
    fouls         PF   ~ Poisson(poss * exp(eta_pf))  (committed by the defense)

The beta-binomial concentration ``phi_s`` captures night-to-night variation
(hot shooting nights, sloppy nights) beyond pure binomial noise; it matters
for getting upset rates right.  Points are not modelled directly: they are
whatever the simulated makes add up to.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Probability-type stats (logit link) in the order used by the latent vector.
STATS = ["to", "ftr", "three", "p2", "p3", "orb", "ft", "ast", "stl", "blk", "pf"]
S = len(STATS)
NO_DEFENSE = {"ft"}  # the defense has no say in free-throw percentage
OFF_IDX = list(range(S))
DEF_STATS = [s for s in STATS if s not in NO_DEFENSE]
D = S + len(DEF_STATS) + 1  # offense + defense + pace
PACE = D - 1
LABELS = [f"off_{s}" for s in STATS] + [f"def_{s}" for s in DEF_STATS] + ["pace"]

# Free-throw attempts per shooting-foul trip are 1/0.44: two shots plus the
# occasional and-one / three-shot foul.  Used by the possession estimate and the simulator.
FT_TRIP = 0.44

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


def artifact_dir(season: int) -> Path:
    return ARTIFACTS / str(season)


def def_index() -> np.ndarray:
    """Column of the latent vector holding each stat's defensive effect (-1: none)."""
    out, k = [], S
    for s in STATS:
        if s in NO_DEFENSE:
            out.append(-1)
        else:
            out.append(k)
            k += 1
    return np.array(out)


@dataclass
class ModelData:
    teams: list[str]
    off: np.ndarray       # offense team index per team-game row
    dfn: np.ndarray       # defense team index
    home: np.ndarray      # +1 offense at home, -1 away, 0 neutral
    counts: dict          # name -> (successes, trials) arrays
    k: np.ndarray         # successes stacked as (S-1) x rows, in STATS order (fouls excluded)
    n: np.ndarray         # trials, same shape
    pf: np.ndarray
    poss: np.ndarray      # estimated possessions (exposure for fouls)
    g_a: np.ndarray       # pace: one row per game
    g_b: np.ndarray
    log_pace: np.ndarray


def prepare(games: pd.DataFrame) -> ModelData:
    teams = sorted(set(games.team) | set(games.opponent))
    idx = {t: i for i, t in enumerate(teams)}
    g = games
    poss_team = g.fga - g.oreb + g.tov + FT_TRIP * g.fta
    opp_poss = g.opp_fga - g.opp_oreb + g.opp_tov + FT_TRIP * g.opp_fta
    poss = np.round(poss_team).astype(int).to_numpy()
    trips = np.round(FT_TRIP * g.fta).astype(int).to_numpy()
    fga, fgm, fg3a, fg3m = (g[c].to_numpy() for c in ("fga", "fgm", "fg3a", "fg3m"))
    miss = fga - fgm
    counts = {
        "to": (g.tov.to_numpy(), np.maximum(poss, g.tov.to_numpy())),
        "ftr": (trips, fga + trips),
        "three": (fg3a, fga),
        "p2": (fgm - fg3m, fga - fg3a),
        "p3": (fg3m, fg3a),
        "orb": (np.minimum(g.oreb.to_numpy(), miss), miss),
        "ft": (g.ftm.to_numpy(), g.fta.to_numpy()),
        "ast": (np.minimum(g.ast.to_numpy(), fgm), fgm),
        "stl": (np.minimum(g.opp_stl.to_numpy(), g.tov.to_numpy()), g.tov.to_numpy()),
        "blk": (np.minimum(g.opp_blk.to_numpy(), miss), miss),
    }
    # One pace observation per game (rows come in pairs).
    first = ~g.duplicated("game_id")
    minutes = 40 + 5 * g.ot
    pace = ((poss_team + opp_poss) / 2 * 40 / minutes)[first]
    return ModelData(
        teams=teams,
        off=g.team.map(idx).to_numpy(),
        dfn=g.opponent.map(idx).to_numpy(),
        home=g.home.to_numpy().astype(float),
        counts=counts,
        k=np.stack([counts[s][0] for s in STATS[:-1]]).astype(np.int32),
        n=np.stack([counts[s][1] for s in STATS[:-1]]).astype(np.int32),
        pf=g.opp_pf.to_numpy(),
        poss=poss.astype(float),
        g_a=g.team[first].map(idx).to_numpy(),
        g_b=g.opponent[first].map(idx).to_numpy(),
        log_pace=np.log(pace.to_numpy()),
    )


def base_rates(md: ModelData) -> np.ndarray:
    rates = []
    for s in STATS:
        if s == "pf":
            rates.append(np.log(md.pf.sum() / md.poss.sum()))
        else:
            k, n = md.counts[s]
            p = k.sum() / n.sum()
            rates.append(np.log(p / (1 - p)))
    return np.array(rates)


def numpyro_model(md: ModelData, prior_mu: np.ndarray):
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    T = len(md.teams)
    di = jnp.array(def_index())
    mu = numpyro.sample("mu", dist.Normal(jnp.array(prior_mu), 0.5))
    hfa = numpyro.sample("hfa", dist.Normal(0.0, 0.1).expand([S]))
    log_phi = numpyro.sample("log_phi", dist.Normal(5.0, 1.5).expand([S - 1]))
    tau = numpyro.sample("tau", dist.HalfNormal(0.3).expand([D]))
    L = numpyro.sample("L_omega", dist.LKJCholesky(D, concentration=2.0))
    eps = numpyro.sample("eps", dist.Normal(0.0, 1.0).expand([T, D]))
    z = numpyro.deterministic("z", eps @ (tau[:, None] * L).T)  # T x D, cov = diag(tau) Ω diag(tau)

    off, dfn, home = md.off, md.dfn, jnp.array(md.home)
    zd = jnp.concatenate([z, jnp.zeros((T, 1))], axis=1)  # column -1 -> zero effect
    eta = mu + z[off, :S] + zd[dfn][:, di] + hfa * home[:, None]  # rows x S

    # Beta-binomial log-likelihood for every count stat at once, written out so
    # that only the parameter-dependent lgamma terms are evaluated (the fit is
    # dominated by these).
    phi = jnp.exp(log_phi)[:, None]                       # (S-1) x 1
    p = jnp.clip(jax.nn.sigmoid(eta[:, :S - 1].T), 1e-5, 1 - 1e-5)  # (S-1) x rows
    a, b = phi * p, phi * (1 - p)
    k, n = jnp.asarray(md.k), jnp.asarray(md.n)
    lg = jax.scipy.special.gammaln
    table = lg(jnp.arange(md.n.max() + 1)[None, :] + phi)  # lgamma(n + phi)
    ll = lg(k + a) + lg(n - k + b) - lg(a) - lg(b) + lg(phi) - jnp.take_along_axis(table, n, axis=1)
    numpyro.factor("lik_counts", ll.sum())
    # Personal fouls: Poisson with possessions as exposure.
    log_rate = jnp.log(jnp.asarray(md.poss)) + eta[:, S - 1]
    numpyro.factor("lik_pf", (jnp.asarray(md.pf) * log_rate - jnp.exp(log_rate)).sum())

    mu_pace = numpyro.sample("mu_pace", dist.Normal(np.log(68.0), 0.2))
    sigma_pace = numpyro.sample("sigma_pace", dist.HalfNormal(0.2))
    numpyro.sample("obs_pace", dist.Normal(mu_pace + z[md.g_a, PACE] + z[md.g_b, PACE], sigma_pace),
                   obs=jnp.array(md.log_pace))


def init_values(md: ModelData) -> dict:
    """A sensible starting point: every team average, league-average rates."""
    return {"mu": base_rates(md), "hfa": np.zeros(S), "log_phi": np.full(S - 1, 5.0),
            "tau": np.full(D, 0.1), "L_omega": np.eye(D), "eps": np.zeros((len(md.teams), D)),
            "mu_pace": float(np.mean(md.log_pace)), "sigma_pace": 0.1}


def _init_strategy(md: ModelData):
    import jax.numpy as jnp
    from numpyro.infer import init_to_value

    return init_to_value(values={k: jnp.asarray(v, dtype=jnp.float64) for k, v in init_values(md).items()})


def _fit_vi(md: ModelData, key, steps: int, progress: bool):
    """Mean-field variational fit; also used to initialise NUTS."""
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoNormal
    from numpyro.optim import Adam

    guide = AutoNormal(numpyro_model, init_loc_fn=_init_strategy(md), init_scale=0.01)
    lr = lambda i: 1e-2 * 0.02 ** (i / steps)  # noqa: E731  (decays 50x over the run)
    svi = SVI(numpyro_model, guide, Adam(step_size=lr), Trace_ELBO())
    res = svi.run(key, steps, md, base_rates(md), progress_bar=progress)
    return guide, res


def fit(games: pd.DataFrame, season: int, method="nuts", num_warmup=500, num_samples=250, num_chains=4,
        seed=0, progress=True, max_tree_depth=8, vi_steps=10000) -> Path:
    """Fit the model and save the posterior draws needed for simulation.

    ``method="nuts"`` runs full MCMC (the default).  A variational fit is run
    first and used as the starting point and initial mass matrix, which makes
    NUTS warm-up far quicker and more reliable in ~8,000 dimensions.
    ``method="vi"`` stops after the variational fit: fast, but it understates
    posterior uncertainty.
    """
    os.environ.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={num_chains}")
    import jax
    import numpyro
    from numpyro.infer import Predictive

    # The log-likelihood sums ~120k terms; float32 rounding there wrecks the
    # Hamiltonian energy and stalls NUTS.
    jax.config.update("jax_enable_x64", True)
    numpyro.set_host_device_count(num_chains)
    md = prepare(games)
    key = jax.random.PRNGKey(seed)
    guide, vi = _fit_vi(md, key, vi_steps, progress)
    losses = np.asarray(vi.losses)
    diag = {"vi_final_loss": float(losses[-200:].mean()), "vi_steps": vi_steps}
    if method == "nuts":
        from numpyro.infer import MCMC, NUTS, init_to_value

        start = guide.median(vi.params)
        inv_mass = {(site,): np.asarray(vi.params[f"{site}_auto_scale"]).ravel() ** 2 for site in start}
        kernel = NUTS(numpyro_model, target_accept_prob=0.85, max_tree_depth=max_tree_depth,
                      init_strategy=init_to_value(values=start), inverse_mass_matrix=inv_mass)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                    chain_method="parallel" if jax.local_device_count() >= num_chains else "sequential",
                    progress_bar=progress)
        mcmc.run(jax.random.PRNGKey(seed + 1), md, base_rates(md), extra_fields=("diverging", "num_steps"))
        post = mcmc.get_samples()
        diag |= summarize_diagnostics(mcmc, mcmc.get_extra_fields())
    elif method == "vi":
        n = num_samples * num_chains
        post = Predictive(guide, params=vi.params, num_samples=n)(jax.random.PRNGKey(seed + 1), md, base_rates(md))
        post["z"] = Predictive(numpyro_model, posterior_samples=post, return_sites=["z"])(
            jax.random.PRNGKey(seed + 2), md, base_rates(md))["z"]
    else:
        raise ValueError(method)

    out = artifact_dir(season)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "posterior.npz",
        z=np.asarray(post["z"], dtype=np.float32),
        mu=np.asarray(post["mu"]), hfa=np.asarray(post["hfa"]),
        phi=np.exp(np.asarray(post["log_phi"])), tau=np.asarray(post["tau"]),
        mu_pace=np.asarray(post["mu_pace"]), sigma_pace=np.asarray(post["sigma_pace"]),
        omega=np.asarray(post["L_omega"] @ np.swapaxes(post["L_omega"], -1, -2)).mean(0),
    )
    diag["method"] = method
    (out / "teams.json").write_text(json.dumps(md.teams))
    (out / "diagnostics.json").write_text(json.dumps(diag, indent=1))
    return out


def summarize_diagnostics(mcmc, extra) -> dict:
    from numpyro.diagnostics import summary

    s = summary(mcmc.get_samples(group_by_chain=True), prob=0.9)
    keys = ["mu", "hfa", "log_phi", "tau", "mu_pace", "sigma_pace", "z"]
    out = {}
    for k in keys:
        r = np.asarray(s[k]["r_hat"]).ravel()
        e = np.asarray(s[k]["n_eff"]).ravel()
        out[k] = {"max_r_hat": float(np.nanmax(r)), "min_n_eff": float(np.nanmin(e))}
    out["divergences"] = int(np.asarray(extra["diverging"]).sum())
    out["mean_leapfrog_steps"] = float(np.asarray(extra["num_steps"]).mean())
    return out


@dataclass
class Posterior:
    teams: list[str]
    z: np.ndarray          # draws x T x D
    mu: np.ndarray         # draws x S
    hfa: np.ndarray        # draws x S
    phi: np.ndarray        # draws x (S-1)
    mu_pace: np.ndarray    # draws
    sigma_pace: np.ndarray
    omega: np.ndarray      # D x D posterior-mean correlation of the latent skills
    lead_beta: np.ndarray = None  # draws; change in 2nd-half margin per point of half-time lead

    def __post_init__(self):
        if self.lead_beta is None:
            self.lead_beta = np.zeros(self.n_draws)

    @property
    def n_draws(self) -> int:
        return self.z.shape[0]

    def team_index(self, names) -> np.ndarray:
        idx = {t: i for i, t in enumerate(self.teams)}
        return np.array([idx[n] for n in names])


def add_to_posterior(season: int, **arrays):
    path = artifact_dir(season) / "posterior.npz"
    npz = dict(np.load(path))
    npz.update(arrays)
    np.savez_compressed(path, **npz)


def load_posterior(season: int) -> Posterior:
    d = artifact_dir(season)
    npz = np.load(d / "posterior.npz")
    teams = json.loads((d / "teams.json").read_text())
    keys = ("z", "mu", "hfa", "phi", "mu_pace", "sigma_pace", "omega", "lead_beta")
    return Posterior(teams=teams, **{k: npz[k] for k in keys if k in npz})
