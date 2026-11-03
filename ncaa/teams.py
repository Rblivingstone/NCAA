"""Team analysis: what a team is good at, what it is vulnerable to, and how it is trending.

Everything is read off the posterior over the team's 22 latent skills.

Strengths and weaknesses.  For each skill, its contribution is how many
points per 100 possessions the team's efficiency (offense) or opponents'
efficiency (defense) would change if that one skill were average, at the
team's other skills.  Reported with the team's percentile among D1 teams and
a plain-language label ("protects the ball", "sends opponents to the line").

Matchup vulnerabilities.  An opponent *style* is an average opponent made one
standard deviation (across teams) better at one thing: elite 3-point
shooting, crashing the offensive glass, heavy fouling, ball pressure, tempo...
The team's expected margin against that opponent minus against a plain
average opponent is how much the style hurts it; subtracting the same change
for an average team leaves what is specific to this team, i.e. a
vulnerability (negative) or a style it handles well (positive).  This is
computed over posterior draws, so each comes with the probability that the
team really is more (or less) exposed than an average team.
"""
from __future__ import annotations

import numpy as np

from .model import PACE, S, STATS, Posterior, def_index
from .simulate import expected_efficiency

DI = def_index()

# (latent column, label when high, label when low, good direction: +1 high is good for the team, -1 low is good, 0 style)
SKILLS = [
    ("off", "to", "Turns the ball over", "Protects the ball", -1),
    ("off", "ftr", "Gets to the free-throw line", "Rarely draws shooting fouls", +1),
    ("off", "three", "Three-point heavy offense", "Plays inside-out, few threes", 0),
    ("off", "p2", "Finishes well inside the arc", "Struggles to score inside", +1),
    ("off", "p3", "Shoots threes well", "Shoots threes poorly", +1),
    ("off", "orb", "Crashes the offensive glass", "Gets few second chances", +1),
    ("off", "ft", "Makes free throws", "Misses free throws", +1),
    ("off", "ast", "Moves the ball (assisted offense)", "Isolation-heavy offense", 0),
    ("off", "stl", "Prone to live-ball steals", "Rarely gets picked off", -1),
    ("off", "blk", "Gets shots blocked", "Rarely blocked", -1),
    ("def", "to", "Forces turnovers", "Rarely forces turnovers", +1),
    ("def", "ftr", "Sends opponents to the line", "Defends without fouling", -1),
    ("def", "three", "Gives up a lot of threes", "Runs teams off the 3-point line", 0),
    ("def", "p2", "Soft interior defense", "Protects the rim", -1),
    ("def", "p3", "Gives up good 3-point shooting", "Defends the arc", -1),
    ("def", "orb", "Gives up offensive rebounds", "Controls the defensive glass", -1),
    ("def", "stl", "Generates steals", "Few steals", +1),
    ("def", "blk", "Blocks shots", "Few blocks", +1),
]

# Opponent styles: (name, description, latent column of the *opponent*, direction of "better").
STYLES = [
    ("Elite 3-point shooting", "an opponent that makes threes at a high rate", ("off", "p3"), +1),
    ("Three-point volume", "an opponent that lives behind the arc", ("off", "three"), +1),
    ("Interior scoring", "an opponent that finishes well inside", ("off", "p2"), +1),
    ("Offensive rebounding", "an opponent that crashes the glass", ("off", "orb"), +1),
    ("Drawing fouls", "an opponent that gets to the line", ("off", "ftr"), +1),
    ("Ball pressure", "a defense that forces turnovers", ("def", "to"), +1),
    ("Heavy fouling", "a defense that fouls a lot (sends you to the line)", ("def", "ftr"), +1),
    ("Rim protection", "a defense that shuts down two-point shots", ("def", "p2"), -1),
    ("Perimeter defense", "a defense that contests threes", ("def", "p3"), -1),
    ("Defensive rebounding", "a defense that limits second chances", ("def", "orb"), -1),
]


def column(side: str, stat: str | None) -> int:
    if side == "pace":
        return PACE
    j = STATS.index(stat)
    return j if side == "off" else int(DI[j])


def _eta(mu, zo, zd):
    zdef = zd[..., np.maximum(DI, 0)] * (DI >= 0)
    return mu + zo[..., :S] + zdef


def _net(mu, za, zb):
    """Expected net efficiency of a vs b, points per 100 possessions (tempo-free)."""
    return 100 * (expected_efficiency(_eta(mu, za, zb)) - expected_efficiency(_eta(mu, zb, za)))


def team_index(post: Posterior, team: str) -> int:
    return post.teams.index(team)


def skill_profile(post: Posterior, team: str, draws: int = 300) -> list[dict]:
    """Each skill's contribution (points per 100 possessions), percentile, and label."""
    rng = np.random.default_rng(0)
    k = rng.choice(post.n_draws, size=min(draws, post.n_draws), replace=False)
    i = team_index(post, team)
    z = post.z[k, i].astype(float)                                   # draws x D
    mu = post.mu[k]
    d1 = [j for j, t in enumerate(post.teams) if t != "Non-D1"]
    zbar = post.z.mean(0)[d1]                                        # teams x D
    out = []
    for side, stat, hi, lo, good in SKILLS:
        c = column(side, stat)
        z0 = z.copy()
        z0[:, c] = 0.0
        if side == "off":
            base = expected_efficiency(mu + z[:, :S])
            alt = expected_efficiency(mu + z0[:, :S])
            impact = 100 * (base - alt)                              # + = helps the team's offense
        else:
            def opp_eff(zz):
                zdef = zz[:, np.maximum(DI, 0)] * (DI >= 0)
                return expected_efficiency(mu + zdef)
            impact = 100 * (opp_eff(z0) - opp_eff(z))                # + = holds opponents down
        value = float(z[:, c].mean())
        above = float((zbar[:, c] < value).mean())
        # Share of D1 teams this team is better than (for style skills: "more than").
        better = above if good >= 0 else 1 - above
        out.append({"side": side, "stat": stat, "label": hi if value > 0 else lo,
                    "percentile": better, "impact": float(impact.mean()),
                    "impact_lo": float(np.quantile(impact, 0.05)), "impact_hi": float(np.quantile(impact, 0.95)),
                    "style": good == 0})
    return out


def style_vulnerabilities(post: Posterior, team: str, draws: int = 300) -> list[dict]:
    """How much each opponent style hurts this team beyond how it hurts an average team."""
    rng = np.random.default_rng(1)
    k = rng.choice(post.n_draws, size=min(draws, post.n_draws), replace=False)
    i = team_index(post, team)
    zt = post.z[k, i].astype(float)
    mu = post.mu[k]
    avg = np.zeros_like(zt)
    d1 = [j for j, t in enumerate(post.teams) if t != "Non-D1"]
    sd = post.z.mean(0)[d1].std(0)
    out = []
    for name, desc, (side, stat), sign in STYLES:
        c = column(side, stat)
        opp = np.zeros_like(zt)
        opp[:, c] = sign * sd[c]
        d_team = _net(mu, zt, opp) - _net(mu, zt, avg)
        d_avg = _net(mu, avg, opp) - _net(mu, avg, avg)
        rel = d_team - d_avg
        out.append({"style": name, "description": desc, "margin_change": float(d_team.mean()),
                    "relative": float(rel.mean()), "relative_lo": float(np.quantile(rel, 0.05)),
                    "relative_hi": float(np.quantile(rel, 0.95)), "p_worse": float((rel < 0).mean())})
    return sorted(out, key=lambda r: r["relative"])


def tempo_note(post: Posterior, team: str) -> dict:
    """The team's own tempo and what one SD slower / faster games do to its edge over an average team."""
    i = team_index(post, team)
    zbar = post.z.mean(0)
    d1 = [j for j, t in enumerate(post.teams) if t != "Non-D1"]
    sd = zbar[d1, PACE].std()
    mu, z = post.mu.mean(0), zbar[i]
    poss = float(np.exp(post.mu_pace.mean() + z[PACE]))
    net = float(_net(mu[None], z[None], np.zeros((1, len(z))))[0])
    base = float(np.exp(post.mu_pace.mean() + z[PACE]))
    slow = float(np.exp(post.mu_pace.mean() + z[PACE] - sd))
    return {"possessions": poss, "percentile": float((zbar[d1, PACE] < z[PACE]).mean()),
            "margin_per_game": net * base / 100, "slow_cost": net * (slow - base) / 100}


def assessment(profile: list[dict], styles: list[dict], rating: dict, tempo: dict | None = None) -> list[str]:
    """A few plain sentences summarising the team."""
    def fmt(r):
        return f"{r['label'].lower()} ({r['impact']:+.1f} pts/100, better than {round(100 * r['percentile'])}% of D1)"
    ranked = sorted([r for r in profile if not r["style"]], key=lambda r: r["impact"])
    strengths = [r for r in ranked[::-1] if r["impact"] > 0.5][:3]
    weaknesses = [r for r in ranked if r["impact"] < -0.5][:2]
    lines = [f"Rated {rating['net']:+.1f} points per 100 possessions better than an average D1 team "
             f"(#{rating['rank']}; offense {rating['ortg']:.1f}, defense {rating['drtg']:.1f})."]
    if strengths:
        lines.append("Strengths: " + "; ".join(fmt(r) for r in strengths) + ".")
    if weaknesses:
        lines.append("Weaknesses: " + "; ".join(fmt(r) for r in weaknesses) + ".")
    hurt = [s for s in styles if s["relative"] < -0.1 and s["p_worse"] > 0.8][:2]
    likes = [s for s in styles[::-1] if s["relative"] > 0.1 and s["p_worse"] < 0.2][:2]
    if hurt:
        lines.append("Most exposed to " + " and ".join(f"{s['description']} ({s['relative']:+.1f} pts/100 vs an average team)"
                                                         for s in hurt) + ".")
    if likes:
        lines.append("Handles " + " and ".join(f"{s['description']} better than most ({s['relative']:+.1f})" for s in likes) + ".")
    if tempo:
        lines.append(f"Plays at {tempo['possessions']:.0f} possessions per 40 minutes (faster than {round(100 * tempo['percentile'])}% of teams). "
                     + (f"An opponent that slows the game by one standard deviation trims its expected edge over an average team "
                        f"by {abs(tempo['slow_cost']):.1f} points." if tempo["margin_per_game"] > 0 else
                        f"Slower games help it hang around: one SD slower narrows its expected deficit by {abs(tempo['slow_cost']):.1f} points."))
    return lines


def form_history(post: Posterior, team: str, end_date, period_days: int = 14) -> list[dict]:
    """Within-season form (offense + defense, pts/100 relative to now) by two-week period, oldest first."""
    import pandas as pd

    i = team_index(post, team)
    fm, lo, hi = post.form_mean[i], post.form_lo[i], post.form_hi[i]      # 2 x periods
    out = []
    for p in range(fm.shape[1] - 1, -1, -1):
        date = pd.Timestamp(end_date) - pd.Timedelta(days=period_days * p)
        out.append({"date": date.strftime("%Y-%m-%d"), "offense": float(fm[0, p]), "defense": float(fm[1, p]),
                    "net": float(fm[:, p].sum()), "net_lo": float(lo[:, p].sum()), "net_hi": float(hi[:, p].sum())})
    return out
