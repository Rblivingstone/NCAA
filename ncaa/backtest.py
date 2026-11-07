"""Ten tournaments, out of sample: the model against seeds, a KenPom-style efficiency rating, ESPN BPI and Vegas.

For every season the model is fitted on regular-season and conference-tournament games only, up to the
First Four (nothing from the tournament), each season on its own, and then forecasts the whole bracket at
Selection Sunday. The comparison sources:

* **Seeds**: P(a beats b) a logistic curve in the seed difference, fitted on the tournaments *before* that
  season (back to 2012), so it is out of sample too. Selection Sunday.
* **Efficiency margin** (KenPom-style, rebuilt here, not KenPom's own numbers): offensive and defensive points
  per 100 possessions adjusted for opponent and home court by ridge least squares on the same games the model
  sees; a game's margin is the rating gap at the two teams' average pace, its win chance a normal curve around
  it (sd 11 points). Selection Sunday.
* **ESPN BPI** and **Vegas** (consensus closing line, moneylines with the bookmaker's margin taken out, else the
  spread): read from ESPN's archived game files. Both are set just before each game, so they know earlier
  tournament results and late injury news; they are scored on games only, never on brackets.

Measures, per source and season:

* log loss and Brier score of the actual tournament games (all 67, and the 63 without the First Four), from the
  source's chance that the eventual winner would win that matchup;
* the source's most likely bracket (the same exact search for every Selection Sunday source): ESPN
  Tournament Challenge score, the score the source expected of it, and whether its champion won;
* a five-bracket portfolio: the most likely bracket for each of the source's five likeliest champions, scored
  by the best of the five.

Run ``python -m ncaa backtest`` (fetches, fits and simulates whatever is missing; about 15 minutes a season)
and the results land in ``results/backtest/<season>.json`` for the web page.
"""
from __future__ import annotations

import io
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, model
from .bracket import (ESPN_POINTS, Bracket, actual_picks, bracket_log_likelihood, espn_points, load_simulation,
                      most_likely_bracket)

SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]      # no tournament in 2020
SEED_HISTORY_FROM = 2012                     # the archive's first season with seeds and pre-game lines
RESULTS = Path(__file__).resolve().parent.parent / "results" / "backtest"
MARGIN_SD = 11.0
PORTFOLIO = 5
SOURCES = {
    "model": "This model",
    "efficiency": "Efficiency margin (KenPom-style)",
    "seed": "Seeds",
    "bpi": "ESPN BPI (game day)",
    "vegas": "Vegas (game day)",
}
BRACKET_SOURCES = ["model", "efficiency", "seed"]


def _phi(x):
    from math import erf, sqrt
    return 0.5 * (1 + np.vectorize(erf)(np.asarray(x) / sqrt(2)))


# --- comparison sources -----------------------------------------------------

def seed_curve(season: int, log=print) -> float:
    """Slope k of P(a beats b) = 1 / (1 + exp(-k (seed_b - seed_a))), fitted on tournaments before ``season``."""
    xs, ys = [], []
    for y in range(SEED_HISTORY_FROM, season):
        if y == 2020:
            continue
        try:
            s = data._fetch_parquet(f"{data.BASE}/schedules/parquet/mbb_schedule_{y}.parquet")
        except Exception as e:                                          # pragma: no cover - network
            log(f"seed history {y}: {e}")
            continue
        t = data._tournament_games(s)
        t = data.with_seeds(t[t.status_type_completed.fillna(False).astype(bool)], log)
        d = (t.away_current_rank - t.home_current_rank).to_numpy(float)     # home's seed advantage
        won = (t.home_score > t.away_score).to_numpy(float)
        ok = np.isfinite(d) & (d != 0)
        xs.append(d[ok]); ys.append(won[ok])
    x, yv = np.concatenate(xs), np.concatenate(ys)
    k = 0.15
    for _ in range(50):                                                 # Newton steps on the log-likelihood
        p = 1 / (1 + np.exp(-k * x))
        g = ((yv - p) * x).sum(); h = -(p * (1 - p) * x * x).sum()
        k -= g / h
    return float(k)


def efficiency_ratings(games: pd.DataFrame, ridge: float = 1.0) -> pd.DataFrame:
    """Opponent- and venue-adjusted offence and defence (points per 100 possessions) and pace, per team."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import lsqr

    g = games.copy()
    poss = lambda f: f.fga - f.oreb + f.tov + 0.475 * f.fta
    g["poss"] = 0.5 * (poss(g) + (g.opp_fga - g.opp_oreb + g.opp_tov + 0.475 * g.opp_fta))
    g = g[g.poss > 40]
    teams = sorted(set(g.team) | set(g.opponent))
    ix = {t: i for i, t in enumerate(teams)}
    T, n = len(teams), len(g)
    eff = (100 * g.pts / g.poss).to_numpy()
    loc = g.home.to_numpy(float) if "home" in g else np.zeros(n)
    rows = np.arange(n)
    # columns: offence (T), defence (T), home edge (1); intercept removed by centring
    A = csr_matrix((np.r_[np.ones(n), -np.ones(n), loc], (np.r_[rows, rows, rows],
                    np.r_[g.team.map(ix), T + g.opponent.map(ix), np.full(n, 2 * T)])), shape=(n, 2 * T + 1))
    mu = eff.mean()
    sol = lsqr(A, eff - mu, damp=np.sqrt(ridge))[0]                    # ridge: a light pull toward average
    off, dfn = sol[:T], sol[T:2 * T]
    pace = g.groupby("team").poss.mean().reindex(teams).fillna(g.poss.mean())
    return pd.DataFrame({"team": teams, "off": off, "def": dfn, "em": off + dfn, "pace": pace.to_numpy()})


def efficiency_matrix(br: Bracket, ratings: pd.DataFrame) -> np.ndarray:
    r = ratings.set_index("team").reindex(br.teams)
    em = r.em.fillna(r.em.min()).to_numpy(); pace = r.pace.fillna(68).to_numpy()
    margin = (em[:, None] - em[None, :]) * (pace[:, None] + pace[None, :]) / 200
    return _phi(margin / MARGIN_SD)


def seed_matrix(br: Bracket, k: float) -> np.ndarray:
    s = np.array([br.seed[t] for t in br.teams], dtype=float)
    return 1 / (1 + np.exp(-k * (s[None, :] - s[:, None])))


def _vegas_p_home(pc: dict) -> float | None:
    """Home win chance from the consensus line: moneylines with the margin removed, else the spread."""
    ml = lambda o: (o or {}).get("moneyLine")
    h, a = ml(pc.get("homeTeamOdds")), ml(pc.get("awayTeamOdds"))
    imp = lambda m: 100 / (m + 100) if m > 0 else -m / (-m + 100)
    try:
        if h is not None and a is not None:
            ph, pa = imp(float(h)), imp(float(a))
            return ph / (ph + pa)
        if pc.get("spread") is not None:
            return float(_phi(-float(pc["spread"]) / MARGIN_SD))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return None


def game_day_forecasts(game_ids, log=print) -> dict:
    """{game_id: {"home", "away", "bpi", "vegas"}} (home win chances) from ESPN's archived game files."""
    from .live import RAW_GAME_JSON, espn_pregame

    out = {}
    for gid in game_ids:
        try:
            with urllib.request.urlopen(RAW_GAME_JSON.format(game_id=int(gid)), timeout=60) as r:
                d = json.loads(r.read())
        except Exception as e:                                          # pragma: no cover - network
            log(f"game {gid}: {e}")
            continue
        comp = d["header"]["competitions"][0]
        order = sorted(comp["competitors"], key=lambda c: c["homeAway"] != "home")
        home, away = (c["team"]["location"] for c in order)
        pre = espn_pregame(d) if d.get("predictor") else None              # BPI only: not ESPN's tip-off chart
        vegas = next((v for v in (_vegas_p_home(pc) for pc in d.get("pickcenter") or []) if v is not None), None)
        out[int(gid)] = {"home": home, "away": away, "bpi": pre["p_home"] if pre else None, "vegas": vegas,
                         "line": next((pc.get("details") for pc in d.get("pickcenter") or []), None)}
    return out


# --- scoring ----------------------------------------------------------------

def advancement(br: Bracket, P: np.ndarray) -> np.ndarray:
    """Exact chance each team wins each game (games x teams) when every game follows the pairwise matrix ``P``."""
    T = len(br.teams)
    win = {}
    for g in br.games:
        sides = []
        for kind, v in g.sources:
            if kind == "team":
                x = np.zeros(T); x[v] = 1; sides.append(x)
            else:
                sides.append(win[v])
        a, b = sides
        win[g.id] = a * (P @ b) + b * (P @ a)
    return np.stack([win[g.id] for g in br.games])


def bracket_scores(br: Bracket, picks: dict, actual: dict, pwin: np.ndarray | None = None, sims=None) -> dict:
    named = {g: br.teams[t] for g, t in picks.items()}
    real = sum(ESPN_POINTS[br.games[br.gidx[g]].round] for g, t in named.items() if actual.get(g) == t)
    if sims is not None:
        exp = espn_points(br, named, sims, actual)["expected"]
    else:
        exp = float(sum(ESPN_POINTS[br.games[br.gidx[g]].round] * pwin[br.gidx[g], t] for g, t in picks.items()))
    return {"champion": named["R6"], "final_four": [named[g] for g in sorted(named) if g.startswith("R4")],
            "espn": float(real), "expected": exp, "champion_right": named["R6"] == actual.get("R6")}


def evaluate(season: int, log=print) -> dict:
    """Score every source on one season (the model's fit and simulation must exist)."""
    br = Bracket(data.load_bracket(season))
    actual = actual_picks(br)
    sims, summary, W = load_simulation(season)
    games = data.load_games(season)
    eff = efficiency_ratings(games)
    k = seed_curve(season, log)
    mats = {"model": W.mean(0), "efficiency": efficiency_matrix(br, eff), "seed": seed_matrix(br, k)}

    # the tournament's real games, with each source's chance for the winner
    sched = data.load_schedule(season)
    tg = sched[sched.tournament.astype(bool) & sched.completed.astype(bool)]
    gd = game_day_forecasts(tg.game_id.tolist(), log)
    rows = []
    for g in tg.itertuples():
        if g.home not in br.tidx or g.away not in br.tidx:
            continue
        hw = g.home_score > g.away_score
        w, l = (g.home, g.away) if hw else (g.away, g.home)
        wi, li = br.tidx[w], br.tidx[l]
        row = {"game_id": int(g.game_id), "winner": w, "loser": l, "seeds": [br.seed[w], br.seed[l]],
               "first_four": _is_first_four(br, w, l)}
        for name, M in mats.items():
            row[name] = float(M[wi, li])
        f = gd.get(int(g.game_id), {})
        for src in ("bpi", "vegas"):
            p = f.get(src)                                                # ESPN's home team's chance
            if p is None or f.get("home") not in (g.home, g.away):
                row[src] = None
                continue
            p_winner_side = p if f["home"] == w else 1 - p
            row[src] = float(p_winner_side)
        row["line"] = f.get("line")
        rows.append(row)

    # brackets: the most likely bracket of every Selection Sunday source, and its five-champion portfolio
    brackets = {}
    for name in BRACKET_SOURCES:
        Wsrc = W if name == "model" else mats[name][None]
        res = most_likely_bracket(br, Wsrc, n_champions=10)
        pwin = None if name == "model" else advancement(br, mats[name])
        best = bracket_scores(br, res["picks"], actual, pwin, sims if name == "model" else None)
        port = [bracket_scores(br, c["picks"], actual, pwin, sims if name == "model" else None)
                for c in res["by_champion"][:PORTFOLIO]]
        title = summary["reach"] if name == "model" else None
        p_title = (np.array(title)[:, -1] if title is not None else advancement(br, mats[name])[br.gidx["R6"]])
        brackets[name] = {**best, "log_likelihood": float(bracket_log_likelihood(br, res["picks"], Wsrc)),
                          "portfolio": [{"champion": p["champion"], "espn": p["espn"], "expected": p["expected"]}
                                        for p in port],
                          "portfolio_best": max(p["espn"] for p in port),
                          "title_favourite": br.teams[int(np.argmax(p_title))],
                          "p_title_favourite": float(np.max(p_title)),
                          "p_title_actual": float(p_title[br.tidx[actual["R6"]]]) if actual.get("R6") else None}
    return {"season": season, "champion": actual.get("R6"), "seed_slope": k, "n_games": len(rows),
            "games": rows, "brackets": brackets, "sources": SOURCES,
            "fit": json.loads((model.artifact_dir(season) / "diagnostics.json").read_text()).get("method"),
            "cutoff": data.load_bracket(season)["start"], "updated": pd.Timestamp.now(tz="UTC").isoformat()}


def _is_first_four(br: Bracket, a: str, b: str) -> bool:
    return any(g.round == 0 and {br.teams[v] for _, v in g.sources} == {a, b} for g in br.games)


# --- pipeline -----------------------------------------------------------------

def run(seasons=None, sims: int = 10000, force: bool = False, log=print) -> list[Path]:
    """Fetch, fit (variational, as the live site does) and simulate any season missing, then score it."""
    from .__main__ import main as cli

    out = []
    RESULTS.mkdir(parents=True, exist_ok=True)
    for season in seasons or SEASONS:
        t0 = time.time()
        if not (data.season_dir(season) / "bracket.json").exists():
            log(f"{season}: fetching")
            data.fetch(season)
        if force or not (model.artifact_dir(season) / "posterior.npz").exists():
            log(f"{season}: fitting")
            cli(["fit", "--season", str(season), "--method", "vi"])
        if force or not (model.artifact_dir(season) / "tournament_sims.npz").exists():
            log(f"{season}: simulating")
            cli(["simulate", "--season", str(season), "--sims", str(sims)])
        res = evaluate(season, log)
        p = RESULTS / f"{season}.json"
        p.write_text(json.dumps(res, default=float))
        log(f"{season}: scored in {time.time() - t0:.0f}s -> {p}")
        out.append(p)
    return out


def load_all() -> list[dict]:
    if not RESULTS.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))]


def summary(results: list[dict]) -> dict:
    """Totals per source: game-level accuracy on the games every source covers, brackets, head-to-head."""
    out = {"seasons": [r["season"] for r in results], "sources": SOURCES, "games": {}, "brackets": {}, "by_season": [],
           "coverage": {src: [r["season"] for r in results if any(g.get(src) is not None for g in r["games"])]
                        for src in SOURCES}}
    allg = [g for r in results for g in r["games"]]
    for scope, keep in (("all", lambda g: True), ("no_first_four", lambda g: not g["first_four"])):
        gs = [g for g in allg if keep(g)]
        res = {}
        for src in SOURCES:
            have = [g for g in gs if g.get(src) is not None]
            if not have:
                continue
            p = np.clip(np.array([g[src] for g in have]), 1e-6, 1 - 1e-6)
            m = np.clip(np.array([g["model"] for g in have]), 1e-6, 1 - 1e-6)
            res[src] = {"n": len(have), "log_loss": float(-np.log(p).mean()), "brier": float(((1 - p) ** 2).mean()),
                        "accuracy": float((p > 0.5).mean() + 0.5 * (p == 0.5).mean()),
                        # head to head with the model on exactly these games
                        "model_log_loss": float(-np.log(m).mean()), "model_brier": float(((1 - m) ** 2).mean()),
                        "ll_diff_se": float((np.log(p) - np.log(m)).std(ddof=1) / np.sqrt(len(have))) if len(have) > 1 else None}
        out["games"][scope] = res
    for src in BRACKET_SOURCES:
        b = [r["brackets"][src] for r in results if src in r["brackets"]]
        if not b:
            continue
        out["brackets"][src] = {
            "n": len(b), "champions_right": sum(x["champion_right"] for x in b),
            "title_favourite_right": sum(x["title_favourite"] == r["champion"] for x, r in zip(b, results)),
            "espn": float(np.mean([x["espn"] for x in b])), "expected": float(np.mean([x["expected"] for x in b])),
            "portfolio_best": float(np.mean([x["portfolio_best"] for x in b])),
            "portfolio_champion_right": sum(any(p["champion"] == r["champion"] for p in x["portfolio"])
                                            for x, r in zip(b, results)),
            "beats_model": sum(x["espn"] > r["brackets"]["model"]["espn"] for x, r in zip(b, results)),
            "ties_model": sum(x["espn"] == r["brackets"]["model"]["espn"] for x, r in zip(b, results))}
    for r in results:
        row = {"season": r["season"], "champion": r["champion"], "sources": {}}
        for src in SOURCES:
            have = [g for g in r["games"] if g.get(src) is not None]
            if have:
                p = np.clip(np.array([g[src] for g in have]), 1e-6, 1 - 1e-6)
                row["sources"][src] = {"n": len(have), "log_loss": float(-np.log(p).mean()),
                                       "brier": float(((1 - p) ** 2).mean())}
            if src in r["brackets"]:
                row["sources"].setdefault(src, {}).update({k: r["brackets"][src][k] for k in
                                                           ("espn", "expected", "champion", "champion_right",
                                                            "portfolio_best", "title_favourite", "p_title_actual")})
        out["by_season"].append(row)
    return out
