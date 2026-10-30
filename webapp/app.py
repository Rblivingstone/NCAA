"""Flask front end for the bracket simulator.

    python -m ncaa build --season 2026      # once: fetch, fit, simulate
    flask --app webapp.app run              # then browse http://127.0.0.1:5000
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np
from flask import Flask, abort, jsonify, render_template, request

from ncaa import data, model
from ncaa.bracket import (Bracket, Constraints, actual_picks, bracket_log_likelihood, espn_points, expected_points,
                          favourites, is_upset, load_simulation, max_expected_points_bracket, most_likely_bracket,
                          resolve_picks, score_bracket, simulate_tournament, upset_distribution)
from ncaa.evaluate import game_unusualness, tournament_unusualness
from ncaa.pool import Pool, PublicModel, best_pool_bracket
from ncaa.simulate import BOX_STATS, simulate_games

SEASON = int(os.environ.get("NCAA_SEASON", 2026))

app = Flask(__name__)
app.json.sort_keys = False  # keep box-score stats in their natural order


@lru_cache(maxsize=1)
def state():
    br = Bracket(data.load_bracket(SEASON))
    sims, summary, W = load_simulation(SEASON)
    post = model.load_posterior(SEASON)
    return br, sims, summary, W, post


@lru_cache(maxsize=1)
def actual() -> dict[str, str]:
    """The real tournament results as {game id: winner}."""
    return actual_picks(state()[0])


@app.route("/")
def index():
    return render_template("index.html", season=SEASON, page="bracket")


@app.route("/score")
def score_page():
    return render_template("score.html", season=SEASON, page="score")


@app.route("/matchup")
def matchup_page():
    return render_template("matchup.html", season=SEASON, page="matchup")


@app.route("/teams")
def teams_page():
    return render_template("teams.html", season=SEASON, page="teams")


@app.route("/api/summary")
def api_summary():
    return jsonify(state()[2])


@app.route("/api/actual")
def api_actual():
    return jsonify(actual())


@lru_cache(maxsize=1)
def compare() -> dict:
    """The real tournament set against the model's pre-tournament distribution.

    Per game: who actually occupied each slot and how likely the model thought
    that was, who won, the model's head-to-head probability for that actual
    matchup, and whether the model's favourite in it lost ("wrong").
    """
    br, _, summary, W, _ = state()
    act = actual()
    Wm = W.mean(0)
    games, played = {}, {}
    for i, g in enumerate(br.games):
        if g.id not in act:
            continue
        occ = [v if kind == "team" else played[v] for kind, v in g.sources]
        w = br.tidx[act[g.id]]
        l = occ[1] if occ[0] == w else occ[0]
        played[g.id] = w
        sides = summary["games"][i]["sides"]
        top = [max(sd, key=sd.get) for sd in sides]
        games[g.id] = {"round": g.round, "teams": [br.teams[t] for t in occ], "winner": br.teams[w],
                       "loser": br.teams[l], "p_winner": float(Wm[w, l]),
                       "wrong": bool(Wm[w, l] < 0.5),
                       "p_slot": [float(sd.get(str(t), 0.0)) for sd, t in zip(sides, occ)],
                       "model_top": [br.teams[int(t)] for t in top]}
    rounds = []
    for r in range(7):
        gs = [v for v in games.values() if v["round"] == r]
        if not gs:
            continue
        slots = [(t, p, top) for v in gs for t, p, top in zip(v["teams"], v["p_slot"], v["model_top"])]
        rounds.append({"round": r, "games": len(gs), "model_wrong": sum(v["wrong"] for v in gs),
                       "slots": len(slots), "top_pick_there": sum(t == top for t, _, top in slots),
                       "mean_p_slot": float(np.mean([p for _, p, _ in slots]))})
    return {"games": games, "rounds": rounds}


@app.route("/api/compare")
def api_compare():
    return jsonify(compare())


@app.route("/api/score", methods=["POST"])
def api_score():
    br, sims, summary, W, _ = state()
    if W is None:
        abort(503, "win matrix missing; run `python -m ncaa simulate`")
    picks = (request.get_json(silent=True) or {}).get("picks", {})
    out = score_bracket(br, picks, W, summary, sims)
    out["expected_points"] = expected_points(br, picks, summary)
    # Predicted (over simulated tournaments) vs realised ESPN score of these
    # picks; only meaningful for picks made before the tournament.
    out["espn"] = espn_points(br, picks, sims, actual()) if picks else None
    return jsonify(out)


@app.route("/api/favorites")
def api_favorites():
    """Fill the bracket game by game, always advancing the head-to-head favourite."""
    br, _, _, W, _ = state()
    return jsonify({g: br.teams[t] for g, t in favourites(br, W.mean(0)).items()})


FINAL_FOUR_GAMES = ("R4", "R5", "R6")  # region finals onward decide the Final Four and beyond


@lru_cache(maxsize=256)
def optimize(objective: str, champion: str, n_upsets: int | None, gap: int, keep_final_four: bool) -> dict:
    """Build the optimal bracket under the requested objective and constraints."""
    br, sims, summary, W, _ = state()
    forced = {}
    if keep_final_four:
        # Hold the region finals onward at the unconstrained optimum, so any
        # upsets have to come from games that do not change the Final Four.
        base = optimize(objective, champion, None, gap, False)
        forced = {g: br.tidx[t] for g, t in base["picks"].items() if g.startswith(FINAL_FOUR_GAMES)}
    if champion:
        forced["R6"] = br.tidx[champion]
    c = Constraints(forced=forced, n_upsets=n_upsets, upset_gap=gap)
    by_champion = []
    if objective == "points":
        picks = max_expected_points_bracket(br, summary, c)["picks"]
    else:
        res = most_likely_bracket(br, W, c, n_champions=1 if champion else 10)
        picks, by_champion = res["picks"], res["by_champion"]
    named = {g: br.teams[t] for g, t in picks.items()}
    played = resolve_picks(br, named)
    best_ll = max([r["log_likelihood"] for r in by_champion] or [0.0])
    return {
        "picks": named,
        "champion": named["R6"],
        "log_likelihood": bracket_log_likelihood(br, picks, W),
        "expected_points": expected_points(br, named, summary),
        "espn": espn_points(br, named, sims, actual()),
        "upsets": [{"id": p["id"], "winner": br.teams[p["winner"]], "loser": br.teams[p["loser"]],
                    "seeds": [br.seed[br.teams[p["winner"]]], br.seed[br.teams[p["loser"]]]]}
                   for p in played if is_upset(br, p["winner"], p["loser"], gap)],
        "by_champion": [{"champion": br.teams[r["champion"]], "log_likelihood": r["log_likelihood"],
                         "relative": float(np.exp(r["log_likelihood"] - best_ll)),
                         "p_title": summary["reach"][r["champion"]][-1]} for r in by_champion],
    }


@app.route("/api/optimize")
def api_optimize():
    """Optimal bracket: ?objective=likelihood|points&champion=&upsets=&gap=&keep_final_four=0|1"""
    br = state()[0]
    objective = request.args.get("objective", "likelihood")
    champion = request.args.get("champion", "")
    upsets = request.args.get("upsets", "")
    gap = int(request.args.get("gap", 1))
    if objective not in ("likelihood", "points") or (champion and champion not in br.tidx):
        abort(400, "bad objective or champion")
    try:
        n_upsets = int(upsets) if upsets else None
        keep = n_upsets is not None and request.args.get("keep_final_four") == "1"
        return jsonify(optimize(objective, champion, n_upsets, gap, keep))
    except ValueError as e:
        abort(422, str(e))


# How the public might pick, a priori: weight on team quality vs seed line,
# and chalkiness (>1 = favourites advanced more often than they really win).
PUBLICS = {
    "mixed": ("A mix of seed-driven and informed entrants, somewhat chalky", PublicModel((0.3, 0.8), (1.0, 2.0))),
    "chalky": ("Mostly follows seeds, very chalky", PublicModel((0.0, 0.3), (1.5, 2.5))),
    "informed": ("Picks close to the model's view", PublicModel((0.8, 1.0), (0.9, 1.3))),
    "random": ("Picks upsets more freely than the model expects", PublicModel((0.3, 0.8), (0.4, 0.8))),
}


@lru_cache(maxsize=len(PUBLICS))
def pool_for(public: str) -> Pool:
    br, sims, summary, W, _ = state()
    return Pool(br, sims, W.mean(0), summary, PUBLICS[public][1])


@lru_cache(maxsize=64)
def pool_result(n: int, public: str) -> dict:
    br, _, summary, W, _ = state()
    pool = pool_for(public)
    res = best_pool_bracket(pool, n, summary)
    named = {g: br.teams[t] for g, t in res["picks"].items()}
    mp = {g: br.teams[t] for g, t in res["max_points_picks"].items()}
    title = np.array(summary["reach"])[:, -1]
    q_title = pool.q[br.gidx["R6"]]
    ff = [g.id for g in br.games if g.round == 4]
    return {
        "n": n, "public": public, "public_label": PUBLICS[public][0],
        "picks": named, "champion": named["R6"], "lambda": res["lambda"],
        "p_win": res["p_win"], "p_win_max_points": res["p_win_max_points"], "baseline": res["baseline"],
        "final_four": [named[g] for g in ff], "max_points_final_four": [mp[g] for g in ff],
        "changed": [{"id": g, "round": br.games[br.gidx[g]].round, "from": mp[g], "to": named[g]}
                    for g in named if named[g] != mp[g]],
        "log_likelihood": bracket_log_likelihood(br, res["picks"], W),
        "espn": espn_points(br, named, state()[1], actual()),
        "champions": [{"team": br.teams[t], "seed": br.seed[br.teams[t]], "p_model": float(title[t]),
                       "p_public": float(q_title[t]), "ratio": float(title[t] / max(q_title[t], 1e-4))}
                      for t in np.argsort(-title)[:12]],
    }


@app.route("/api/pool")
def api_pool():
    """Bracket maximising P(finishing first) against n rivals: ?n=100&public=mixed"""
    n = int(request.args.get("n", 100))
    public = request.args.get("public", "mixed")
    if public not in PUBLICS or not 1 <= n <= 1_000_000:
        abort(400, "bad pool size or public model")
    return jsonify(pool_result(n, public))


@app.route("/api/publics")
def api_publics():
    return jsonify({k: v[0] for k, v in PUBLICS.items()})


@app.route("/api/simulate_one")
def api_simulate_one():
    """One random tournament, played game by game from the model (?seed= to replay one)."""
    br, _, _, W, post = state()
    seed = request.args.get("seed")
    seed = int(seed) if seed not in (None, "") else int(np.random.default_rng().integers(2**31))
    sim = simulate_tournament(br, post, 1, seed=seed, full_box=True)
    games = {}
    for i, g in enumerate(br.games):
        a, b = (int(x) for x in sim["sides"][0, i])
        games[g.id] = {"teams": [br.teams[a], br.teams[b]], "winner": br.teams[int(sim["winners"][0, i])],
                       "score": sim["scores"][0, i].tolist(), "ot": int(sim["ot"][0, i]),
                       "box": {s: v[0, i].tolist() for s, v in sim["box"].items()},
                       "p_winner": float(W.mean(0)[int(sim["winners"][0, i]), b if sim["winners"][0, i] == a else a])}
    picks = {gid: v["winner"] for gid, v in games.items()}
    played = resolve_picks(br, picks)
    return jsonify({
        "seed": seed, "posterior_draw": int(sim["draws"][0]), "picks": picks, "games": games,
        "champion": picks["R6"],
        "upsets": [{"id": p["id"], "winner": br.teams[p["winner"]], "loser": br.teams[p["loser"]],
                    "seeds": [br.seed[br.teams[p["winner"]]], br.seed[br.teams[p["loser"]]]]}
                   for p in played if is_upset(br, p["winner"], p["loser"])],
    })


@app.route("/api/upsets")
def api_upsets():
    """How many upsets the simulated tournaments contain, for each seed-gap definition."""
    br, sims, *_ = state()
    return jsonify({gap: upset_distribution(br, sims, gap) for gap in (1, 3, 5)})


@app.route("/api/teams")
def api_teams():
    return jsonify(state()[2]["ratings"])


@lru_cache(maxsize=1)
def real_games():
    """Every real game with box scores: the tournament (out of sample) and the season (used in fitting)."""
    import pandas as pd

    t = data.load_tournament_games(SEASON).assign(kind="tournament")
    r = data.load_games(SEASON).assign(kind="regular season")
    return pd.concat([t, r], ignore_index=True)


def _meeting_rows(a: str, b: str):
    g = real_games()
    return g[(g.team == a) & (g.opponent == b)].sort_values("date", ascending=False)


@lru_cache(maxsize=1)
def tournament_report() -> dict:
    """Every tournament game's whole-box-score unusualness, plus how they spread."""
    games = tournament_unusualness(state()[4], data.load_tournament_games(SEASON))
    p = np.array([g["p_value"] for g in games])
    return {"games": sorted(games, key=lambda g: g["p_value"]),
            "hist": np.histogram(p, bins=10, range=(0, 1))[0].tolist(),
            "share_below_05": float((p < 0.05).mean()), "share_below_10": float((p < 0.10).mean()),
            "median": float(np.median(p))}


@app.route("/api/tournament_unusualness")
def api_tournament_unusualness():
    return jsonify(tournament_report())


@app.route("/api/meetings")
def api_meetings():
    """Real games between two teams, tournament games first."""
    rows = _meeting_rows(request.args.get("a"), request.args.get("b"))
    rows = rows.sort_values(["kind", "date"], ascending=[False, False])
    return jsonify([{"game_id": int(r.game_id), "date": r.date.strftime("%b %d, %Y"), "kind": r.kind,
                     "home": int(r.home), "score": [int(r.pts), int(r.opp_pts)], "ot": int(r.ot)}
                    for r in rows.itertuples()])


def compare_to_game(res: dict, row) -> dict:
    """Where a real game's box score falls in the simulated distributions.

    ``percentile`` is the mid-p value P(sim < x) + P(sim = x)/2, so a
    well-calibrated model puts real values uniformly between 0 and 1.
    """
    stats = {}
    for s in BOX_STATS:
        if s == "poss":
            real = [row.fga - row.oreb + row.tov + 0.44 * row.fta, row.opp_fga - row.opp_oreb + row.opp_tov + 0.44 * row.opp_fta]
            real = [round(float(np.mean(real)), 1)] * 2
        else:
            real = [int(getattr(row, s)), int(getattr(row, "opp_" + s))]
        sim = res[s]
        pct = [float((sim[:, j] < real[j]).mean() + 0.5 * (sim[:, j] == real[j]).mean()) for j in range(2)]
        stats[s] = {"actual": real, "percentile": pct,
                    "p5": np.percentile(sim, 5, axis=0).round(1).tolist(),
                    "p95": np.percentile(sim, 95, axis=0).round(1).tolist()}
    margin = res["pts"][:, 0] - res["pts"][:, 1]
    m = int(row.pts - row.opp_pts)
    pcts = [v for st in stats.values() for v in st["percentile"]]
    return {"game_id": int(row.game_id), "date": row.date.strftime("%b %d, %Y"), "kind": row.kind, "ot": int(row.ot),
            "stats": stats, "margin": m,
            "margin_percentile": float((margin < m).mean() + 0.5 * (margin == m).mean()),
            "p_actual_winner": float(res["a_wins"].mean() if m > 0 else 1 - res["a_wins"].mean()),
            "outside_90": sum(1 for p in pcts if p < 0.05 or p > 0.95), "n_stats": len(pcts)}


@app.route("/api/matchup")
def api_matchup():
    """Simulate full box scores for a head-to-head game."""
    _, _, _, _, post = state()
    a, b = request.args.get("a"), request.args.get("b")
    n = min(int(request.args.get("n", 5000)), 50000)
    home = int(request.args.get("home", 0))
    if a not in post.teams or b not in post.teams or a == b:
        abort(400, "pick two different teams")
    rng = np.random.default_rng()
    k = rng.integers(post.n_draws, size=n)
    ia, ib = post.team_index([a, b])
    res = simulate_games(post, k, np.full(n, ia), np.full(n, ib), rng, home=home)
    pts = res["pts"]
    margin = pts[:, 0] - pts[:, 1]
    lo, hi = int(np.percentile(margin, 0.5)), int(np.percentile(margin, 99.5))
    edges = np.arange(lo, hi + 3, 2)
    box = {s: {"mean": res[s].mean(0).round(1).tolist(),
               "p10": np.percentile(res[s], 10, axis=0).round(1).tolist(),
               "p90": np.percentile(res[s], 90, axis=0).round(1).tolist()} for s in BOX_STATS}
    samples = [{s: res[s][i].tolist() for s in BOX_STATS} | {"ot": int(res["ot"][i])} for i in range(8)]
    actual_game = None
    if request.args.get("game_id"):
        rows = _meeting_rows(a, b)
        rows = rows[rows.game_id == int(request.args["game_id"])]
        if len(rows):
            row = next(rows.itertuples())
            actual_game = compare_to_game(res, row)
            u = game_unusualness(post, ia, ib, home, row, res=res)
            actual_game["joint"] = {k: u[k] for k in ("distance", "p_value", "dims", "hist", "edges", "drivers")}
    return jsonify({
        "teams": [a, b], "n": n, "p_a_wins": float(res["a_wins"].mean()),
        "p_ot": float((res["ot"] > 0).mean()),
        "margin": {"mean": float(margin.mean()), "sd": float(margin.std()),
                   "hist": np.histogram(margin, bins=edges)[0].tolist(), "edges": edges.tolist()},
        "total": {"mean": float(pts.sum(1).mean()), "sd": float(pts.sum(1).std())},
        "box": box, "samples": samples, "actual": actual_game,
        # Density of every box-score stat for both teams over the simulated games.
        "density": {s: (lambda g: {"grid": g.round(2).tolist(),
                                   "y": [kde(res[s][:, j], g).round(5).tolist() for j in range(2)],
                                   "mean": res[s].mean(0).round(2).tolist()})(
            np.linspace(np.percentile(res[s], 0.5) - 2, np.percentile(res[s], 99.5) + 2, 90))
            for s in BOX_STATS},
    })


def kde(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Gaussian kernel density (Silverman bandwidth, at least 0.6 so integer counts read as a smooth curve)."""
    x = samples.astype(float)
    n = len(x)
    sd = x.std()
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    bw = max(0.9 * min(sd, iqr / 1.34 if iqr > 0 else sd) * n ** -0.2, 0.6)
    dens = np.zeros_like(grid)
    for chunk in np.array_split(x, max(1, n // 2000)):
        dens += np.exp(-0.5 * ((grid[:, None] - chunk[None, :]) / bw) ** 2).sum(1)
    return dens / (n * bw * np.sqrt(2 * np.pi))


@app.route("/api/h2h")
def api_h2h():
    """Head-to-head win probabilities between tournament teams (row beats column)."""
    br, _, _, W, _ = state()
    return jsonify({"teams": br.teams, "p": np.round(W.mean(0), 4).tolist()})


@app.route("/api/team_list")
def api_team_list():
    return jsonify(sorted(state()[4].teams))


# ---------------------------------------------------------------------------
# Season tracking and live games
# ---------------------------------------------------------------------------
import threading
import time as _time

import pandas as pd

from ncaa import season as season_mod
from ncaa.ingame import forecast as live_forecast, live_unusualness, pregame
from ncaa.live import ESPNFeed, ReplayFeed

LIVE_SEASON = int(os.environ.get("NCAA_LIVE_SEASON", 2027))
_espn, _replay = ESPNFeed(), ReplayFeed()
_history: dict = {}          # game id -> [(elapsed, our P(home), ESPN P(home))] seen while polling live
_lock = threading.Lock()


@lru_cache(maxsize=4)
def _season_post(season: int, stamp: float):
    return season_mod.current_posterior(season)


def season_post(season: int):
    """Latest posterior for a season (re-read when the nightly update rewrites it)."""
    path = model.artifact_dir(season) / "posterior.npz"
    return _season_post(season, path.stat().st_mtime if path.exists() else 0.0)


@app.route("/season")
def season_page():
    return render_template("season.html", season=SEASON, live_season=LIVE_SEASON, page="season")


@app.route("/live")
def live_page():
    return render_template("live.html", season=SEASON, live_season=LIVE_SEASON, page="live")


@app.route("/api/season/status")
def api_season_status():
    """Where the season stands: last update, and how the out-of-sample ledger is scoring."""
    season = int(request.args.get("season", LIVE_SEASON))
    team = request.args.get("team", "")
    post, label = season_post(season)
    path = model.artifact_dir(season) / "status.json"
    status = json.loads(path.read_text()) if path.exists() else {}
    led = season_mod.load_ledger(season)
    summ = {"games": len(led)}
    if len(led):
        won = (led.home_pts > led.away_pts).to_numpy()
        p = led.p_home.clip(1e-3, 1 - 1e-3).to_numpy()
        u = led.unusualness.to_numpy()
        summ |= {"log_loss": float(-np.mean(np.where(won, np.log(p), np.log(1 - p)))),
                 "brier": float(np.mean((p - won) ** 2)), "accuracy": float(np.mean((p > 0.5) == won)),
                 "unusual_hist": np.histogram(u, bins=10, range=(0, 1))[0].tolist(),
                 "share_unusual_05": float(np.mean(u < 0.05))}
    rows = led[(led.home == team) | (led.away == team)] if team else led
    rows = rows.sort_values("date", ascending=False).head(300)
    return jsonify({"season": season, "posterior": label, "status": status, "summary": summ,
                    "teams": sorted(post.teams) if post is not None else [],
                    "games": json.loads(rows.to_json(orient="records", date_format="iso"))})


@app.route("/api/season/upcoming")
def api_season_upcoming():
    season = int(request.args.get("season", LIVE_SEASON))
    post, label = season_post(season)
    if post is None:
        abort(404, "no posterior for this season yet")
    date = request.args.get("date") or None
    return jsonify({"posterior": label, "games": season_mod.upcoming(season, post, start=date,
                                                                     days=int(request.args.get("days", 1)))})


@app.route("/api/live/scoreboard")
def api_live_scoreboard():
    date = request.args.get("date") or pd.Timestamp.now(tz="America/New_York").strftime("%Y%m%d")
    try:
        return jsonify({"date": date, "games": _espn.scoreboard(date.replace("-", ""))})
    except Exception as e:  # network or feed trouble: say so instead of failing the page
        return jsonify({"date": date, "games": [], "error": f"ESPN scoreboard unavailable: {e}"})


@app.route("/api/live/replay_games")
def api_replay_games():
    season = int(request.args.get("season", SEASON))
    t = data.load_tournament_games(season)
    if not len(t):
        return jsonify([])
    t = t[t.team < t.opponent].sort_values("date", ascending=False)
    return jsonify([{"game_id": int(r.game_id), "date": r.date.strftime("%b %d"), "teams": [r.team, r.opponent],
                     "score": [int(r.pts), int(r.opp_pts)]} for r in t.itertuples()])


def _model_view(post, state, n=3000):
    """Forecast, pre-game baseline and live unusualness for a state (home team first)."""
    idx = {t: i for i, t in enumerate(post.teams)}
    if not all(t in idx for t in state.teams):
        return None
    a, b = idx[state.teams[0]], idx[state.teams[1]]
    home = 0 if state.neutral else 1
    f = live_forecast(post, a, b, home, state, n=n, seed=0)
    pre = pregame(post, a, b, home, n=n, seed=0)
    u = live_unusualness(post, a, b, home, state, seed=0) if state.status != "pre" else None
    return {"p_win": f["p_win"], "final": f["final"], "margin": f["margin"], "p_overtime": f["p_overtime"],
            "possessions_left": f["possessions_left"], "effective_draws": f["effective_draws"],
            "pregame": pre, "unusualness": u}


@lru_cache(maxsize=64)
def _replay_timeline(season: int, game_id: int) -> list:
    post, _ = season_post(season)
    length = _replay.length(game_id)
    out = []
    for m in np.arange(0, length + 0.01, 1.0):
        st = _replay.game(game_id, float(m))
        v = _model_view_light(post, st)
        out.append({"elapsed": float(st.elapsed), "ours": v, "espn": st.espn_home_wp,
                    "score": [st.box[t]["pts"] for t in st.teams]})
    return out


def _model_view_light(post, state):
    idx = {t: i for i, t in enumerate(post.teams)}
    if not all(t in idx for t in state.teams):
        return None
    return live_forecast(post, idx[state.teams[0]], idx[state.teams[1]], 0 if state.neutral else 1, state,
                         n=800, seed=1)["p_win"][0]


@app.route("/api/live/game")
def api_live_game():
    """A game's live state with the model's Bayesian in-game forecast.

    ?source=espn&id=<game id>  — live from ESPN (history builds up as it is polled)
    ?source=replay&id=<game id>&minute=<m>&season=2026 — a finished game replayed at minute m
    """
    source = request.args.get("source", "espn")
    gid = int(request.args["id"])
    try:
        if source == "replay":
            season = int(request.args.get("season", SEASON))
            state = _replay.game(gid, float(request.args.get("minute", 0)))
            timeline = [t for t in _replay_timeline(season, gid) if t["elapsed"] <= state.elapsed + 1e-6]
        else:
            season = int(request.args.get("season", LIVE_SEASON))
            state = _espn.game(gid)
            timeline = None
    except Exception as e:
        return jsonify({"error": f"could not load game {gid}: {e}"}), 502
    post, label = season_post(season)
    view = _model_view(post, state) if post is not None else None
    if timeline is None:
        with _lock:
            hist = _history.setdefault(gid, [])
            if view and (not hist or hist[-1]["elapsed"] < state.elapsed - 1e-6):
                hist.append({"elapsed": state.elapsed, "ours": view["p_win"][0], "espn": state.espn_home_wp,
                             "score": [state.box.get(t, {}).get("pts", 0) for t in state.teams]})
            timeline = list(hist)
    return jsonify({"state": state.to_json(), "model": view, "posterior": label, "timeline": timeline})


@lru_cache(maxsize=32)
def _bracket_asof(asof: str) -> dict:
    """The distribution bracket conditioned on every result final before ``asof``."""
    br, _, _, _, post = state()
    cut = pd.Timestamp(asof)
    played = {}
    for r in br.spec.get("results", []):
        if "date" in r and pd.Timestamp(r["date"]) + pd.Timedelta(hours=2.5) <= cut:
            played[frozenset(r["teams"])] = r["winner"]
    forced = {}
    for _ in range(7):  # resolve game ids round by round
        for g in br.games:
            if g.id in forced:
                continue
            teams = [br.teams[v] if kind == "team" else (br.teams[forced[v]] if v in forced else None)
                     for kind, v in g.sources]
            if None not in teams and frozenset(teams) in played:
                forced[g.id] = br.tidx[played[frozenset(teams)]]
    sims = simulate_tournament(br, post, 4000, seed=11, forced=forced)
    from ncaa.bracket import summarize
    out = summarize(br, sims)
    out["bracket"] = state()[2]["bracket"]
    out["ratings"] = state()[2]["ratings"]
    out["asof"], out["games_final"] = cut.isoformat(), len(forced)
    return out


@app.route("/api/bracket_asof")
def api_bracket_asof():
    """?asof=<ISO time>: the bracket as it stood then, with every finished game locked in."""
    return jsonify(_bracket_asof(request.args.get("asof", pd.Timestamp.now(tz="UTC").isoformat())))


@app.route("/api/tournament_days")
def api_tournament_days():
    br = state()[0]
    days = sorted({pd.Timestamp(r["date"]).tz_convert("America/New_York").strftime("%Y-%m-%d")
                   for r in br.spec.get("results", []) if "date" in r})
    return jsonify(days)


if __name__ == "__main__":
    app.run(debug=True)
