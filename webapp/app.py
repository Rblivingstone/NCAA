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
from ncaa.evaluate import game_unusualness, kde, tournament_unusualness
from ncaa.pool import ARCHETYPES, Pool, PoolSpec, best_pool_bracket
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


@app.route("/healthz")
def healthz():
    """Liveness plus whether the models are built (the worker builds them on first start)."""
    ready = (model.artifact_dir(SEASON) / "summary.json").exists()
    return jsonify({"ok": True, "ready": ready, "season": SEASON, "live_season": LIVE_SEASON}), (200 if ready else 503)


@app.before_request
def _building():
    """Until the first build finishes, every page says so instead of erroring."""
    if request.path.startswith(("/static", "/healthz")) or (model.artifact_dir(SEASON) / "summary.json").exists():
        return None
    return ("<!doctype html><meta http-equiv='refresh' content='60'><body style='font-family:system-ui;padding:40px'>"
            "<h2>Building the models…</h2><p>The first start downloads the season's data, fits the model and simulates "
            "the tournament. This takes 20–90 minutes depending on the server; the page refreshes itself.</p>", 503)


@app.route("/")
def dashboard_page():
    return render_template("dashboard.html", season=SEASON, live_season=LIVE_SEASON, page="dashboard")


@app.route("/bracket")
def index():
    return render_template("index.html", season=SEASON, live_season=LIVE_SEASON, page="bracket")


@app.route("/score")
def score_page():
    return render_template("score.html", season=SEASON, live_season=LIVE_SEASON, page="score")


@app.route("/matchup")
def matchup_page():
    return render_template("matchup.html", season=SEASON, live_season=LIVE_SEASON, page="matchup")


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


# Pool presets: archetype mixes (shares of casual / informed / chalk / upset entrants) and pick correlation.
POOL_PRESETS = {
    "office": ("Office pool: mostly casual fans", {"casual": 0.6, "informed": 0.1, "chalk": 0.2, "upset": 0.1}, 0.3),
    "mixed": ("A mix of seed-driven and informed entrants", {"casual": 0.45, "informed": 0.2, "chalk": 0.25, "upset": 0.1}, 0.0),
    "chalky": ("Mostly favourites", {"casual": 0.3, "informed": 0.1, "chalk": 0.6, "upset": 0.0}, 0.2),
    "sharp": ("Analytics crowd: close to the model", {"casual": 0.1, "informed": 0.7, "chalk": 0.15, "upset": 0.05}, 0.4),
    "chaos": ("Upset hunters", {"casual": 0.3, "informed": 0.1, "chalk": 0.1, "upset": 0.5}, 0.0),
}


@lru_cache(maxsize=8)
def pool_for(mix: tuple, rho: float) -> Pool:
    br, sims, summary, W, _ = state()
    return Pool(br, sims, W.mean(0), summary, PoolSpec(mix=mix, rho=rho))


def _pool_spec_from(req: dict) -> tuple[int, tuple, float]:
    n = int(req.get("n", 100))
    mix = req.get("mix") or POOL_PRESETS["mixed"][1]
    mix = tuple(sorted((k, round(float(v), 3)) for k, v in mix.items() if k in ARCHETYPES and float(v) > 0))
    rho = round(min(max(float(req.get("rho", 0.0)), 0.0), 0.95), 2)
    if not mix or not 1 <= n <= 1_000_000:
        abort(400, "bad pool size or mix")
    return n, mix, rho


@lru_cache(maxsize=64)
def pool_result(n: int, mix: tuple, rho: float) -> dict:
    br, sims, summary, W, _ = state()
    pool = pool_for(mix, rho)
    res = best_pool_bracket(pool, n, summary)
    named = {g: br.teams[t] for g, t in res["picks"].items()}
    mp = {g: br.teams[t] for g, t in res["max_points_picks"].items()}
    title = np.array(summary["reach"])[:, -1]
    q_title = pool.q[br.gidx["R6"]]
    ff = [g.id for g in br.games if g.round == 4]
    return {
        "n": n, "mix": dict(mix), "rho": rho,
        "picks": named, "champion": named["R6"], "lambda": res["lambda"],
        "p_win": res["p_win"], "p_win_max_points": res["p_win_max_points"], "baseline": res["baseline"],
        "finish": pool.finish(res["picks"], n), "finish_max_points": pool.finish(res["max_points_picks"], n),
        "winning_score": pool.winning_score(n),
        "final_four": [named[g] for g in ff], "max_points_final_four": [mp[g] for g in ff],
        "changed": [{"id": g, "round": br.games[br.gidx[g]].round, "from": mp[g], "to": named[g]}
                    for g in named if named[g] != mp[g]],
        "log_likelihood": bracket_log_likelihood(br, res["picks"], W),
        "espn": espn_points(br, named, sims, actual()),
        "champions": [{"team": br.teams[t], "seed": br.seed[br.teams[t]], "p_model": float(title[t]),
                       "p_public": float(q_title[t]), "ratio": float(title[t] / max(q_title[t], 1e-4))}
                      for t in np.argsort(-title)[:12]],
    }


@app.route("/api/pool", methods=["GET", "POST"])
def api_pool():
    """Bracket maximising P(finishing first), with pool-difficulty analysis.

    POST {"n": 100, "mix": {"casual": .45, "informed": .2, "chalk": .25, "upset": .1}, "rho": 0.3,
          "picks": {game id: team}}  (picks optional: also analyse your own bracket)
    """
    req = request.get_json(silent=True) or {k: request.args.get(k) for k in ("n", "rho") if request.args.get(k)}
    if request.args.get("preset") in POOL_PRESETS:
        _, mix_, rho_ = POOL_PRESETS[request.args["preset"]]
        req = {"n": req.get("n", 100), "mix": mix_, "rho": rho_}
    n, mix, rho = _pool_spec_from(req)
    out = dict(pool_result(n, mix, rho))
    picks = req.get("picks") if isinstance(req, dict) else None
    br = state()[0]
    if picks and len(resolve_picks(br, picks)) == len(br.games):
        idx = {g: br.tidx[t] for g, t in picks.items()}
        out["finish_yours"] = pool_for(mix, rho).finish(idx, n)
    return jsonify(out)


@app.route("/api/pool_presets")
def api_pool_presets():
    return jsonify({"archetypes": {k: v[0] for k, v in ARCHETYPES.items()},
                    "presets": {k: {"label": v[0], "mix": v[1], "rho": v[2]} for k, v in POOL_PRESETS.items()}})


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
    """Simulate full box scores for a head-to-head game (?season=, default the live season)."""
    season = int(request.args.get("season", LIVE_SEASON))
    post = state()[4] if season == SEASON else season_post(season)[0]
    if post is None:
        abort(404, "no model for this season yet")
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
        rows = _meeting_rows(a, b) if season == SEASON else pd.DataFrame(columns=["game_id"])
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


@app.route("/api/h2h")
def api_h2h():
    """Head-to-head win probabilities between tournament teams (row beats column)."""
    br, _, _, W, _ = state()
    return jsonify({"teams": br.teams, "p": np.round(W.mean(0), 4).tolist()})


@app.route("/api/team_list")
def api_team_list():
    season = int(request.args.get("season", LIVE_SEASON))
    if season == SEASON:
        return jsonify(sorted(state()[4].teams))
    d1 = data.d1_set(season)
    post = season_post(season)[0]
    teams = [t for t in (post.teams if post is not None else []) if t != "Non-D1" and (not d1 or t in d1)]
    return jsonify(sorted(teams))


# ---------------------------------------------------------------------------
# Season tracking and live games
# ---------------------------------------------------------------------------
import threading
import time as _time

import pandas as pd

from ncaa import season as season_mod
from ncaa.ingame import SHIFT_LEVEL, forecast as live_forecast, live_unusualness, pregame, shifts, stat_densities
from ncaa.live import FallbackFeed, ReplayFeed, parse_summary

LIVE_SEASON = int(os.environ.get("NCAA_LIVE_SEASON", 2027))
_espn, _replay = FallbackFeed(), ReplayFeed()   # ESPN, or the game archive once a game is final
_lock = threading.Lock()
_forecasters: dict = {}
_track_lock = threading.Lock()


def _tracked(season: int, gid: int):
    """A game as the worker's tracker has recorded it: (track, live state), or (None, None) if never seen.

    The worker polls every game from its scheduled tip-off; if its record is stale (worker down or behind),
    the game is polled here instead, which also reconstructs any missing history from the play-by-play."""
    from ncaa import tracker

    track = tracker.load_track(season, gid)
    if tracker.is_stale(track):
        try:
            with _track_lock:
                fc = _forecasters.setdefault(season, tracker.Forecaster(season))
                track = tracker.track_game(season, gid, _espn, fc, log=lambda *_: None)
        except Exception as e:
            app.logger.warning("could not refresh game %s: %r", gid, e)
            if track is None:
                raise
    raw = tracker.load_summary(season, gid)
    if raw is None:
        return track, None
    return track, parse_summary(raw, until=999) if track.get("status") == "post" else parse_summary(raw)


@lru_cache(maxsize=4)
def _season_post(season: int, stamp: float):
    return season_mod.current_posterior(season)


def _stamp(season: int) -> float:
    """Changes whenever the season's model does: a refit, or a game folded into the ratings."""
    d = model.artifact_dir(season)
    return max([p.stat().st_mtime for p in (d / "posterior.npz", d / "team_state.npz") if p.exists()] or [0.0])


def season_post(season: int):
    """The season's model as of now (re-read after a refit or a game update)."""
    return _season_post(season, _stamp(season))


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
                 "share_unusual_05": float(np.mean(u < 0.05)),
                 "vs_espn": _vs_espn(led)}
    rows = led[(led.home == team) | (led.away == team)] if team else led
    rows = rows.sort_values("date", ascending=False).head(3000)
    games = json.loads(rows.to_json(orient="records", date_format="iso"))
    ranks = rank_map(season)
    for g in games:
        g |= game_flags(g["home"], g["away"], g.get("p_home"), ranks, result=(g["home_pts"], g["away_pts"]))
    return jsonify({"season": season, "posterior": label, "status": status, "summary": summ,
                    "teams": sorted(t for t in post.teams if t != "Non-D1") if post is not None else [],
                    "games": games})


def _vs_espn(led: pd.DataFrame) -> dict | None:
    """This model vs ESPN, on the games both forecast before tip-off: log-loss, Brier, and the running
    (cumulative) log-loss of each through the season."""
    if "espn_p_home" not in led:
        return None
    both = led[led.espn_p_home.notna()].sort_values("date")
    if not len(both):
        return {"games": 0, "checked": int(led.espn_checked.sum()) if "espn_checked" in led else 0}
    won = (both.home_pts > both.away_pts).to_numpy()
    out = {"games": int(len(both)), "checked": int(led.espn_checked.sum())}
    losses = {}
    for name, col in (("ours", "p_home"), ("espn", "espn_p_home")):
        p = both[col].astype(float).clip(1e-3, 1 - 1e-3).to_numpy()
        ll = -np.where(won, np.log(p), np.log(1 - p))
        losses[name] = ll
        out[name] = {"log_loss": float(ll.mean()), "brier": float(np.mean((p - won) ** 2)),
                     "accuracy": float(np.mean((p > 0.5) == won))}
    d = losses["ours"] - losses["espn"]          # per game: > 0 when ESPN did better
    out["diff"] = float(d.mean())
    out["diff_se"] = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None
    days = both.date.dt.strftime("%Y-%m-%d").to_numpy()
    n = np.arange(1, len(both) + 1)
    cum = {k: np.cumsum(v) / n for k, v in losses.items()}
    last = np.r_[days[1:] != days[:-1], True]   # one point per day: the running mean after that day's games
    out["running"] = [{"date": days[i], "games": int(n[i]), "ours": float(cum["ours"][i]), "espn": float(cum["espn"][i])}
                      for i in np.flatnonzero(last)]
    return out


@app.route("/api/season/upcoming")
def api_season_upcoming():
    season = int(request.args.get("season", LIVE_SEASON))
    post, label = season_post(season)
    if post is None:
        abort(404, "no posterior for this season yet")
    date = request.args.get("date") or None
    ranks = rank_map(season)
    games = season_mod.upcoming(season, post, start=date, days=int(request.args.get("days", 1)))
    for g in games:
        g |= game_flags(g["home"], g["away"], g.get("p_home"), ranks)
    return jsonify({"posterior": label, "games": games})


@app.route("/api/live/scoreboard")
def api_live_scoreboard():
    date = request.args.get("date") or pd.Timestamp.now(tz="America/New_York").strftime("%Y%m%d")
    try:
        return jsonify({"date": date, "games": _espn.scoreboard(date.replace("-", ""))})
    except Exception as e:  # ESPN unreachable: list the day from the schedule and what the tracker has seen
        return jsonify({"date": date, "games": _schedule_board(date.replace("-", "")),
                        "error": f"ESPN scoreboard unavailable ({e}); showing the schedule"})


def _schedule_board(date: str) -> list[dict]:
    from ncaa import tracker

    try:
        sched = data.load_schedule(LIVE_SEASON)
    except Exception:
        return []
    day = sched[sched.date.dt.tz_convert("America/New_York").dt.strftime("%Y%m%d") == date]
    out = []
    for r in day.sort_values("date").itertuples():
        tr = tracker.load_track(LIVE_SEASON, r.game_id) or {}
        pts = (tr.get("points") or [{}])[-1].get("score")
        status = tr.get("status") or ("post" if r.completed else "pre")
        score = pts or ([int(r.home_score), int(r.away_score)] if r.completed else [0, 0])
        out.append({"game_id": int(r.game_id), "teams": [r.home, r.away], "score": score, "neutral": bool(r.neutral),
                    "status": status, "detail": tr.get("detail") or ("Final" if r.completed else
                                                                      r.date.tz_convert("America/New_York").strftime("%I:%M %p ET").lstrip("0")),
                    "start": r.date.isoformat()})
    return out


@app.route("/api/live/replay_games")
def api_replay_games():
    season = int(request.args.get("season", SEASON))
    t = data.load_tournament_games(season)
    if not len(t):
        return jsonify([])
    t = t[t.team < t.opponent].sort_values("date", ascending=False)
    return jsonify([{"game_id": int(r.game_id), "date": r.date.strftime("%b %d"), "teams": [r.team, r.opponent],
                     "score": [int(r.pts), int(r.opp_pts)]} for r in t.itertuples()])


@lru_cache(maxsize=64)
def _pregame_cached(post_id: int, a: int, b: int, home: int, n: int = 3000):
    return pregame(_posts[post_id], a, b, home, n=n, seed=0, samples=True)


_posts: dict = {}


def _pre(post, a, b, home):
    _posts[id(post)] = post
    return _pregame_cached(id(post), a, b, home)


def _with_teams(post, state):
    """A team the model has no history for starts at the average of its group (D1, or non-D1 opponents)."""
    d1 = set()
    for season in {LIVE_SEASON, SEASON}:
        try:
            d1 |= data.d1_set(season)
        except Exception:
            pass
    return model.with_teams(post, state.teams, d1 or None)


def _model_view(post, state, n=3000):
    """Forecast, pre-game baseline and live unusualness for a state (home team first)."""
    post = _with_teams(post, state)
    idx = {t: i for i, t in enumerate(post.teams)}
    a, b = idx[state.teams[0]], idx[state.teams[1]]
    home = 0 if state.neutral else 1
    f = live_forecast(post, a, b, home, state, n=n, seed=0)
    pre = _pre(post, a, b, home)
    u = live_unusualness(post, a, b, home, state, seed=0) if state.status != "pre" else None
    return {"p_win": f["p_win"], "final": f["final"], "margin": f["margin"], "p_overtime": f["p_overtime"],
            "possessions_left": f["possessions_left"], "effective_draws": f["effective_draws"],
            "pregame": {"p_win": pre["p_win"], "final": pre["final"]}, "unusualness": u,
            # End-of-game density of every stat now vs pre-game, and how far it has moved.
            "densities": stat_densities(f["samples"], pre["samples"]), "shift_level": SHIFT_LEVEL}


@lru_cache(maxsize=64)
def _replay_timeline(season: int, game_id: int) -> list:
    post, _ = season_post(season)
    length = _replay.length(game_id)
    out = []
    for m in np.arange(0, length + 0.01, 1.0):
        st = _replay.game(game_id, float(m))
        p, q = _model_view_light(post, st)
        out.append({"elapsed": float(st.elapsed), "ours": p, "espn": st.espn_home_wp,
                    "score": [st.box[t]["pts"] for t in st.teams], "q": q})
    return out


def _model_view_light(post, state):
    """Win probability and per-stat shift q at one moment (for timelines)."""
    post = _with_teams(post, state)
    idx = {t: i for i, t in enumerate(post.teams)}
    a, b, home = idx[state.teams[0]], idx[state.teams[1]], 0 if state.neutral else 1
    f = live_forecast(post, a, b, home, state, n=1500, seed=1)
    return f["p_win"][0], shifts(f["samples"], _pre(post, a, b, home)["samples"])


@app.route("/api/live/game")
def api_live_game():
    """A game's live state with the model's Bayesian in-game forecast.

    ?source=espn&id=<game id>  — live from ESPN, with the history the worker has tracked since tip-off
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
            track, state = _tracked(season, gid)
            if state is None:
                raise ValueError("no data from the live feed yet")
            timeline = track.get("points", [])
    except Exception as e:
        app.logger.exception("could not load game %s", gid)      # full traceback in the server log
        return jsonify({"error": f"could not load game {gid}: {type(e).__name__}: {e}"}), 502
    post, label = season_post(season)
    view = _model_view(post, state) if post is not None else None
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


# ---------------------------------------------------------------------------
# Rankings, team pages, games
# ---------------------------------------------------------------------------
from ncaa import teams as teams_mod
from ncaa.live import LiveState, STATS as LIVE_STATS
from ncaa.ratings import team_ratings

SEASONS = sorted({SEASON, LIVE_SEASON})


@lru_cache(maxsize=4)
def _ratings(season: int, stamp: float):
    post, label = season_post(season)
    br = state()[0] if season == SEASON else None
    return team_ratings(post, br), label


def ratings_for(season: int):
    return _ratings(season, _stamp(season))


@app.route("/rankings")
def rankings_page():
    return render_template("rankings.html", season=SEASON, live_season=LIVE_SEASON, seasons=SEASONS, page="rankings",
                           default_season=int(request.args.get("season", LIVE_SEASON)))


@app.route("/teams")
def teams_redirect():
    from flask import redirect
    return redirect("/rankings")


@app.route("/api/rankings")
def api_rankings():
    """Teams ranked by overall (net), offense (ORtg) or defense (DRtg)."""
    season = int(request.args.get("season", LIVE_SEASON))
    rows, label = ratings_for(season)
    d1 = set(data.d1_teams(season))
    rows = [r for r in rows if r["team"] != "Non-D1" and (not d1 or r["team"] in d1)]
    # How far each team has moved through this season's game-by-game updates.
    up = season_mod.load_updates(season)
    moved = {}
    if len(up):
        for t, g in up.groupby("team"):
            g = g.sort_values("date")
            moved[t] = {"games": len(g), "moved": float(g.net_after.iloc[-1] - g.net_before.iloc[0]),
                        "last": float(g.net_after.iloc[-1] - g.net_before.iloc[-1]), "last_date": g.date.iloc[-1]}
    rows = [dict(r, **moved.get(r["team"], {"games": 0, "moved": 0.0, "last": 0.0, "last_date": None})) for r in rows]
    rank = {}
    for key, rev in (("net", True), ("ortg", True), ("drtg", False)):
        for i, r in enumerate(sorted(rows, key=lambda r: r[key], reverse=rev), 1):
            rank.setdefault(r["team"], {})[key] = i
    out = [dict(r, rank_overall=rank[r["team"]]["net"], rank_offense=rank[r["team"]]["ortg"],
                rank_defense=rank[r["team"]]["drtg"]) for r in rows]
    st_path = model.artifact_dir(season) / "status.json"
    status = json.loads(st_path.read_text()) if st_path.exists() else {}
    return jsonify({"season": season, "posterior": label, "teams": out, "games_folded_in": len(up) // 2 if len(up) else 0,
                    "last_update": max(moved[t]["last_date"] for t in moved) if moved else status.get("updated")})


@app.route("/team/<path:name>")
def team_page(name):
    return render_template("team.html", season=SEASON, live_season=LIVE_SEASON, seasons=SEASONS, team=name, page="rankings")


def _end_of_data(season: int):
    g = data.load_games(season)
    return g.date.max() if len(g) else pd.Timestamp.now(tz="UTC")


@app.route("/api/team")
def api_team():
    season = int(request.args.get("season", LIVE_SEASON))
    name = request.args.get("name", "")
    post, label = season_post(season)
    if post is None or name not in post.teams:
        abort(404, "team not in the model for this season")
    rows, _ = ratings_for(season)
    rating = next(r for r in rows if r["team"] == name)
    profile = teams_mod.skill_profile(post, name)
    styles = teams_mod.style_vulnerabilities(post, name)
    tempo = teams_mod.tempo_note(post, name)
    hist = season_mod.load_ratings_history(season)
    hist = hist[hist.team == name].to_dict("records") if len(hist) else []
    up = season_mod.load_updates(season)
    per_game = up[up.team == name].sort_values("date").to_dict("records") if len(up) else []
    # Games: results (with the out-of-sample pre-game forecast when the ledger has it) and upcoming.
    sched = data.load_schedule(season)
    games = []
    if len(sched):
        led = season_mod.load_ledger(season).set_index("game_id") if len(season_mod.load_ledger(season)) else None
        mine = sched[(sched.home == name) | (sched.away == name)].sort_values("date")
        for r in mine.itertuples():
            row = {"game_id": int(r.game_id), "date": r.date.isoformat(), "home": r.home, "away": r.away,
                   "neutral": bool(r.neutral), "completed": bool(r.completed),
                   "score": [int(r.home_score), int(r.away_score)] if r.completed else None}
            if led is not None and r.game_id in led.index:
                lr = led.loc[r.game_id]
                p_home = float(lr.p_home) if lr.home == r.home else 1 - float(lr.p_home)
                row |= {"p_home_pregame": p_home, "unusualness": float(lr.unusualness)}
            games.append(row)
    return jsonify({"season": season, "posterior": label, "team": name, "rating": rating,
                    "assessment": teams_mod.assessment(profile, styles, rating, tempo),
                    "profile": profile, "styles": styles, "tempo": tempo,
                    "form": teams_mod.form_history(post, name, _end_of_data(season)),
                    "history": hist, "updates": per_game, "games": games})


@app.route("/games")
def games_page():
    return render_template("games.html", season=SEASON, live_season=LIVE_SEASON, seasons=SEASONS, page="games")


@app.route("/api/games")
def api_games():
    """Games for a date (or all of a team's games), with forecasts for the ones not yet played."""
    season = int(request.args.get("season", LIVE_SEASON))
    sched = data.load_schedule(season)
    if not len(sched):
        return jsonify({"games": [], "dates": []})
    team = request.args.get("team", "")
    if team:
        rows = sched[(sched.home == team) | (sched.away == team)]
    else:
        day = pd.Timestamp(request.args.get("date") or pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d"))
        local = sched.date.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
        rows = sched[local == day]
    rows = rows.sort_values("date")
    post, label = season_post(season)
    ranks = rank_map(season)
    led = season_mod.load_ledger(season)
    led = led.set_index("game_id") if len(led) else None
    upcoming_ids = {u["game_id"]: u for u in
                    (season_mod.upcoming(season, post, start=rows.date.min().strftime("%Y-%m-%d"),
                                         days=(rows.date.max() - rows.date.min()).days + 2)
                     if post is not None and len(rows) and (~rows.completed).any() and len(rows) <= 120 else [])}
    out = []
    for r in rows.itertuples():
        row = {"game_id": int(r.game_id), "date": r.date.isoformat(), "home": r.home, "away": r.away,
               "neutral": bool(r.neutral), "completed": bool(r.completed), "tournament": bool(r.tournament),
               "score": [int(r.home_score), int(r.away_score)] if r.completed else None}
        if r.game_id in upcoming_ids and "p_home" in upcoming_ids[r.game_id]:
            u = upcoming_ids[r.game_id]
            row |= {"p_home": u["p_home"], "pred_score": u["pred_score"]}
        if led is not None and r.game_id in led.index:
            lr = led.loc[r.game_id]
            row |= {"p_home_pregame": float(lr.p_home) if lr.home == r.home else 1 - float(lr.p_home),
                    "unusualness": float(lr.unusualness)}
        p = row.get("p_home_pregame", row.get("p_home"))
        row |= game_flags(r.home, r.away, p, ranks, result=tuple(row["score"]) if row["score"] else None)
        out.append(row)
    local_dates = sorted(sched.date.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d").unique())
    return jsonify({"season": season, "posterior": label, "games": out, "dates": local_dates})


@app.route("/game/<int:game_id>")
def game_page(game_id):
    return render_template("game.html", season=SEASON, live_season=LIVE_SEASON, game_id=game_id, page="games")


def _final_state_from_box(season: int, game_id: int, row) -> LiveState | None:
    g = real_games()
    rows = g[g.game_id == game_id]
    if not len(rows):
        return None
    box = {}
    for t, prefix in ((row.home, ""), (row.away, "")):
        r = rows[rows.team == t]
        if not len(r):
            return None
        r = r.iloc[0]
        box[t] = {k: int(r[k]) for k in LIVE_STATS}
    return LiveState(game_id, [row.home, row.away], ["", ""], bool(row.neutral), "post", 2, 0.0, box, detail="Final")


@app.route("/api/game")
def api_game():
    """One game, whatever its state: prior before tip-off, live during, update history after."""
    season = int(request.args.get("season", LIVE_SEASON))
    gid = int(request.args["id"])
    sched = data.load_schedule(season)
    row = sched[sched.game_id == gid]
    if not len(row):
        abort(404, "game not found in this season's schedule")
    row = next(row.itertuples())
    post, label = season_post(season)
    meta = {"season": season, "date": row.date.isoformat(), "home": row.home, "away": row.away,
            "neutral": bool(row.neutral), "tournament": bool(row.tournament)}
    led = season_mod.load_ledger(season)
    if len(led) and gid in set(led.game_id):
        lr = led[led.game_id == gid].iloc[0]
        meta["ledger"] = {"p_home": float(lr.p_home) if lr.home == row.home else 1 - float(lr.p_home),
                          "unusualness": float(lr.unusualness), "model": lr.model}
    timeline, kind, state_ = None, "pre", None
    now = pd.Timestamp.now(tz="UTC")
    from ncaa import tracker

    track = tracker.load_track(season, gid)
    if track is None and not row.completed and row.date - tracker.POLL_BEFORE <= now:
        try:
            track, _ = _tracked(season, gid)        # the worker hasn't got to it (yet): live feed or archive
        except Exception as e:
            meta["note"] = f"live feed unavailable ({e})"
    if track is not None and track.get("status") in ("in", "post"):
        try:
            track, live = _tracked(season, gid)     # refreshes a live game if the worker is behind
        except Exception as e:
            live = None
            meta["note"] = f"live feed unavailable ({e})"
        if live is not None:
            kind = "live" if track["status"] == "in" else "post"
            state_, timeline = live, track.get("points", [])
    if state_ is None and row.completed:
        kind = "post"
        try:
            state_ = _replay.game(gid, 999)
            timeline = _replay_timeline(season, gid)
        except Exception as e:
            meta["note"] = f"play-by-play not available ({e}); showing the final box score against the prior"
            state_ = _final_state_from_box(season, gid, row)
    if state_ is None:
        state_ = LiveState(gid, [row.home, row.away], ["", ""], bool(row.neutral), "pre" if kind == "pre" else "post",
                           1, 20.0, {}, detail="Scheduled" if kind == "pre" else "Final")
    view = _model_view(post, state_) if post is not None else None
    p_pre = meta.get("ledger", {}).get("p_home")
    if p_pre is None and view is not None:
        p_pre = float(view["pregame"]["p_win"][0])
    res = None
    if kind == "post" and state_ is not None and state_.box:
        res = tuple(state_.box.get(t, {}).get("pts", 0) for t in (row.home, row.away))
    meta |= game_flags(row.home, row.away, p_pre, rank_map(season), result=res)
    if kind == "live" and view is not None and meta.get("underdog"):
        p_live_dog = float(view["p_win"][0]) if meta["underdog"] == row.home else 1 - float(view["p_win"][0])
        meta["p_underdog_live"] = p_live_dog
    return jsonify({"kind": kind, "meta": meta, "state": state_.to_json(), "model": view, "posterior": label,
                    "timeline": timeline})


# ---------------------------------------------------------------------------
# Rankings lookup, upset flags and the model dashboard
# ---------------------------------------------------------------------------
UPSET_GAP = 20          # ranking places between the teams before a game counts as favourite vs underdog
UPSET_WATCH = 0.35      # the model gives the underdog at least this chance: "upset watch"


@lru_cache(maxsize=4)
def _rank_map(season: int, stamp: float) -> dict:
    rows, _ = ratings_for(season)
    d1 = data.d1_set(season)
    rows = [r for r in rows if r["team"] != "Non-D1" and (not d1 or r["team"] in d1)]
    return {r["team"]: i for i, r in enumerate(sorted(rows, key=lambda r: -r["net"]), 1)}


def rank_map(season: int) -> dict:
    """Current overall rank of every D1 team (1 = best)."""
    try:
        return _rank_map(season, _stamp(season))
    except Exception:
        return {}


def game_flags(home: str, away: str, p_home, ranks: dict, result=None) -> dict:
    """Both teams' current ranks and, when the ranking makes one team the clear underdog, its chance by the
    model: "watch" at 35% or more, "pick" when the model favours it outright (e.g. at home). A non-D1 team
    is always the underdog. With a result, whether the underdog won."""
    rh, ra = ranks.get(home), ranks.get(away)
    out = {"rank_home": rh, "rank_away": ra}
    if p_home is None or p_home != p_home or (rh is None and ra is None):
        return out
    p_home = float(p_home)
    if rh is None:
        dog, fav, p_dog = home, away, p_home
    elif ra is None:
        dog, fav, p_dog = away, home, 1 - p_home
    elif abs(rh - ra) < UPSET_GAP:
        return out
    elif rh < ra:
        dog, fav, p_dog = away, home, 1 - p_home
    else:
        dog, fav, p_dog = home, away, p_home
    out |= {"underdog": dog, "favorite": fav, "p_underdog": p_dog,
            "upset_level": "pick" if p_dog >= 0.5 else "watch" if p_dog >= UPSET_WATCH else None}
    if result:
        out["upset"] = (dog == home) == (result[0] > result[1])
    return out


def _ledger_records(led: pd.DataFrame, ranks: dict) -> list:
    recs = json.loads(led.to_json(orient="records", date_format="iso"))
    for g in recs:
        g |= game_flags(g["home"], g["away"], g.get("p_home"), ranks, result=(g["home_pts"], g["away_pts"]))
        won = g["home_pts"] > g["away_pts"]
        g["p_winner"] = g["p_home"] if won else 1 - g["p_home"]
        e = g.get("espn_p_home")
        g["espn_p_winner"] = None if e is None else (e if won else 1 - e)
    return recs


@app.route("/api/dashboard")
def api_dashboard():
    """Everything the model dashboard shows: out-of-sample performance, calibration, best and worst calls,
    and how the Bayesian updates have moved the ratings and shrunk their uncertainty."""
    season = int(request.args.get("season", LIVE_SEASON))
    post, label = season_post(season)
    path = model.artifact_dir(season) / "status.json"
    out = {"season": season, "posterior": label, "status": json.loads(path.read_text()) if path.exists() else {}}
    ranks = rank_map(season)
    led = season_mod.load_ledger(season)
    out["games"] = int(len(led))
    if len(led):
        led = led.sort_values("date")
        won = (led.home_pts > led.away_pts).to_numpy()
        p = led.p_home.astype(float).clip(1e-3, 1 - 1e-3).to_numpy()
        pw = np.where(won, p, 1 - p)
        ll = -np.log(pw)
        u = led.unusualness.astype(float).to_numpy()
        out["metrics"] = {"log_loss": float(ll.mean()), "brier": float(np.mean((p - won) ** 2)),
                          "accuracy": float(np.mean(pw > 0.5)), "n": int(len(led)),
                          "mean_conf": float(np.mean(np.maximum(p, 1 - p))),
                          "share_unusual_05": float(np.mean(u < 0.05)), "median_unusual": float(np.median(u))}
        # Calibration: how often the favourite won, by how strongly it was favoured.
        fav = np.maximum(p, 1 - p)
        edges = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0001])
        out["calibration"] = [{"lo": float(edges[i]), "hi": float(min(edges[i + 1], 1)), "n": int(m.sum()),
                               "predicted": float(fav[m].mean()) if m.any() else None,
                               "actual": float((pw[m] > 0.5).mean()) if m.any() else None}
                              for i in range(5) for m in [(fav >= edges[i]) & (fav < edges[i + 1])]]
        out["unusual_hist"] = np.histogram(u, bins=10, range=(0, 1))[0].tolist()
        days = led.date.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d").to_numpy() \
            if led.date.dt.tz is not None else led.date.dt.strftime("%Y-%m-%d").to_numpy()
        n = np.arange(1, len(led) + 1)
        cum_ll, cum_acc = np.cumsum(ll) / n, np.cumsum(pw > 0.5) / n
        last = np.r_[days[1:] != days[:-1], True]
        first = np.r_[True, days[1:] != days[:-1]]
        starts = np.flatnonzero(first)
        out["daily"] = [{"date": days[j], "games": int(n[j]), "day_games": int(j - starts[starts <= j][-1] + 1),
                         "log_loss": float(cum_ll[j]), "accuracy": float(cum_acc[j]),
                         "day_accuracy": float(np.mean(pw[starts[starts <= j][-1]:j + 1] > 0.5))}
                        for j in np.flatnonzero(last)]
        recs = _ledger_records(led, ranks)
        for g, l in zip(recs, ll):
            g["log_loss"] = float(l)
        right = [g for g in recs if g["p_winner"] > 0.5]
        out["best_calls"] = sorted(right, key=lambda g: -g["p_winner"])[:8]
        vs = [g for g in right if g["espn_p_winner"] is not None and g["espn_p_winner"] < 0.5]
        out["beat_espn"] = sorted(vs, key=lambda g: -(g["p_winner"] - g["espn_p_winner"]))[:8]
        out["upsets_called"] = sorted([g for g in recs if g.get("upset") and g.get("upset_level")],
                                      key=lambda g: -g["p_underdog"])[:8]
        out["most_wrong"] = sorted([g for g in recs if g["p_winner"] < 0.5], key=lambda g: g["p_winner"])[:10]
        out["most_unusual"] = sorted(recs, key=lambda g: g["unusualness"])[:8]
        out["vs_espn"] = _vs_espn(led)
    up = season_mod.load_updates(season)
    d1 = data.d1_set(season)
    if len(up):
        up = up[up.team.isin(d1)] if d1 else up
        up = up.sort_values(["date", "game_id"])
        up["k"] = up.groupby("team").cumcount() + 1
        # Uncertainty against games played: each team's sd before its first game, then after each one.
        sd = [{"games": 0, "sd": float(up[up.k == 1].sd_before.mean()), "teams": int((up.k == 1).sum())}]
        sd += [{"games": int(k), "sd": float(g.sd_after.mean()), "teams": int(len(g))}
               for k, g in up.groupby("k") if len(g) >= 3]
        out["uncertainty"] = sd
        per = up.groupby("team").agg(first=("net_before", "first"), last=("net_after", "last"),
                                     sd0=("sd_before", "first"), sd1=("sd_after", "last"), games=("k", "max"))
        per["moved"] = per["last"] - per["first"]
        mv = lambda df: [{"team": t, "moved": float(r.moved), "from": float(r["first"]), "to": float(r["last"]),
                          "sd_from": float(r.sd0), "sd_to": float(r.sd1), "games": int(r.games),
                          "rank": ranks.get(t)} for t, r in df.iterrows()]
        out["risers"] = mv(per[per.moved > 0].sort_values("moved", ascending=False).head(8))
        out["fallers"] = mv(per[per.moved < 0].sort_values("moved").head(8))
        out["moves_hist"] = np.histogram(up.net_after - up.net_before, bins=24, range=(-12, 12))[0].tolist()
        # The latest games' updates, both teams before and after.
        recent = []
        lp = {int(r.game_id): r for r in led.itertuples()} if len(led) else {}
        for gid, g in list(up.groupby("game_id", sort=False))[::-1][:30]:
            if len(g) != 2:
                continue
            l = lp.get(int(gid))
            teams_ = [{"team": r.team, "pts": int(r.pts), "before": float(r.net_before), "sd_before": float(r.sd_before),
                       "after": float(r.net_after), "sd_after": float(r.sd_after), "games": int(r.k),
                       "rank": ranks.get(r.team)} for r in g.itertuples()]
            p_first = None
            if l is not None:
                p_first = float(l.p_home) if l.home == teams_[0]["team"] else 1 - float(l.p_home)
            recent.append({"game_id": int(gid), "date": str(g.date.iloc[0])[:10], "teams": teams_, "p_first": p_first,
                           "source": g.source.iloc[0] if "source" in g else ""})
        out["recent_updates"] = sorted(recent, key=lambda r: r["date"], reverse=True)
        out["updates"] = int(len(up) // 2)
    if ranks:
        rows, _ = ratings_for(season)
        by = {r["team"]: r for r in rows}
        out["top"] = [{"team": t, "rank": k, "net": by[t]["net"], "net_lo": by[t]["net_lo"], "net_hi": by[t]["net_hi"]}
                      for t, k in sorted(ranks.items(), key=lambda x: x[1])[:12]]
    return jsonify(out)


if __name__ == "__main__":
    app.run(debug=True)
