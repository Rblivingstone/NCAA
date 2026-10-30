"""Flask front end for the bracket simulator.

    python -m ncaa build --season 2026      # once: fetch, fit, simulate
    flask --app webapp.app run              # then browse http://127.0.0.1:5000
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from flask import Flask, abort, jsonify, render_template, request

from ncaa import data, model
from ncaa.bracket import (Bracket, Constraints, actual_picks, bracket_log_likelihood, expected_points,
                          favourites, is_upset, load_simulation, max_expected_points_bracket, most_likely_bracket,
                          resolve_picks, score_bracket, upset_distribution)
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
    br = state()[0]
    return jsonify(actual_picks(br))


@app.route("/api/score", methods=["POST"])
def api_score():
    br, sims, summary, W, _ = state()
    if W is None:
        abort(503, "win matrix missing; run `python -m ncaa simulate`")
    picks = (request.get_json(silent=True) or {}).get("picks", {})
    out = score_bracket(br, picks, W, summary, sims)
    out["expected_points"] = expected_points(br, picks, summary)
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
    br, _, summary, W, _ = state()
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


@app.route("/api/upsets")
def api_upsets():
    """How many upsets the simulated tournaments contain, for each seed-gap definition."""
    br, sims, *_ = state()
    return jsonify({gap: upset_distribution(br, sims, gap) for gap in (1, 3, 5)})


@app.route("/api/teams")
def api_teams():
    return jsonify(state()[2]["ratings"])


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
               "p10": np.percentile(res[s], 10, axis=0).tolist(),
               "p90": np.percentile(res[s], 90, axis=0).tolist()} for s in BOX_STATS}
    samples = [{s: res[s][i].tolist() for s in BOX_STATS} | {"ot": int(res["ot"][i])} for i in range(8)]
    return jsonify({
        "teams": [a, b], "n": n, "p_a_wins": float(res["a_wins"].mean()),
        "p_ot": float((res["ot"] > 0).mean()),
        "margin": {"mean": float(margin.mean()), "sd": float(margin.std()),
                   "hist": np.histogram(margin, bins=edges)[0].tolist(), "edges": edges.tolist()},
        "total": {"mean": float(pts.sum(1).mean()), "sd": float(pts.sum(1).std())},
        "box": box, "samples": samples,
    })


@app.route("/api/team_list")
def api_team_list():
    return jsonify(sorted(state()[4].teams))


if __name__ == "__main__":
    app.run(debug=True)
