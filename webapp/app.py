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
from ncaa.bracket import (Bracket, actual_picks, bracket_log_likelihood, expected_points, favourites,
                          load_simulation, max_expected_points_bracket, most_likely_bracket, score_bracket)
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


@lru_cache(maxsize=1)
def optimal_brackets():
    br, _, summary, W, _ = state()
    named = lambda picks: {g: br.teams[t] for g, t in picks.items()}  # noqa: E731
    greedy = favourites(br, W.mean(0))
    ml = most_likely_bracket(br, W)
    ep = max_expected_points_bracket(br, summary)
    out = {}
    for key, picks in (("favourites", greedy), ("likelihood", ml["picks"]), ("points", ep["picks"])):
        out[key] = {"picks": named(picks), "log_likelihood": bracket_log_likelihood(br, picks, W),
                    "expected_points": expected_points(br, named(picks), summary),
                    "champion": br.teams[picks["R6"]]}
    out["likelihood"]["differs_from_favourites"] = [g for g in greedy if greedy[g] != ml["picks"][g]]
    return out


@app.route("/api/optimal")
def api_optimal():
    """Globally optimal brackets: most likely, and highest expected (ESPN) score."""
    return jsonify(optimal_brackets())


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
