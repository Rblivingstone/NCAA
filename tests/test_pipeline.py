import numpy as np
import pytest

from ncaa import data, model
from ncaa.bracket import (Bracket, actual_picks, resolve_picks, score_bracket, simulate_tournament,
                          summarize)
from ncaa.simulate import BOX_STATS, simulate_games


@pytest.fixture(scope="module")
def bracket():
    return Bracket(data.load_bracket(2026))


@pytest.fixture(scope="module")
def post(bracket):
    """A small synthetic posterior over the tournament teams (no MCMC needed)."""
    rng = np.random.default_rng(0)
    T, K = len(bracket.teams), 20
    strength = np.linspace(0.3, -0.3, T)
    z = np.zeros((K, T, model.D), dtype=np.float32)
    di = model.def_index()
    for s in ("p2", "p3"):
        j = model.STATS.index(s)
        z[:, :, j] = strength
        z[:, :, di[j]] = -strength
    z += rng.normal(0, 0.02, z.shape).astype(np.float32)
    rates = np.array([0.16, 0.2, 0.39, 0.51, 0.34, 0.3, 0.72, 0.53, 0.55, 0.1])  # STATS order, fouls last
    mu = np.append(np.log(rates / (1 - rates)), np.log(0.25))
    return model.Posterior(teams=bracket.teams, z=z, mu=np.tile(mu, (K, 1)), hfa=np.zeros((K, model.S)),
                           phi=np.full((K, model.S - 1), 200.0), mu_pace=np.full(K, np.log(68)),
                           sigma_pace=np.full(K, 0.07), omega=np.eye(model.D))


def test_bracket_shape(bracket):
    assert len(bracket.teams) == 68
    assert len(bracket.games) == 67
    assert sum(g.round == 0 for g in bracket.games) == 4
    assert len(bracket.possible_teams("R6")) == 68


def test_actual_results_cover_every_game(bracket):
    picks = actual_picks(bracket)
    assert len(picks) == 67
    assert len(resolve_picks(bracket, picks)) == 67


def test_inconsistent_picks_are_ignored(bracket):
    picks = actual_picks(bracket)
    r1 = next(g for g in bracket.games if g.id == "R1-East-0")
    loser = [bracket.teams[v] for k, v in r1.sources if bracket.teams[v] != picks["R1-East-0"]][0]
    bad = dict(picks, **{"R2-East-0": loser})  # a team that already lost cannot win later
    assert "R2-East-0" not in {p["id"] for p in resolve_picks(bracket, bad)}


def test_simulated_box_scores_are_consistent(post):
    n = 5000
    rng = np.random.default_rng(1)
    res = simulate_games(post, rng.integers(20, size=n), np.zeros(n, int), np.full(n, 67), rng)
    for s in BOX_STATS:
        assert res[s].shape == (n, 2)
    pts = 2 * (res["fgm"] - res["fg3m"]) + 3 * res["fg3m"] + res["ftm"]
    assert (pts == res["pts"]).all()
    assert (res["pts"][:, 0] != res["pts"][:, 1]).all()  # overtime breaks every tie
    assert (res["fg3m"] <= res["fg3a"]).all() and (res["fgm"] <= res["fga"]).all()
    assert 60 < res["pts"].mean() < 95
    assert res["a_wins"].mean() > 0.8  # strongest vs weakest team


def test_tournament_summary(bracket, post):
    sims = simulate_tournament(bracket, post, 500, seed=2)
    s = summarize(bracket, sims)
    for g in s["games"]:
        assert abs(sum(t["p_win"] for t in g["teams"]) - 1) < 1e-9
        for side in g["sides"]:
            assert abs(sum(side.values()) - 1) < 1e-9
    champ = np.array(s["reach"])[:, -1]
    assert abs(champ.sum() - 1) < 1e-9


def test_score_with_coin_flip_matrix(bracket, post):
    sims = simulate_tournament(bracket, post, 200, seed=3)
    s = summarize(bracket, sims)
    W = np.full((5, 68, 68), 0.5, dtype=np.float32)
    r = score_bracket(bracket, actual_picks(bracket), W, s, sims)
    assert r["n_games"] == 67
    assert r["log_likelihood"] == pytest.approx(67 * np.log(0.5), rel=1e-5)
    assert 0 <= r["percentile"]["percentile"] <= 1


def _greedy(bracket, Wm):
    winners = {}
    for g in bracket.games:
        a, b = (v if kind == "team" else winners[v] for kind, v in g.sources)
        winners[g.id] = a if Wm[a, b] >= 0.5 else b
    return winners


def test_optimal_bracket_beats_greedy_when_matchups_matter(bracket):
    from ncaa.bracket import bracket_log_likelihood, most_likely_bracket

    T = len(bracket.teams)
    W = np.full((3, T, T), 0.5, dtype=np.float64)
    seed = {(bracket.region_of[t], bracket.seed[t]): bracket.tidx[t] for t in bracket.teams}
    one, sixteen = seed[("East", 1)], seed[("East", 16)]
    others = [seed[("East", 8)], seed[("East", 9)]]
    # Greedy takes the 16 seed (55%), but any bracket through it scores at most
    # 0.55 * 0.5 in the next game, while the 1 seed path gives 0.45 * 0.99.
    W[:, sixteen, one], W[:, one, sixteen] = 0.55, 0.45
    for o in others:
        W[:, one, o], W[:, o, one] = 0.99, 0.01
    greedy = _greedy(bracket, W.mean(0))
    best = most_likely_bracket(bracket, W)
    assert greedy["R1-East-0"] == sixteen
    assert best["picks"]["R1-East-0"] == one
    assert best["log_likelihood"] == pytest.approx(bracket_log_likelihood(bracket, greedy, W)
                                                   + np.log(0.45 * 0.99 / (0.55 * 0.5)))


def test_optimal_bracket_is_at_least_as_good_as_random_brackets(bracket):
    from ncaa.bracket import bracket_log_likelihood, most_likely_bracket

    rng = np.random.default_rng(5)
    T = len(bracket.teams)
    base = rng.beta(2, 2, size=(T, T))
    W = np.empty((4, T, T))
    for k in range(4):
        m = np.clip(base + rng.normal(0, 0.05, base.shape), 0.02, 0.98)
        m = np.triu(m, 1)
        W[k] = m + (1 - m.T) * np.tril(np.ones((T, T)), -1)
        np.fill_diagonal(W[k], 0.5)
    best = most_likely_bracket(bracket, W)
    assert best["log_likelihood"] >= bracket_log_likelihood(bracket, _greedy(bracket, W.mean(0)), W) - 1e-9
    for _ in range(200):
        winners = {}
        for g in bracket.games:
            a, b = (v if kind == "team" else winners[v] for kind, v in g.sources)
            winners[g.id] = a if rng.random() < 0.5 else b
        assert best["log_likelihood"] >= bracket_log_likelihood(bracket, winners, W) - 1e-9


def test_max_expected_points_bracket_is_consistent(bracket, post):
    from ncaa.bracket import ESPN_POINTS, expected_points, max_expected_points_bracket

    sims = simulate_tournament(bracket, post, 400, seed=4)
    s = summarize(bracket, sims)
    res = max_expected_points_bracket(bracket, s)
    named = {g: bracket.teams[t] for g, t in res["picks"].items()}
    assert len(resolve_picks(bracket, named)) == 67               # a valid, complete bracket
    assert expected_points(bracket, named, s) == pytest.approx(res["expected_points"])
    chalk = {g["id"]: bracket.teams[max(g["teams"], key=lambda t: t["p_win"])["team"]] for g in s["games"]}
    assert res["expected_points"] >= expected_points(bracket, chalk, s) - 1e-9 or \
        len(resolve_picks(bracket, chalk)) < 67
    assert ESPN_POINTS[6] == 320
