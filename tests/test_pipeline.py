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


def _random_W(T, seed, draws=4):
    rng = np.random.default_rng(seed)
    base = rng.beta(2, 2, size=(T, T))
    W = np.empty((draws, T, T))
    for k in range(draws):
        m = np.clip(base + rng.normal(0, 0.05, base.shape), 0.02, 0.98)
        m = np.triu(m, 1)
        W[k] = m + (1 - m.T) * np.tril(np.ones((T, T)), -1)
        np.fill_diagonal(W[k], 0.5)
    return W


def test_upset_constrained_dp_matches_brute_force(bracket):
    import itertools

    from ncaa.bracket import Constraints, _dp, is_upset

    T = len(bracket.teams)
    L = np.log(_random_W(T, 11).mean(0))
    pod = ["R1-East-0", "R1-East-1", "R1-East-2", "R1-East-3", "R2-East-0", "R2-East-1", "R3-East-0"]
    # Everything outside the pod is fixed: greedy on L, except the East final
    # goes to the bottom half so it does not depend on the pod.
    fixed = {}
    for g in bracket.games:
        a, b = (v if kind == "team" else fixed.get(v) for kind, v in g.sources)
        if g.id in pod:
            continue
        if g.id == "R4-East-0":
            fixed[g.id] = b
            continue
        fixed[g.id] = a if L[a, b] >= L[b, a] else b
    games = {g.id: g for g in bracket.games}

    def score(picks):
        total, ups = 0.0, 0
        for g in bracket.games:
            a, b = (v if kind == "team" else picks[v] for kind, v in g.sources)
            w = picks[g.id]
            l = b if w == a else a
            total += L[w, l]
            ups += is_upset(bracket, w, l)
        return total, ups

    best_by_k = {}
    for choice in itertools.product([0, 1], repeat=len(pod)):
        picks = dict(fixed)
        for gid, side in zip(pod, choice):
            kind, v = games[gid].sources[side]
            picks[gid] = v if kind == "team" else picks[v]
        sc, k = score(picks)
        best_by_k[k] = max(best_by_k.get(k, -np.inf), sc)
    for k, target in best_by_k.items():
        picks, val, _ = _dp(bracket, lambda g: L, Constraints(forced=fixed, n_upsets=k))
        assert val == pytest.approx(target)
        assert score(picks) == (pytest.approx(target), k)


def test_best_bracket_by_champion(bracket):
    from ncaa.bracket import best_bracket_by_champion, most_likely_bracket

    W = _random_W(len(bracket.teams), 3)
    T = len(bracket.teams)
    per_champ = best_bracket_by_champion(bracket, W, range(T))
    assert all(r["picks"]["R6"] == r["champion"] for r in per_champ)
    # Restarting from every champion can only help, and the overall search
    # (which restarts from the most promising champions) is never worse than
    # any single champion's best bracket that it tried.
    overall = most_likely_bracket(bracket, W, n_champions=T)
    assert overall["log_likelihood"] >= per_champ[0]["log_likelihood"] - 1e-9
    few = most_likely_bracket(bracket, W, n_champions=0)
    assert overall["log_likelihood"] >= few["log_likelihood"] - 1e-9


def test_espn_points_prediction_and_realisation(bracket, post):
    from ncaa.bracket import espn_points, expected_points

    sims = simulate_tournament(bracket, post, 300, seed=6)
    s = summarize(bracket, sims)
    actual = actual_picks(bracket)
    r = espn_points(bracket, actual, sims, actual)
    assert r["realized"] == 1920                                  # a perfect bracket scores everything
    assert r["expected"] == pytest.approx(expected_points(bracket, actual, s))
    assert r["delta"] == pytest.approx(r["realized"] - r["expected"])
    assert sum(b["realized_points"] for b in r["by_round"]) == r["realized"]


def test_one_shot_tournament_is_reproducible_with_box_scores(bracket, post):
    a = simulate_tournament(bracket, post, 1, seed=42, full_box=True)
    b = simulate_tournament(bracket, post, 1, seed=42, full_box=True)
    assert (a["winners"] == b["winners"]).all()
    assert (a["box"]["pts"] == a["scores"]).all()
    named = {g.id: bracket.teams[int(a["winners"][0, i])] for i, g in enumerate(bracket.games)}
    assert len(resolve_picks(bracket, named)) == 67               # a complete, consistent bracket


def test_public_brackets_are_valid_and_follow_chalkiness(bracket):
    from ncaa.pool import PublicModel, simulate_public

    T = len(bracket.teams)
    Wm = _random_W(T, 8).mean(0)
    picks = simulate_public(bracket, Wm, 200, PublicModel((0.0, 0.0), (1e4, 1e4)), seed=1)
    for row in picks[:20]:
        named = {g.id: bracket.teams[int(t)] for g, t in zip(bracket.games, row)}
        assert len(resolve_picks(bracket, named)) == 67
    # Pure seed-following with extreme chalkiness: every R64 game goes to the better seed.
    r64 = [i for i, g in enumerate(bracket.games) if g.round == 1]
    seeds = np.array([bracket.seed[t] for t in bracket.teams])
    for i in r64:
        g = bracket.games[i]
        a, b = (v for kind, v in g.sources if kind == "team") if all(k == "team" for k, _ in g.sources) else (None, None)
        if a is not None:
            assert (seeds[picks[:, i]] == min(seeds[a], seeds[b])).all()


def test_pool_win_probability_matches_monte_carlo(bracket, post):
    from ncaa.pool import Pool, PublicModel

    sims = simulate_tournament(bracket, post, 400, seed=9)
    s = summarize(bracket, sims)
    Wm = _random_W(len(bracket.teams), 4).mean(0)
    pool = Pool(bracket, sims, Wm, s, PublicModel(), n_public=500, seed=2)
    mine = pool.leverage_bracket(0.0)
    rows = np.arange(400)
    ours = pool.scores(mine)
    from ncaa.pool import score_brackets
    opp = score_brackets(pool.public_picks, pool.winners, pool.pts)
    rng = np.random.default_rng(0)
    for N in (1, 5, 30):
        exact = pool.win_probability(mine, N, rows)
        trials = []
        for _ in range(40):
            idx = rng.integers(opp.shape[1], size=(400, N))
            best = opp[rows[:, None], idx]
            top = best.max(1)
            ties = (best == top[:, None]).sum(1)
            trials.append(np.where(ours > top, 1.0, np.where(ours == top, 1.0 / (ties + 1), 0.0)).mean())
        assert exact == pytest.approx(np.mean(trials), abs=0.01)


def test_unusualness_is_uniform_for_games_from_the_model(post):
    from types import SimpleNamespace

    from ncaa.evaluate import JOINT_STATS, game_unusualness

    rng = np.random.default_rng(3)
    n = 3000
    res = simulate_games(post, rng.integers(20, size=n), np.full(n, 5), np.full(n, 40), rng)
    fresh = simulate_games(post, rng.integers(20, size=200), np.full(200, 5), np.full(200, 40), rng)
    ps = []
    for i in range(200):
        row = SimpleNamespace(**{s: fresh[s][i, 0] for s in JOINT_STATS + ["pts"]},
                              **{"opp_" + s: fresh[s][i, 1] for s in JOINT_STATS + ["pts"]})
        ps.append(game_unusualness(post, 5, 40, 0, row, res=res)["p_value"])
    ps = np.array(ps)
    assert 0.4 < np.median(ps) < 0.6
    assert np.mean(ps < 0.1) < 0.17


def test_live_tournament_respects_results(bracket, post):
    picks = actual_picks(bracket)
    first_round = {g.id: bracket.tidx[picks[g.id]] for g in bracket.games if g.round <= 1}
    sims = simulate_tournament(bracket, post, 300, seed=5, forced=first_round, live={"R2-East-0": 1.0})
    for gid, t in first_round.items():
        assert (sims["winners"][:, bracket.gidx[gid]] == t).all()
    i = bracket.gidx["R2-East-0"]
    assert (sims["winners"][:, i] == sims["sides"][:, i, 0]).all()        # live P(top wins) = 1


def test_replay_box_matches_official_box_and_forecast_is_sane(post):
    import json
    from pathlib import Path

    from ncaa.ingame import forecast
    from ncaa.live import parse_summary

    d = json.loads((Path(__file__).parent / "fixtures" / "401856600.json").read_text())
    final = parse_summary(d)
    replayed = parse_summary(d, until=999)
    assert final.status == "post" and replayed.status == "post"
    assert final.box == replayed.box                     # play-by-play rebuild == official box score
    half = parse_summary(d, until=20)
    assert half.period == 1 and half.box["Michigan"]["pts"] == 33 and half.box["UConn"]["pts"] == 29
    # A game that is over is certain; a game in progress keeps the score so far.
    f = forecast(post, 0, 1, 0, replayed, n=300, seed=0)
    assert f["p_win"][0] == 1.0
    g = forecast(post, 0, 1, 0, half, n=300, seed=0)
    assert (g["samples"]["pts"][:, 0] >= 33).all() and (g["samples"]["pts"][:, 1] >= 29).all()
    assert 0 < g["p_win"][0] < 1


def test_live_parser_reads_nested_espn_plays():
    """ESPN's live API nests play fields ({"type": {"text": ...}}); the archive flattens them."""
    import json
    from pathlib import Path

    from ncaa.live import parse_summary

    d = json.loads((Path(__file__).parent / "fixtures" / "401856600.json").read_text())

    def nest(p):
        out = {}
        for k, v in p.items():
            node = out
            *path, last = k.split(".")
            for part in path:
                node = node.setdefault(part, {})
            node[last] = v
        return out

    nested = dict(d, plays=[nest(p) for p in d["plays"]])
    for minute in (12.5, 33.0, 999):
        assert parse_summary(nested, until=minute).box == parse_summary(d, until=minute).box


def test_preseason_posterior_carries_last_season_over(post):
    from ncaa.model import preseason_posterior

    pre = preseason_posterior(post, rho=0.65)
    zbar_prev, zbar_new = post.z.mean(0), pre.z.mean(0)
    slope = (zbar_prev * zbar_new).sum() / (zbar_prev ** 2).sum()
    assert 0.55 < slope < 0.75                                     # shrunk toward average by ~rho
    assert pre.z.std(0).mean() > post.z.std(0).mean()              # and more uncertain
