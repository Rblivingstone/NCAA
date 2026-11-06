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


def test_shift_probability_superiority():
    from ncaa.ingame import superiority

    rng = np.random.default_rng(0)
    x = rng.poisson(10, 5000)
    assert superiority(x, x) == pytest.approx(0.5)
    assert superiority(x, rng.poisson(10, 5000)) == pytest.approx(0.5, abs=0.02)
    assert superiority(x + 20, x) > 0.999 and superiority(x, x + 20) < 0.001
    # A known final value: q is that value's mid-percentile in the pre-game distribution.
    pre = rng.poisson(10, 5000)
    q = superiority(np.full(10, 13), pre)
    assert q == pytest.approx((pre < 13).mean() + 0.5 * (pre == 13).mean())


def test_pick_correlation_keeps_shares_but_raises_agreement(bracket):
    from ncaa.pool import PoolSpec, simulate_public

    Wm = _random_W(len(bracket.teams), 12).mean(0)
    mix = (("casual", 0.5), ("chalk", 0.5))
    lo = simulate_public(bracket, Wm, 6000, PoolSpec(mix, 0.0), seed=1, climates=1)
    hi = simulate_public(bracket, Wm, 6000, PoolSpec(mix, 0.8), seed=1, climates=6000)
    r64 = [i for i, g in enumerate(bracket.games) if g.round == 1]
    top = [bracket.games[i].sources[0][1] for i in r64]
    share_lo = (lo[:, r64] == top).mean(0)
    share_hi = (hi[:, r64] == top).mean(0)
    assert np.abs(share_lo - share_hi).max() < 0.03                # same pick shares...
    hi_c = simulate_public(bracket, Wm, 6000, PoolSpec(mix, 0.8), seed=1, climates=3)
    agree = lambda P: (P[0::3][:1000] == P[3::3][:1000]).mean()  # noqa: E731  same climate pairs
    assert agree(hi_c) > agree(lo) + 0.03                           # ...but more similar brackets


def test_pool_finish_probabilities_match_monte_carlo(bracket, post):
    from ncaa.pool import Pool, PoolSpec, score_brackets

    sims = simulate_tournament(bracket, post, 600, seed=21)
    s = summarize(bracket, sims)
    Wm = _random_W(len(bracket.teams), 5).mean(0)
    pool = Pool(bracket, sims, Wm, s, PoolSpec(rho=0.5), n_public=900, seed=3, climates=3)
    mine = pool.leverage_bracket(0.0)
    rows = np.arange(600)
    ours = pool.scores(mine)
    clim = np.arange(900) % 3
    rng = np.random.default_rng(0)
    N = 9
    res = pool.finish(mine, N, rows)
    first, top, last = [], [], []
    for _ in range(30):
        for t in rows:
            riv = pool.public_picks[clim == t % 3]
            opp = score_brackets(riv[rng.integers(len(riv), size=N)], pool.winners[t:t + 1], pool.pts)[0]
            above, ties = (opp > ours[t]).sum(), (opp == ours[t]).sum()
            first.append(1 / (ties + 1) if above == 0 else 0)
            last.append(1 / (ties + 1) if (opp < ours[t]).sum() == 0 else 0)
    assert res["p_first"] == pytest.approx(np.mean(first), abs=0.02)
    assert res["p_last"] == pytest.approx(np.mean(last), abs=0.02)
    assert res["p_first"] <= res["p_top10"] + 1e-9
    w = pool.winning_score(N, rows)
    assert w["p10"] <= w["median"] <= w["p90"]


def test_team_analysis_is_zero_for_an_average_team(post):
    import dataclasses

    from ncaa.teams import skill_profile, style_vulnerabilities

    z = post.z.copy()
    z[:, 3] = 0.0                                              # make one team exactly average
    p = dataclasses.replace(post, z=z)
    team = post.teams[3]
    assert all(abs(r["impact"]) < 1e-9 for r in skill_profile(p, team))
    assert all(abs(s["relative"]) < 1e-9 for s in style_vulnerabilities(p, team))
    strong = post.teams[0]                                     # synthetic best shooter
    prof = {(r["side"], r["stat"]): r for r in skill_profile(post, strong)}
    assert prof[("off", "p2")]["impact"] > 0 and prof[("def", "p2")]["impact"] > 0


def test_laplace_update_moves_ratings_the_right_way(post):
    from ncaa.updates import init_state, laplace_update, net_rating

    st = init_state(post, "synthetic")
    st.cov *= 25.0                               # the synthetic posterior is very tight; give the prior room to move
    a, b = post.teams[30], post.teams[31]
    before = [net_rating(st, post, t) for t in (a, b)]
    hot = {"pts": 100, "fgm": 38, "fga": 60, "fg3m": 12, "fg3a": 25, "ftm": 12, "fta": 15, "oreb": 14,
           "dreb": 30, "tov": 6, "ast": 24, "stl": 9, "blk": 6, "pf": 12}
    cold = {"pts": 55, "fgm": 20, "fga": 66, "fg3m": 3, "fg3a": 24, "ftm": 12, "fta": 16, "oreb": 6,
            "dreb": 20, "tov": 18, "ast": 8, "stl": 3, "blk": 1, "pf": 18}
    laplace_update(st, post, a, b, 0.0, hot, cold, 40.0)
    after = [net_rating(st, post, t) for t in (a, b)]
    assert after[0][0] > before[0][0] + 0.5 and after[1][0] < before[1][0] - 0.5   # winner up, loser down
    assert after[0][1] < before[0][1] and after[1][1] < before[1][1]               # both less uncertain


def test_live_final_is_scored_then_applied_once(tmp_path, monkeypatch):
    import json
    import shutil
    from pathlib import Path

    from ncaa import model, season
    from ncaa.live import parse_summary
    from ncaa.updates import load_updates

    real = model.artifact_dir(2026)
    if not (real / "posterior.npz").exists():
        pytest.skip("needs the 2026 posterior (python -m ncaa build --season 2026)")
    monkeypatch.setattr(model, "ARTIFACTS", tmp_path)
    (tmp_path / "2026").mkdir()
    for f in ("posterior.npz", "teams.json"):
        shutil.copy(real / f, tmp_path / "2026" / f)
    final = parse_summary(json.loads((Path(__file__).parent / "fixtures" / "401856600.json").read_text()))
    assert season.apply_live_final(2026, final, log=lambda *_: None) == 1
    up = load_updates(2026)
    assert set(up.team) == {"Michigan", "UConn"} and len(up) == 2
    led = season.load_ledger(2026)
    assert len(led) == 1 and 0.5 < led.p_home.iloc[0] < 0.9      # the pre-game forecast, logged before updating
    assert season.apply_live_final(2026, final, log=lambda *_: None) == 0   # never applied twice

    raw = json.loads((Path(__file__).parent / "fixtures" / "401856600.json").read_text())

    class Feed:  # the watch loop picks finals off the scoreboard
        def scoreboard(self, date):
            return [{"game_id": final.game_id, "status": "post"}]

        def summary(self, gid):
            return raw

    season.watch(2026, every=0, log=lambda *_: None, feed=Feed(), rounds=1)
    assert len(load_updates(2026)) == 2


def test_tracker_follows_a_game_from_tipoff(tmp_path, monkeypatch):
    """The worker tracks a game from its scheduled start, with no one watching: the first poll mid-game
    reconstructs the history so far from the plays, later polls add points, and the final finishes the
    history and updates the ratings once."""
    import json
    import shutil
    from pathlib import Path

    import pandas as pd

    from ncaa import tracker
    from ncaa.live import summary_at
    from ncaa.updates import load_updates

    real = model.artifact_dir(2026)
    if not (real / "posterior.npz").exists():
        pytest.skip("needs the 2026 posterior (python -m ncaa build --season 2026)")
    monkeypatch.setattr(model, "ARTIFACTS", tmp_path)
    (tmp_path / "2026").mkdir()
    for f in ("posterior.npz", "teams.json"):
        shutil.copy(real / f, tmp_path / "2026" / f)
    raw = json.loads((Path(__file__).parent / "fixtures" / "401856600.json").read_text())
    gid = 401856600
    tip = pd.Timestamp("2026-04-07 01:20", tz="UTC")
    sched = pd.DataFrame({"game_id": [gid], "date": [tip], "home": ["Michigan"], "away": ["UConn"],
                          "neutral": [True], "completed": [False]})
    monkeypatch.setattr(tracker.data, "load_schedule", lambda season: sched)

    class Feed:  # ESPN as the game goes on; the scoreboard is down, so only the schedule says it's on
        minute = 0.0

        def scoreboard(self, date):
            raise OSError("scoreboard down")

        def summary(self, g):
            return raw if self.minute >= 41 else summary_at(raw, self.minute)

    feed = Feed()
    assert tracker.games_to_poll(2026, [], now=tip - pd.Timedelta(hours=1)) == ([], [])   # before tip-off
    assert tracker.games_to_poll(2026, [], now=tip + pd.Timedelta(minutes=30)) == ([gid], [])

    feed.minute = 12.5                 # first look 12.5 minutes in: minutes 0-12 rebuilt from the plays
    fc = tracker.Forecaster(2026)
    tr = tracker.track_game(2026, gid, feed, fc)
    el = [p["elapsed"] for p in tr["points"]]
    assert tr["status"] == "in" and el[0] == 0 and el[-1] == 12.5 and len(el) == 14
    assert all(0 < p["ours"] < 1 and set(p["q"]) for p in tr["points"])

    feed.minute = 13.2                 # the next poll adds its point (and the minute-13 grid point it skipped)
    tr = tracker.track_game(2026, gid, feed, fc)
    assert [p["elapsed"] for p in tr["points"]][-3:] == [12.5, 13.0, 13.2]

    feed.minute = 99                   # final: the rest of the game is filled in and the ratings update
    tr = tracker.track_game(2026, gid, feed, fc, log=lambda *_: None)
    el = [p["elapsed"] for p in tr["points"]]
    assert tr["status"] == "post" and tr["applied"] and el[-1] == 40 and len(el) == 43   # minutes 0-39, the two polls, the final
    assert tr["points"][-1]["ours"] in (0.0, 1.0) and tr["points"][-1]["score"] == [69, 63]
    assert len(load_updates(2026)) == 2
    assert tracker.load_track(2026, gid)["points"] == tr["points"]          # stored for the game page
    assert tracker.games_to_poll(2026, [], now=tip + pd.Timedelta(hours=3)) == ([], [])   # done
    tracker.run(2026, feed, every=0, log=lambda *_: None, rounds=1)
    assert len(load_updates(2026)) == 2                                     # never applied twice


def test_fallback_feed_uses_the_archive_when_espn_fails(monkeypatch):
    """A server ESPN won't answer still gets finished games, from the per-game archive."""
    import json
    from pathlib import Path

    from ncaa import live

    raw = json.loads((Path(__file__).parent / "fixtures" / "401856600.json").read_text())

    class Blocked:
        def summary(self, gid):
            raise OSError("403 Forbidden")

    urls = []
    monkeypatch.setattr(live, "_get_json", lambda url, timeout=20: urls.append(url) or raw)
    feed = live.FallbackFeed(Blocked())
    assert feed.game(401856600).status == "post" and urls == [live.RAW_GAME_JSON.format(game_id=401856600)]

    def missing(url, timeout=20):
        raise OSError("404")

    monkeypatch.setattr(live, "_get_json", missing)       # not archived yet (game still going): ESPN's error
    with pytest.raises(RuntimeError, match="403 Forbidden"):
        feed.summary(401856600)


def test_unseen_teams_start_at_their_groups_average():
    """No history: a D1 team starts at the average D1 team, a non-D1 school at the pooled non-D1 rating,
    both as uncertain as a random team."""
    real = model.artifact_dir(2026)
    if not (real / "posterior.npz").exists():
        pytest.skip("needs the 2026 posterior (python -m ncaa build --season 2026)")
    post = model.load_posterior(2026)
    d1 = {t for t in post.teams if t != "Non-D1"} | {"New D1 U"}
    ext = model.with_teams(post, ["New D1 U", "Tiny College", "Michigan"], d1)
    assert ext.teams[-2:] == ["New D1 U", "Tiny College"] and ext.z.shape[1] == len(post.teams) + 2
    assert model.with_teams(post, ["Tiny College", "New D1 U"], d1) is ext            # cached
    known = model.known_skills(post)
    new_d1, tiny = ext.z[:, -2].astype(float), ext.z[:, -1].astype(float)
    nd1 = post.z[:, post.teams.index("Non-D1")].astype(float)
    sd = known.std(0)
    assert np.all(np.abs(new_d1.mean(0) - known.mean(0)) < 0.25 * sd)                 # the average D1 team
    assert np.all(np.abs(tiny.mean(0) - nd1.mean(0)) < 0.25 * sd)                     # the average non-D1 opponent
    assert np.allclose(new_d1.std(0), np.sqrt(sd ** 2 + post.z[:, ~np.isin(post.teams, ["Non-D1"])].mean(1).std(0) ** 2), rtol=0.2)
    assert model.with_teams(post, ["Michigan"]) is post                               # nothing to add


def test_espn_pregame_forecast_goes_in_the_ledger(tmp_path, monkeypatch):
    """ESPN's pre-game probability is read at tip-off, oriented to the ledger's first team, recorded once,
    and a 50% placeholder (ESPN had no view) counts as no forecast."""
    import copy
    import json
    from pathlib import Path

    import pandas as pd

    from ncaa import season
    from ncaa.live import espn_pregame

    raw = json.loads((Path(__file__).parent / "fixtures" / "401856600.json").read_text())
    e = espn_pregame(raw)
    assert e["home"] == "Michigan" and abs(e["p_home"] - 0.697) < 1e-9
    flat = copy.deepcopy(raw)
    flat["winprobability"][0]["homeWinPercentage"] = 0.5
    assert espn_pregame(flat) is None

    monkeypatch.setattr(model, "ARTIFACTS", tmp_path)
    (tmp_path / "2026").mkdir()
    pd.DataFrame([{"game_id": 401856600, "date": "2026-04-07", "home": "UConn", "away": "Michigan",
                   "home_pts": 63, "away_pts": 69, "p_home": 0.4},
                  {"game_id": 1, "date": "2026-04-06", "home": "A", "away": "B", "home_pts": 1, "away_pts": 0,
                   "p_home": 0.5}]).to_csv(season.ledger_path(2026), index=False)

    class Feed:
        calls = 0

        def summary(self, gid):
            Feed.calls += 1
            if gid == 1:
                raise OSError("no such game")
            return raw

    assert season.fill_espn(2026, feed=Feed(), log=lambda *_: None) == 1
    led = season.load_ledger(2026).set_index("game_id")
    assert abs(led.loc[401856600, "espn_p_home"] - (1 - 0.697)) < 1e-9          # P(UConn), the first team
    assert led.loc[401856600, "espn_checked"] and not led.loc[1, "espn_checked"]  # retried later
    season.fill_espn(2026, feed=Feed(), log=lambda *_: None)
    assert Feed.calls == 3                                                       # the recorded game isn't fetched again


def test_non_d1_school_gets_its_own_rating_that_moves_with_its_games(tmp_path, monkeypatch):
    """A non-D1 school starts at the pooled non-D1 rating and drifts with its own results; the D1 side of
    the game updates exactly as before (against the pool), so nothing is counted twice."""
    import json
    import shutil
    from pathlib import Path

    import pandas as pd

    from ncaa import season
    from ncaa.live import parse_summary
    from ncaa.updates import net_rating

    real = model.artifact_dir(2026)
    if not (real / "posterior.npz").exists():
        pytest.skip("needs the 2026 posterior (python -m ncaa build --season 2026)")
    final = parse_summary(json.loads((Path(__file__).parent / "fixtures" / "401856600.json").read_text()))

    def run(tmp, school):
        monkeypatch.setattr(model, "ARTIFACTS", tmp)
        (tmp / "2026").mkdir(parents=True)
        for f in ("posterior.npz", "teams.json"):
            shutil.copy(real / f, tmp / "2026" / f)
        rows = []
        for k in range(3):                        # three close losses to Michigan
            rec = {"game_id": 900 + k, "date": pd.Timestamp("2026-04-07", tz="UTC") + pd.Timedelta(days=k),
                   "team": "Michigan", "opponent": school, "home": 1, "ot": 0, "kind": "regular season"}
            for s_ in season.LIVE_BOX:
                rec[s_], rec["opp_" + s_] = final.box["Michigan"][s_], final.box["UConn"][s_]
            rows.append(rec)
        season.process_final(2026, pd.DataFrame(rows), source="test", log=lambda *_: None)
        return season.current_state(2026, create=False)

    st = run(tmp_path / "named", "Tiny College")
    base, _ = season.base_posterior(2026)
    assert "Tiny College" in st.teams
    pooled0 = net_rating(season.current_state(2026, create=False), base, "Non-D1")
    tiny = net_rating(st, base, "Tiny College")
    led = season.load_ledger(2026)
    assert list(led.away) == ["Tiny College"] * 3                       # the school's name, not the pool
    assert led.p_home.iloc[2] < led.p_home.iloc[0]                       # later forecasts use its own rating

    pooled_only = run(tmp_path / "pooled", "Non-D1")                     # the same games, played as the pool
    i, j = st.index("Michigan"), pooled_only.index("Michigan")
    assert np.allclose(st.mean[i], pooled_only.mean[j]) and np.allclose(st.mean[st.index("Non-D1")],
                                                                        pooled_only.mean[pooled_only.index("Non-D1")])
    assert tiny[0] > pooled0[0] + 10                                     # well above the pool after 3 close games


def test_upset_flags_follow_the_ranking():
    """The lower-ranked team is the underdog; it is flagged at 35%+ ("watch") or when favoured ("pick"),
    a non-D1 team is always the underdog, close rankings are never flagged, and a result says who won."""
    from webapp.app import game_flags

    ranks = {"Big": 10, "Small": 200, "Near": 15}
    assert game_flags("Small", "Big", 0.40, ranks)["upset_level"] == "watch"          # home underdog at 40%
    assert game_flags("Small", "Big", 0.55, ranks)["upset_level"] == "pick"
    assert game_flags("Big", "Small", 0.80, ranks)["upset_level"] is None              # 20% is no alert
    assert "underdog" not in game_flags("Big", "Near", 0.5, ranks)                      # 5 places apart
    f = game_flags("Big", "Tiny College", 0.97, ranks, result=(60, 62))
    assert f["underdog"] == "Tiny College" and f["upset"] and f["rank_away"] is None
    assert game_flags("Small", "Big", 0.40, ranks, result=(70, 60))["upset"]
    assert not game_flags("Small", "Big", 0.40, ranks, result=(60, 70))["upset"]


def test_dashboard_api_shape():
    """The dashboard endpoint answers for the live season, with or without games scored yet."""
    import webapp.app as A

    if not (model.artifact_dir(A.SEASON) / "summary.json").exists():
        pytest.skip("needs the built tournament season")
    r = A.app.test_client().get("/api/dashboard")
    assert r.status_code == 200
    d = r.get_json()
    assert d["season"] == A.LIVE_SEASON and "posterior" in d and "games" in d
    if d["games"]:
        assert {"metrics", "calibration", "daily", "most_wrong", "best_calls"} <= set(d)
        assert all(g["p_winner"] < 0.5 for g in d["most_wrong"])
        assert sum(b["n"] for b in d["calibration"]) == d["games"]


def test_betting_line_and_spread_summary():
    """The line is the home team's expected margin (minus the quoted spread); the spread summary measures
    misses, bias and 90% coverage, and recovers an approximate prediction for old ledger rows."""
    import json
    from pathlib import Path

    from ncaa.live import betting_line
    from webapp.app import _spreads

    raw = json.loads((Path(__file__).parent / "fixtures" / "401856600.json").read_text())
    raw["pickcenter"] = [{"details": "MICH -4.5", "spread": -4.5, "provider": {"name": "Book"}}]
    ln = betting_line(raw)
    assert ln["home"] == "Michigan" and ln["home_margin"] == 4.5
    assert betting_line({**raw, "pickcenter": []}) is None

    recs = [{"game_id": 1, "date": "2026-11-05", "home": "A", "away": "B", "neutral": False, "home_pts": 80, "away_pts": 70,
             "pred_margin": 6.0, "margin_lo": -12.0, "margin_hi": 24.0, "line_margin": 8.0, "margin_percentile": 0.6},
            {"game_id": 2, "date": "2026-11-05", "home": "C", "away": "D", "neutral": True, "home_pts": 60, "away_pts": 70,
             "pred_margin": None, "margin_lo": None, "margin_hi": None, "line_margin": None, "margin_percentile": 0.5}]
    s = _spreads(recs)
    assert s["n"] == 2 and s["approx"] == 1
    assert abs(s["points"][1]["pred"] - (-10)) < 1e-6          # median of the simulations = the real margin
    assert abs(s["mae"] - 2.0) < 1e-6 and s["coverage90"] == 1.0
    assert s["vs_line"] == {"games": 1, "model_mae": 4.0, "line_mae": 2.0}


def test_upset_precision_recall_and_auc(monkeypatch):
    """Upset detection scores games with a pre-game ranking underdog: precision and recall at the 35% and
    50% flags, and a ROC-AUC that is 1 when every upset outscored every non-upset."""
    import numpy as np

    import webapp.app as A

    assert A._auc(np.array([0.9, 0.2]), np.array([True, False])) == 1.0
    assert A._auc(np.array([0.3, 0.3]), np.array([True, False])) == 0.5
    assert A._auc(np.array([0.3]), np.array([True])) is None
    ranks = {f"T{i}": i for i in range(1, 300)}
    monkeypatch.setattr(A, "_pregame_rankers", lambda season: (lambda date: (ranks, True)))
    g = lambda gid, dog_p, dog_won: {"game_id": gid, "date": "2026-11-05", "home": "T1", "away": "T200", "neutral": True,
                                     "p_home": 1 - dog_p, "home_pts": 60 if dog_won else 80, "away_pts": 70}
    recs = [g(1, 0.55, True), g(2, 0.40, False), g(3, 0.20, True), g(4, 0.10, False),
            {**g(5, 0.5, True), "away": "T5"}]                      # 4 places apart: not counted
    u = A._upset_detection(recs, 2027)
    assert u["games"] == 4 and u["upsets"] == 2 and u["base_rate"] == 0.5
    watch, pick = u["thresholds"]
    assert (watch["tp"], watch["fp"], watch["fn"]) == (1, 1, 1) and watch["precision"] == 0.5 and watch["recall"] == 0.5
    assert (pick["tp"], pick["fp"]) == (1, 0) and pick["precision"] == 1.0
    assert u["auc"] == 0.75                                          # upsets at .55, .20 vs non-upsets at .40, .10
    assert u["roc"][0] == {"fpr": 0.0, "tpr": 0.0, "t": 1.0} and u["roc"][-1]["fpr"] == 1.0 == u["roc"][-1]["tpr"]


def test_win_detection_over_all_games():
    """Every game as "first-listed team wins": accuracy, precision/recall for both outcomes at 50%, and
    ROC-AUC for the model and (on the games ESPN called) for ESPN."""
    import webapp.app as A

    g = lambda p, won, e=None: {"p_home": p, "home_pts": 80 if won else 60, "away_pts": 70, "espn_p_home": e}
    recs = [g(0.9, True, 0.8), g(0.7, True, 0.6), g(0.6, False, 0.7), g(0.3, False, 0.2), g(0.4, True)]
    w = A._win_detection(recs)
    assert w["games"] == 5 and w["accuracy"] == 0.6
    assert w["confusion"] == {"called_win_won": 2, "called_win_lost": 1, "called_loss_won": 1, "called_loss_lost": 1}
    assert abs(w["win"]["precision"] - 2 / 3) < 1e-9 and abs(w["win"]["recall"] - 2 / 3) < 1e-9
    assert w["loss"]["precision"] == 0.5 and w["loss"]["recall"] == 0.5
    assert abs(w["auc"] - 5 / 6) < 1e-9                      # winners .9 .7 .4 vs losers .6 .3: 5 of 6 pairs right
    assert w["espn"]["games"] == 4 and w["espn"]["auc"] == 0.75 and w["espn"]["model_auc_same_games"] == 1.0
    assert A._win_detection([]) == {"games": 0}


def test_auc_time_series_model_vs_espn():
    """Season-to-date and rolling ROC-AUC, model vs ESPN, one point per day once enough games have both calls,
    with standard errors; a better-informed forecaster shows a higher curve."""
    import numpy as np
    import pandas as pd

    import webapp.app as A

    rng = np.random.default_rng(1)
    days = pd.date_range("2026-11-03", "2027-02-28", freq="D")
    recs = []
    for d in days:
        for _ in range(40):
            truth = rng.normal(0, 1.5)
            won = rng.random() < 1 / (1 + np.exp(-truth))
            sig = lambda noise: float(1 / (1 + np.exp(-(truth + rng.normal(0, noise)))))
            recs.append({"date": d.strftime("%Y-%m-%dT00:00:00Z"), "p_home": sig(1.2), "espn_p_home": sig(0.3),
                         "home_pts": 70 if won else 60, "away_pts": 65})
    s = A._auc_series(recs, 2027)
    assert s["games"] == len(recs) and s["window_days"] == 30
    assert len(s["cumulative"]) == len(days) and len(s["rolling"]) == len(days)
    last = s["cumulative"][-1]
    assert last["games"] == len(recs) and last["espn"] > last["model"] + 0.03 and 0 < last["model_se"] < 0.02
    assert all(p["games"] <= 31 * 40 for p in s["rolling"])            # only the last 30 days
    assert s["since_jan"]["games"] == sum(r["date"] >= "2027-01-01" for r in recs)
    assert A._auc_series(recs[:10], 2027)["cumulative"] == []           # too few games yet


def test_unearned_winning_streaks():
    """A team winning while its box scores pull its rating down reads as an unearned streak with positive
    unexplained wins; one whose rating climbs with the wins reads as earned."""
    import numpy as np
    import pandas as pd

    from ncaa import streaks

    rng = np.random.default_rng(0)
    up, led, gid = [], [], 0
    teams = [f"T{i}" for i in range(30)]
    for k in range(8):
        for t in teams:
            gid += 1
            if t == "Hot":
                continue
            d = rng.normal(0, 1.5); p = 0.5
            won = rng.random() < 1 / (1 + np.exp(-d))           # results follow the play
            up.append({"game_id": gid, "date": f"2026-11-{k + 3:02d}", "team": t, "opponent": "X", "pts": 70 if won else 60,
                       "opp_pts": 65, "net_before": 0.0, "net_after": d})
            led.append({"game_id": gid, "home": t, "away": "X", "p_home": p})
    for name, dnet in (("Hot", -0.6), ("Real", 1.8)):            # 8 straight wins each, at 40% pre-game
        teams.append(name)
        for k in range(8):
            gid += 1
            up.append({"game_id": gid, "date": f"2026-11-{k + 3:02d}", "team": name, "opponent": "X", "pts": 71, "opp_pts": 68,
                       "net_before": 0.0, "net_after": dnet})
            led.append({"game_id": gid, "home": "X", "away": name, "p_home": 0.6})
    lg = streaks.league(pd.DataFrame(up), pd.DataFrame(led), teams)
    hot, real = lg["teams"]["Hot"], lg["teams"]["Real"]
    assert hot["wins"] == 8 and abs(hot["expected_wins"] - 3.2) < 1e-9 and abs(hot["beyond"] - 4.8) < 1e-9
    assert hot["streak"] == {**hot["streak"], "kind": "W", "length": 8} and hot["unearned_wins"] == 8 and hot["close_wins"] == 8
    assert lg["slope"] > 0 and hot["unexplained"] > real["unexplained"] and hot["unexplained_pct"] == 1.0
    assert hot["verdict"]["level"] == "unearned" and real["verdict"]["level"] == "earned"
    g = streaks.team_games(pd.DataFrame(up), pd.DataFrame(led), "Hot")
    assert list(g.won) == [True] * 8 and abs(g.p_pre.iloc[0] - 0.4) < 1e-9     # oriented to the team (it was the away side)


def test_results_reach_every_page_before_the_nightly_schedule(tmp_path, monkeypatch):
    """A game scored in the ledger (or final in the live tracker) shows as completed with its score even
    while the schedule file still lists it as unplayed; a game long past tip-off with no result anywhere is
    marked no-result; and a game folded in twice (live, then again after a refit) keeps one log row."""
    import json

    import pandas as pd

    from ncaa import data, season, tracker
    from ncaa.updates import _write_log_rows, load_updates, updates_path

    monkeypatch.setattr(model, "ARTIFACTS", tmp_path)
    (tmp_path / "2027").mkdir()
    now = pd.Timestamp.now(tz="UTC")
    sched = pd.DataFrame({"game_id": [1, 2, 3, 4], "date": [now - pd.Timedelta(days=1)] * 3 + [now + pd.Timedelta(days=1)],
                          "home": ["Utah", "Duke", "Kansas", "UCLA"], "away": ["Towson", "Army", "Iowa", "USC"],
                          "neutral": False, "completed": False, "home_score": 0, "away_score": 0, "tournament": False})
    monkeypatch.setattr(data, "load_schedule", lambda s: sched)
    pd.DataFrame([{"game_id": 1, "date": "2026-11-06T02:00:00Z", "home": "Towson", "away": "Utah",   # listed the other way
                   "home_pts": 57, "away_pts": 67, "p_home": 0.2, "unusualness": 0.5}]).to_csv(season.ledger_path(2027), index=False)
    (tmp_path / "2027" / "live").mkdir()
    (tmp_path / "2027" / "live" / "2.json").write_text(json.dumps(
        {"game_id": 2, "status": "post", "teams": ["Duke", "Army"], "points": [{"elapsed": 40, "score": [84, 50]}]}))
    s = season.schedule_with_results(2027).set_index("game_id")
    assert s.loc[1, "completed"] and (s.loc[1, "home_score"], s.loc[1, "away_score"]) == (67, 57) and s.loc[1, "result_source"] == "ledger"
    assert s.loc[2, "completed"] and (s.loc[2, "home_score"], s.loc[2, "away_score"]) == (84, 50) and s.loc[2, "result_source"] == "live tracker"
    assert not s.loc[3, "completed"] and s.loc[3, "no_result"]                 # a day past, nothing anywhere
    assert not s.loc[4, "completed"] and not s.loc[4, "no_result"]             # still to be played
    row = lambda src, after: {"game_id": 10, "date": "2026-11-05", "team": "High Point", "opponent": "UMES", "pts": 63,
                              "opp_pts": 49, "net_before": 1.0, "sd_before": 5, "net_after": after, "sd_after": 4.8, "source": src}
    _write_log_rows(updates_path(2027), pd.DataFrame([row("live feed", 2.0)]))
    _write_log_rows(updates_path(2027), pd.DataFrame([row("nightly box scores", 2.5)]))
    up = load_updates(2027)
    assert len(up) == 1 and up.source.iloc[0] == "nightly box scores" and up.net_after.iloc[0] == 2.5
    pd.concat([pd.DataFrame([row("live feed", 2.0)])] * 2).to_csv(updates_path(2027), index=False)   # an old duplicated log
    assert len(load_updates(2027)) == 1


def test_road_to_march_projection_accounting(monkeypatch):
    """The season projection plays out consistently: the whole field, exactly 64 in the round of 64 and one
    champion per simulation, one tournament champion per conference, and odds that are probabilities."""
    import numpy as np
    import pandas as pd

    from ncaa import data, projection

    real = model.artifact_dir(2026)
    if not (real / "posterior.npz").exists():
        pytest.skip("needs the 2026 posterior (python -m ncaa build --season 2026)")
    post = model.load_posterior(2026)
    teams = [t for t in post.teams if t != "Non-D1"][:96]
    conf = {t: 100 + i // 8 for i, t in enumerate(teams)}                    # 12 conferences of 8
    monkeypatch.setattr(data, "d1_set", lambda s: frozenset(teams))
    rng = np.random.default_rng(0)
    rows, gid = [], 0
    now = pd.Timestamp.now(tz="UTC")
    for k in range(20):
        order = rng.permutation(teams)
        for a, b in zip(order[::2], order[1::2]):
            gid += 1
            done = k < 8
            rows.append({"game_id": gid, "date": now + pd.Timedelta(days=k - 8), "home": a, "away": b, "neutral": False,
                         "home_conf": conf[a], "away_conf": conf[b], "completed": done, "tournament": False, "no_result": False,
                         "home_score": int(rng.integers(55, 90)) if done else 0, "away_score": int(rng.integers(55, 90)) if done else 0})
    for season, field, opening in ((2026, 68, 8), (2027, 76, 24)):        # 2027 on: 76 teams, a 12-game Opening Round
        out = projection.project(season, post, pd.DataFrame(rows), n_sims=60, log=lambda *_: None)
        T = out["teams"]
        assert out["field"] == field
        assert abs(sum(r["p_field"] for r in T) - field) < 1e-9
        assert abs(sum(r["reach"]["Round of 64"] for r in T) - 64) < 1e-9
        assert abs(sum(r["reach"]["First Four"] for r in T) - opening) < 1e-9
    assert len(T) == 96 and out["games_left"] == 12 * 48
    assert abs(sum(r["reach"]["Champion"] for r in T) - 1) < 1e-9 and abs(sum(r["reach"]["Final Four"] for r in T) - 4) < 1e-9
    assert abs(sum(r["p_conf_tourney"] for r in T) - 12) < 1e-9 and abs(sum(r["p_conf_regular"] for r in T) - 12) < 1e-9
    for r in T:
        assert 0 <= r["reach"]["Champion"] <= r["reach"]["Final Four"] <= r["reach"]["Sweet 16"] <= r["p_field"] <= 1
        assert r["wins_now"] + r["losses_now"] == 8 and abs(r["wins"] + r["losses"] - 20) < 1e-9
        assert abs(sum(r["seed_dist"]) - r["p_field"]) < 1e-3
    best = max(T, key=lambda r: r["reach"]["Champion"])
    assert best["p_field"] > 0.9 and best["avg_seed"] < 4


def test_why_did_the_rating_move(tmp_path, monkeypatch):
    """Each update's explanation: skill shares that add up to the rating change, the box-score evidence
    against expectation, and a headline; recorded when the update runs, latest line per game and team."""
    import json
    from pathlib import Path

    import numpy as np

    from ncaa import explain, updates
    from ncaa.live import parse_summary

    real = model.artifact_dir(2026)
    if not (real / "posterior.npz").exists():
        pytest.skip("needs the 2026 posterior (python -m ncaa build --season 2026)")
    post = model.load_posterior(2026)
    final = parse_summary(json.loads((Path(__file__).parent / "fixtures" / "401856600.json").read_text()))
    a, b = final.teams
    st = updates.init_state(post, "2026")
    st.cov = st.cov * 25                                            # a wide prior, so the game moves things
    za0, zb0 = st.mean[st.index(a)].copy(), st.mean[st.index(b)].copy()
    updates.laplace_update(st, post, a, b, 0.0, final.box[a], final.box[b])
    g = updates._game_arrays(final.box[a], final.box[b], 40.0)
    ex = explain.explain_update(a, b, za0, st.mean[st.index(a)], zb0, st.mean[st.index(b)], post, 0.0, g["k"], g["n"])
    for t, z0 in ((a, za0), (b, zb0)):
        shares = explain.skill_shares(z0, st.mean[st.index(t)], post.mu.mean(0))
        assert abs(sum(s["points"] for s in shares) - ex[t]["dnet"]) < 0.1 * max(1.0, abs(ex[t]["dnet"]))
        assert ex[t]["headline"].startswith("Rating") and len(ex[t]["evidence"]) == 3
    assert any("offensive rebounds: 22 of 47" in e for e in ex[a]["evidence"] + ex[b]["evidence"])   # UConn crashed the glass

    monkeypatch.setattr(model, "ARTIFACTS", tmp_path)
    explain.record(2027, 7, "2026-11-05", {"Utah": {"opponent": "Towson", "dnet": 1.0, "headline": "old", "evidence": []}})
    explain.record(2027, 7, "2026-11-05", {"Utah": {"opponent": "Towson", "dnet": 1.5, "headline": "new", "evidence": []}})
    got = explain.load(2027, team="Utah")
    assert len(got) == 1 and got[(7, "Utah")]["headline"] == "new"
    st2 = updates.init_state(post, "2026")
    updates.apply_game(2027, st2, post, 401856600, "2026-04-07", a, b, 0.0, final.box[a], final.box[b], 40.0,
                       np.eye(st2.mean.shape[1]), "test")
    assert {k[1] for k in explain.load(2027, game_id=401856600)} == {a, b}          # recorded by the update itself


def test_committee_sheets():
    """Team-sheet numbers: wins above bubble rise with wins and fall with losses to weak teams, quadrant
    counts add up to the games, and the score ranks a team that beat everyone first."""
    import numpy as np

    from ncaa import committee

    rng = np.random.default_rng(1)
    T, n = 40, 3
    eff = np.sort(rng.normal(0, 8, T))[::-1][None].repeat(n, 0).copy()        # team 0 strongest
    a, b = np.array([i for i in range(T) for j in range(T) if i < j]), np.array([j for i in range(T) for j in range(T) if i < j])
    home = rng.choice([-1, 0, 1], len(a))
    won = np.ones(len(a), dtype=bool)                                         # the better-rated team always wins
    won[(a == 30) & (b == 31)] = False                                        # but 30 loses to 31 at ...
    ex = {"sim": np.array([0]), "a": np.array([0]), "b": np.array([1]), "a_won": np.array([True])}  # one extra game, sim 0
    sh = committee.sheets(eff, {"a": a, "b": b, "home": home, "a_won": won}, ex, bubble_rank=10)
    assert sh["wab"][1, 0] > sh["wab"][1, 5] > sh["wab"][1, 20] > sh["wab"][1, 39]
    assert sh["games"][0, 0] == T and sh["games"][1, 0] == T - 1
    tot = sh["q1w"] + sh["q1l"] + sh["q2w"] + sh["q2l"]
    q34 = sh["games"] - tot                                                    # games in quadrants 3-4
    assert (sh["q34l"] <= q34 + 1e-9).all() and (tot <= sh["games"] + 1e-9).all()
    assert sh["q1l"][:, 0].sum() == 0 and (sh["net_rank"][:, 0] == 1).all()
    assert (np.argmax(sh["score"], axis=1) == 0).all()
    assert committee.earliest_round(0, 1) == 1 and committee.earliest_round(0, 3) == 2
    assert committee.earliest_round(0, 7) == 3 and committee.earliest_round(0, 15) == 4
    assert [committee.required_round(m) for m in (0, 1, 2, 3, 5)] == [2, 2, 3, 4, 4]


def test_bracket_rules():
    """The bracket keeps the committee's principles: a conference's top four in different regions, conference
    teams kept apart by how often they met, No. 1 seeds at the closest regional, no team at its own arena, no
    Sunday site for BYU, and a one-line move when nothing else works."""
    import itertools

    import numpy as np

    from ncaa import committee

    T = 64
    conf = np.array([i % 8 for i in range(T)])                                # 8 conferences, 8 teams each
    regionals = [{"id": f"r{i}", "lat": 40.0, "lon": lon, "days": "fri-sun" if i == 3 else "thu-sat"}
                 for i, lon in enumerate((-75, -88, -100, -120))]
    pods = [{"id": f"p{i}", "lat": 35.0 + (i % 2) * 8, "lon": -75 - 6 * i, "days": "fri-sun" if i % 2 else "thu-sat"}
            for i in range(8)]
    loc = {t: (38.0 + (t % 5), -72 - (t * 7) % 50) for t in range(T)}
    loc[3] = (40.0, -121)                                                     # a No. 1 seed in the far west
    blocked = {0: {"p0", "r0"}}
    sunday = {9}
    geo = {"loc": loc, "regionals": regionals, "pods": pods, "blocked": blocked}
    b = committee.Bracketer(conf, geo, sunday=sunday)
    entries = [[t] for t in range(T)]
    meets = {(0, 8): 3, (1, 9): 2, (2, 10): 1, (16, 56): 3}
    meet = lambda x, y: meets.get((min(x, y), max(x, y)), 0)
    placed, info = b.place(entries, meet)
    assert info["conflicts"] == 0
    reg = {e[0]: p for e, p in zip(entries, placed)}
    # every region gets one team per line, and seeds sit where the bracket puts them
    for line in range(16):
        assert sorted(reg[t][0] for t in range(4 * line, 4 * line + 4) if reg[t][2] == line + 1) == \
            sorted(set(reg[t][0] for t in range(4 * line, 4 * line + 4) if reg[t][2] == line + 1))
    assert sum(r[2] for r in placed) == 4 * sum(range(1, 17))
    # protected seeds: a conference's top four on lines 1-4 in four different regions
    prot = [t for t in range(T) if reg[t][2] <= 4]
    for c in set(conf[prot]):
        top = sorted([t for t in prot if conf[t] == c], key=lambda t: t)[:4]
        assert len({reg[t][0] for t in top}) == len(top)
    # same-conference meetings
    for x, y in itertools.combinations(range(T), 2):
        if conf[x] == conf[y] and reg[x][0] == reg[y][0]:
            assert committee.earliest_round(reg[x][1], reg[y][1]) >= committee.required_round(meet(x, y))
    # geography: the far-western No. 1 seed gets the western regional; nobody at a blocked site; BYU off Sundays
    assert reg[3][0] == 3
    assert reg[0][0] != 0 and info["pod_site"][reg[0][0]][reg[0][1] // 4] != 0
    r9 = reg[9][0]
    assert regionals[info["reg_site"][r9]]["days"] != "fri-sun"
    pod9 = info["pod_site"][r9][reg[9][1] // 4]
    assert pod9 is None or pods[pod9]["days"] != "fri-sun"
    # Final Four: best No. 1 seed's region against the fourth
    ones = sorted(range(4), key=lambda r: min(t for t in range(4) if reg[t][0] == r))
    assert info["region_order"] == [ones[0], ones[3], ones[1], ones[2]]
    # a line where no region satisfies the rules: every 5 seed and every 12 seed from one conference, all having
    # met three times (5 v 12 meet in the first round, and any 12 shares a half with a 5) -> moved or reported
    conf2 = np.arange(T) + 100
    group = [16, 17, 18, 19, 44, 45, 46, 47]
    conf2[group] = 7
    meets2 = {(min(x, y), max(x, y)): 3 for x, y in itertools.combinations(group, 2)}
    placed2, info2 = committee.Bracketer(conf2).place(entries, lambda x, y: meets2.get((min(x, y), max(x, y)), 0))
    assert info2["moves"] + info2["conflicts"] > 0
    assert sorted(r[2] for r in placed2) == sorted(r[2] for r in placed)       # still four teams on every line


def test_geo_load(tmp_path, monkeypatch):
    """Geography: team coordinates, and sites a team may not play at (its own arena, more than three games)."""
    import json

    import pandas as pd

    from ncaa import data, geo

    d = tmp_path / "2030"; d.mkdir()
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    pd.DataFrame([{"team": "Louisville", "venue": "KFC Yum! Center", "city": "Louisville", "state": "KY", "lat": 38.25, "lon": -85.76},
                  {"team": "Kentucky", "venue": "Rupp Arena", "city": "Lexington", "state": "KY", "lat": 38.05, "lon": -84.5}]
                 ).to_csv(d / "geo_teams.csv", index=False)
    pd.DataFrame([{"team": "Louisville", "venue": "KFC YUM! Center", "city": "Louisville", "state": "KY", "games": 16},
                  {"team": "Kentucky", "venue": "KFC YUM! Center", "city": "Louisville", "state": "KY", "games": 1}]
                 ).to_csv(d / "geo_venues.csv", index=False)
    site = lambda i, **k: {"id": f"s{i}", "city": "X", "state": "ZZ", "venue": None, "days": None, "lat": 40.0, "lon": -90.0, **k}
    pods = [site(0, city="Louisville", state="KY", venue="KFC Yum! Center")] + [site(i) for i in range(1, 8)]
    regs = [site(10 + i, region=f"R{i}") for i in range(4)]
    (d / "sites.json").write_text(json.dumps({"pods": pods, "regionals": regs}))
    g = geo.load(2030)
    assert g["blocked"] == {"Louisville": {"s0"}} and g["loc"]["Kentucky"] == (38.05, -84.5)
    assert geo.locate({("saint louis", "MO"): (1, 2)}, "St. Louis", "MO") == (1, 2)
    assert geo.locate(geo.cities() if (geo.GEO_DIR / "us_cities.csv").exists() else {("storrs", "CT"): (41.808, -72.25)}, "Storrs", "CT")
