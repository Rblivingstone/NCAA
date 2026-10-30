"""Bracket structure, Monte Carlo tournament simulation, and bracket scoring."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .model import Posterior, artifact_dir
from .simulate import simulate_games

# First-round pairings by seed, top to bottom within a region.
SEED_ORDER = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
ROUNDS = ["First Four", "Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship"]


@dataclass
class Game:
    id: str
    round: int                      # index into ROUNDS
    region: str | None
    sources: list                   # two entries: ("team", field_idx) or ("game", game_id)
    label: str = ""
    seeds: list = field(default_factory=list)  # seed line of each source (regional rounds)


class Bracket:
    """The 68-team field and the 67 games connecting it."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.regions: list[str] = spec["regions"]
        self.teams: list[str] = []
        self.seed: dict[str, int] = {}
        self.region_of: dict[str, str] = {}
        for r in self.regions:
            for s, names in spec["seeds"][r].items():
                for t in names:
                    self.teams.append(t)
                    self.seed[t] = int(s)
                    self.region_of[t] = r
        self.tidx = {t: i for i, t in enumerate(self.teams)}
        self.games: list[Game] = []
        self._build()
        self.gidx = {g.id: i for i, g in enumerate(self.games)}

    def _add(self, g: Game) -> str:
        self.games.append(g)
        return g.id

    def _build(self):
        region_final = {}
        for r in self.regions:
            seeds = self.spec["seeds"][r]

            def entry(s):
                names = seeds[str(s)]
                if len(names) == 1:
                    return ("team", self.tidx[names[0]])
                gid = self._add(Game(f"FF-{r}-{s}", 0, r, [("team", self.tidx[n]) for n in names],
                                     f"{r} {s}-seed play-in", [s, s]))
                return ("game", gid)

            prev = []
            for i, (s1, s2) in enumerate(SEED_ORDER):
                prev.append(self._add(Game(f"R1-{r}-{i}", 1, r, [entry(s1), entry(s2)], seeds=[s1, s2])))
            for rnd in (2, 3, 4):
                nxt = []
                for i in range(0, len(prev), 2):
                    nxt.append(self._add(Game(f"R{rnd}-{r}-{i // 2}", rnd, r,
                                              [("game", prev[i]), ("game", prev[i + 1])])))
                prev = nxt
            region_final[r] = prev[0]
        semis = []
        for i, (r1, r2) in enumerate(self.spec["semifinals"]):
            semis.append(self._add(Game(f"R5-{i}", 5, None, [("game", region_final[r1]),
                                                              ("game", region_final[r2])], f"{r1} vs {r2}")))
        self._add(Game("R6", 6, None, [("game", semis[0]), ("game", semis[1])], "National Championship"))

    def possible_teams(self, gid: str) -> list[int]:
        """Field indices that could appear in a game (from either side)."""
        g = self.games[self.gidx[gid]]
        return [t for side in range(2) for t in self.side_teams(g, side)]

    def side_teams(self, g: Game, side: int) -> list[int]:
        kind, v = g.sources[side]
        if kind == "team":
            return [v]
        return self.possible_teams(v)

    def to_json(self) -> dict:
        return {
            "teams": [{"name": t, "seed": self.seed[t], "region": self.region_of[t]} for t in self.teams],
            "regions": self.regions,
            "semifinals": self.spec["semifinals"],
            "games": [{"id": g.id, "round": g.round, "region": g.region, "label": g.label,
                       "sources": [list(s) for s in g.sources]} for g in self.games],
            "rounds": ROUNDS,
        }


def simulate_tournament(bracket: Bracket, post: Posterior, n_sims: int, seed: int | None = 0,
                        full_box: bool = False) -> dict:
    """Play the whole tournament ``n_sims`` times.

    Each simulated tournament uses a single posterior draw for every game, so
    parameter uncertainty is carried through the bracket coherently.  With
    ``full_box`` the complete simulated box score of every game is returned
    too (``box[stat]`` is sims x games x 2, plus ``ot``).
    """
    from .simulate import BOX_STATS

    rng = np.random.default_rng(seed)
    draws = rng.permutation(np.resize(np.arange(post.n_draws), n_sims))
    pidx = post.team_index(bracket.teams)
    G = len(bracket.games)
    winners = np.full((n_sims, G), -1, dtype=np.int16)
    sides = np.full((n_sims, G, 2), -1, dtype=np.int16)
    scores = np.zeros((n_sims, G, 2), dtype=np.int16)
    box = {s: np.zeros((n_sims, G, 2), dtype=np.int16) for s in BOX_STATS} if full_box else None
    ot = np.zeros((n_sims, G), dtype=np.int16)
    for rnd in range(len(ROUNDS)):
        gs = [i for i, g in enumerate(bracket.games) if g.round == rnd]
        if not gs:
            continue
        for i in gs:
            for side, (kind, v) in enumerate(bracket.games[i].sources):
                sides[:, i, side] = v if kind == "team" else winners[:, bracket.gidx[v]]
        top, bot = sides[:, gs, 0].ravel(), sides[:, gs, 1].ravel()
        k = np.repeat(draws, len(gs))
        res = simulate_games(post, k, pidx[top], pidx[bot], rng, full_box=full_box)
        win = np.where(res["a_wins"], top, bot).reshape(n_sims, len(gs))
        winners[:, gs] = win
        scores[:, gs] = res["pts"].reshape(n_sims, len(gs), 2)
        ot[:, gs] = res["ot"].reshape(n_sims, len(gs))
        if full_box:
            for s in BOX_STATS:
                box[s][:, gs] = res[s].reshape(n_sims, len(gs), 2)
    out = {"winners": winners, "sides": sides, "scores": scores, "draws": draws}
    if full_box:
        out |= {"box": box, "ot": ot}
    return out


def summarize(bracket: Bracket, sims: dict, top_n_matchups: int = 5) -> dict:
    """Distribution bracket: for every game, who is likely to be there and to win."""
    winners, sides, scores = sims["winners"], sims["sides"], sims["scores"]
    n = winners.shape[0]
    T = len(bracket.teams)
    games = []
    for i, g in enumerate(bracket.games):
        side_p = [np.bincount(sides[:, i, s], minlength=T) / n for s in range(2)]
        win_p = np.bincount(winners[:, i], minlength=T) / n
        appear = side_p[0] + side_p[1]
        pair = sides[:, i, 0].astype(np.int32) * T + sides[:, i, 1]
        common = Counter(pair.tolist()).most_common(top_n_matchups)
        matchups = []
        for code, cnt in common:
            a, b = divmod(code, T)
            m = pair == code
            matchups.append({"teams": [int(a), int(b)], "p": cnt / n,
                             "p_top_wins": float((winners[m, i] == a).mean()),
                             "avg_score": [float(scores[m, i, 0].mean()), float(scores[m, i, 1].mean())]})
        teams = [{"team": int(t), "side": int(side_p[1][t] > side_p[0][t]), "p_here": float(appear[t]),
                  "p_win": float(win_p[t])} for t in np.argsort(-appear) if appear[t] > 0]
        games.append({"id": g.id, "sides": [{str(t): float(p[t]) for t in np.flatnonzero(p)} for p in side_p],
                      "teams": teams, "matchups": matchups})
    # Probability each team reaches (i.e. plays in) each round, and wins it all.
    reach = np.zeros((T, len(ROUNDS) + 1))
    for i, g in enumerate(bracket.games):
        for s in range(2):
            reach[:, g.round] += np.bincount(sides[:, i, s], minlength=T) / n
    reach[:, len(ROUNDS)] = np.bincount(winners[:, bracket.gidx["R6"]], minlength=T) / n
    # Teams not in the First Four "reach" the round of 64 directly.
    return {"n_sims": n, "games": games, "reach": reach.round(5).tolist()}


# ---------------------------------------------------------------------------
# Scoring a realised bracket
# ---------------------------------------------------------------------------

def pairwise_win_matrix(bracket: Bracket, post: Posterior, n_draws: int = 100, sims_per_draw: int = 100,
                        seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """P(i beats j) on a neutral floor for every pair of tournament teams, per posterior draw.

    Returns (draw_ids, W) with W of shape (n_draws, T, T); W[k, i, j] + W[k, j, i] = 1.
    Estimated by simulating ``sims_per_draw`` full games for each pair and draw.
    """
    rng = np.random.default_rng(seed)
    T = len(bracket.teams)
    draw_ids = rng.choice(post.n_draws, size=min(n_draws, post.n_draws), replace=False)
    pidx = post.team_index(bracket.teams)
    ii, jj = np.triu_indices(T, 1)
    W = np.full((len(draw_ids), T, T), 0.5, dtype=np.float32)
    chunk = 200  # pairs per batch
    for start in range(0, len(ii), chunk):
        pi, pj = ii[start:start + chunk], jj[start:start + chunk]
        P = len(pi)
        k = np.repeat(draw_ids, P * sims_per_draw)
        a = np.tile(np.repeat(pidx[pi], sims_per_draw), len(draw_ids))
        b = np.tile(np.repeat(pidx[pj], sims_per_draw), len(draw_ids))
        res = simulate_games(post, k, a, b, rng, full_box=False)
        wins = res["a_wins"].reshape(len(draw_ids), P, sims_per_draw).sum(-1)
        # Light smoothing keeps big mismatches from being estimated as impossible.
        p = (wins + 0.5) / (sims_per_draw + 1)
        W[:, pi, pj] = p
        W[:, pj, pi] = 1 - p
    return draw_ids, W


def resolve_picks(bracket: Bracket, picks: dict[str, str]) -> list[dict]:
    """Turn {game_id: winner_name} into the games that were actually played.

    A game counts only when both of its participants are determined and the
    picked winner is one of them.
    """
    played = {}
    out = []
    for g in bracket.games:
        teams = []
        for kind, v in g.sources:
            teams.append(v if kind == "team" else played.get(v))
        w = picks.get(g.id)
        if w is None or None in teams or w not in bracket.tidx or bracket.tidx[w] not in teams:
            continue
        wi = bracket.tidx[w]
        played[g.id] = wi
        out.append({"id": g.id, "round": g.round, "teams": teams, "winner": wi,
                    "loser": teams[1] if teams[0] == wi else teams[0]})
    return out


def _log_mean_exp(x, axis=0):
    m = x.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(x - m).mean(axis=axis, keepdims=True))).squeeze(axis)


def score_bracket(bracket: Bracket, picks: dict[str, str], W: np.ndarray, summary: dict,
                  sims: dict | None = None) -> dict:
    """Likelihood of a realised (possibly partial) bracket under the model.

    log P(bracket) = log E_theta[ prod_g P(winner_g beats loser_g | theta) ]
    averaging the *product* over posterior draws (the outcomes of different
    games are correlated through the shared uncertainty about team skill).
    """
    played = resolve_picks(bracket, picks)
    if not played:
        return {"n_games": 0, "games": []}
    w = np.array([p["winner"] for p in played])
    l = np.array([p["loser"] for p in played])
    per_draw = np.log(W[:, w, l])                     # draws x games
    log_p = float(_log_mean_exp(per_draw.sum(1)))
    p_game = W[:, w, l].mean(0)                       # predictive win prob of each realised winner
    games = []
    for p, pg in zip(played, p_game):
        g = bracket.games[bracket.gidx[p["id"]]]
        # Pre-tournament probability that the winner would win this particular game.
        p_pre = next((t["p_win"] for t in summary["games"][bracket.gidx[p["id"]]]["teams"]
                      if t["team"] == p["winner"]), 0.0)
        games.append({"id": p["id"], "round": ROUNDS[g.round], "winner": bracket.teams[p["winner"]],
                      "loser": bracket.teams[p["loser"]], "p_win": float(pg), "p_pretournament": float(p_pre),
                      "surprise_bits": float(-np.log2(pg))})
    n = len(played)
    out = {
        "n_games": n,
        "log_likelihood": log_p,
        "log2_likelihood": log_p / np.log(2),
        "per_game_log_likelihood": log_p / n,
        "coin_flip_log_likelihood": n * np.log(0.5),
        "naive_product_log_likelihood": float(np.log(p_game).sum()),
        "brier": float(np.mean((1 - p_game) ** 2)),
        "accuracy": float(np.mean(p_game > 0.5)),
        "games": games,
    }
    if sims is not None:
        out["percentile"] = likelihood_percentile(bracket, sims, W, [p["id"] for p in played], log_p)
    return out


def likelihood_percentile(bracket: Bracket, sims: dict, W: np.ndarray, game_ids: list[str], log_p: float,
                          max_sims: int = 4000) -> dict:
    """Where the realised log-likelihood falls among brackets simulated from the model itself.

    If the model is well calibrated the realised bracket should look like a
    typical simulated one (percentile roughly uniform); a very low percentile
    means reality was more chaotic than the model believes.
    """
    gi = [bracket.gidx[g] for g in game_ids]
    win = sims["winners"][:max_sims][:, gi].astype(np.int64)
    sides = sims["sides"][:max_sims][:, gi].astype(np.int64)
    lose = np.where(sides[..., 0] == win, sides[..., 1], sides[..., 0])
    ll = np.empty(len(win))
    for start in range(0, len(win), 500):
        w, l = win[start:start + 500], lose[start:start + 500]
        per = np.log(W[:, w, l]).sum(-1)              # draws x sims
        ll[start:start + 500] = _log_mean_exp(per, axis=0)
    return {"percentile": float((ll < log_p).mean()), "sim_mean": float(ll.mean()),
            "sim_p05": float(np.quantile(ll, 0.05)), "sim_p95": float(np.quantile(ll, 0.95)),
            "hist": np.histogram(ll, bins=30)[0].tolist(),
            "hist_edges": np.histogram(ll, bins=30)[1].round(3).tolist()}


def actual_picks(bracket: Bracket) -> dict[str, str]:
    """Map the real tournament results onto bracket game ids."""
    picks: dict[str, str] = {}
    results = bracket.spec.get("results", [])
    remaining = list(results)
    for _ in range(len(ROUNDS)):
        for g in bracket.games:
            if g.id in picks:
                continue
            teams = []
            for kind, v in g.sources:
                teams.append(bracket.teams[v] if kind == "team" else picks.get(v))
            if None in teams:
                continue
            for r in remaining:
                if set(r["teams"]) == set(teams):
                    picks[g.id] = r["winner"]
                    remaining.remove(r)
                    break
    return picks


def save_simulation(season: int, bracket: Bracket, sims: dict, summary: dict, W=None, draw_ids=None):
    out = artifact_dir(season)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "tournament_sims.npz", **sims)
    (out / "summary.json").write_text(json.dumps(summary))
    if W is not None:
        np.savez_compressed(out / "win_matrix.npz", W=W.astype(np.float16), draw_ids=draw_ids)


def load_simulation(season: int) -> tuple[dict, dict, np.ndarray | None]:
    d = artifact_dir(season)
    sims = dict(np.load(d / "tournament_sims.npz"))
    summary = json.loads((d / "summary.json").read_text())
    W = None
    if (d / "win_matrix.npz").exists():
        W = np.load(d / "win_matrix.npz")["W"].astype(np.float32)
    return sims, summary, W


# ---------------------------------------------------------------------------
# Optimal brackets
# ---------------------------------------------------------------------------

# ESPN Tournament Challenge points per correct pick, by round (First Four unscored).
ESPN_POINTS = [0, 10, 20, 40, 80, 160, 320]
MAX_UPSETS = 32


@dataclass
class Constraints:
    """Restrictions on the bracket search.

    forced:     game id -> team index that must win that game (e.g. a champion,
                or a fixed Final Four).
    n_upsets:   exact number of upsets the bracket must contain (None = any).
    upset_gap:  a win counts as an upset when the winner is seeded at least this
                many lines below the loser (1 = any lower seed; 8-over-9 is not
                an upset, 9-over-8 is).
    """
    forced: dict = field(default_factory=dict)
    n_upsets: int | None = None
    upset_gap: int = 1


def is_upset(bracket: Bracket, winner: int, loser: int, gap: int = 1) -> bool:
    return bracket.seed[bracket.teams[winner]] - bracket.seed[bracket.teams[loser]] >= gap


def _dp(bracket: Bracket, value, c: Constraints = Constraints()) -> tuple[dict[str, int], float, np.ndarray]:
    """Exact dynamic programme over the bracket tree.

    ``value(game)`` returns a T x T matrix: the score of ``w`` beating ``l`` in
    that game.  For every game, every team ``t`` that could win it and every
    upset count ``u``, ``best[g][t, u]`` is the highest total over the whole
    sub-bracket feeding ``g`` given that ``t`` wins ``g`` with ``u`` upsets
    along the way.  A bracket's score is a sum over games and each sub-bracket
    only touches the rest through its winner (and its upset count), so this is
    the global optimum.  In particular ``best["R6"][t]`` is the best bracket
    with champion ``t`` — every champion is tried, and the best one is kept.
    Returns the picks, their score, and that per-champion score vector.
    """
    T = len(bracket.teams)
    U = MAX_UPSETS + 1 if c.n_upsets is not None else 1
    seeds = np.array([bracket.seed[t] for t in bracket.teams])
    upset = (seeds[:, None] - seeds[None, :] >= c.upset_gap) if c.n_upsets is not None \
        else np.zeros((T, T), dtype=bool)
    best, back = {}, {}

    def side(src) -> np.ndarray:
        kind, v = src
        if kind == "team":
            out = np.full((T, U), -np.inf)
            out[v, 0] = 0.0
            return out
        return best[v]

    for g in bracket.games:
        val = value(g)                                            # [w, l]
        sides = [side(g.sources[0]), side(g.sources[1])]
        b = np.full((T, U), -np.inf)
        bk = np.zeros((T, U, 3), dtype=int)                       # opponent, opponent's upsets, own upsets
        for own, other in ((0, 1), (1, 0)):
            mine, theirs = sides[own], sides[other]
            ts = np.flatnonzero(np.isfinite(mine).any(1))
            os_ = np.flatnonzero(np.isfinite(theirs).any(1))
            if not len(ts):
                continue
            # cand[t, o, v]: opponent o's sub-bracket with v upsets, plus this game.
            cand = theirs[os_][None, :, :] + val[np.ix_(ts, os_)][:, :, None]
            up = upset[np.ix_(ts, os_)]
            if U > 1:  # an upset in this game shifts the count by one
                shifted = np.full_like(cand, -np.inf)
                shifted[:, :, 1:] = cand[:, :, :-1]
                cand = np.where(up[:, :, None], shifted, cand)
            oi = cand.argmax(1)                                   # [t, v_total_other]
            cmax = np.take_along_axis(cand, oi[:, None, :], 1)[:, 0, :]
            # max-plus convolution with the winner's own sub-bracket upsets
            tot = mine[ts][:, :, None] + cmax[:, None, :]         # [t, u_own, v]
            for u in range(U):
                uo = np.arange(u + 1)
                vals = tot[:, uo, u - uo]                         # [t, splits]
                k = vals.argmax(1)
                bv = vals[np.arange(len(ts)), k]
                better = bv > b[ts, u]
                tt = ts[better]
                b[tt, u] = bv[better]
                own_u = uo[k[better]]
                o = os_[oi[better, u - own_u]]
                v = u - own_u - upset[tt, o].astype(int)
                bk[tt, u] = np.stack([o, v, own_u], 1)
        if g.id in c.forced:
            keep = np.zeros(T, dtype=bool)
            keep[c.forced[g.id]] = True
            b[~keep] = -np.inf
        best[g.id], back[g.id] = b, bk

    u_final = 0 if U == 1 else c.n_upsets
    root = best["R6"][:, u_final]
    if not np.isfinite(root).any():
        raise ValueError("no bracket satisfies these constraints")
    picks: dict[str, int] = {}
    stack = [("R6", int(root.argmax()), u_final)]
    while stack:
        gid, w, u = stack.pop()
        picks[gid] = w
        g = bracket.games[bracket.gidx[gid]]
        o, v, own_u = back[gid][w, u]
        for kind, src in g.sources:
            if kind == "game":
                mine = w in bracket.possible_teams(src)
                stack.append((src, w, own_u) if mine else (src, int(o), int(v)))
    return picks, float(root.max()), root


def bracket_log_likelihood(bracket: Bracket, picks: dict[str, int], W: np.ndarray) -> float:
    """Exact log P(bracket) = log E_theta prod_g P(winner beats loser | theta)."""
    named = {gid: bracket.teams[t] for gid, t in picks.items()}
    played = resolve_picks(bracket, named)
    w = np.array([p["winner"] for p in played])
    l = np.array([p["loser"] for p in played])
    return float(_log_mean_exp(np.log(W[:, w, l]).sum(1)))


def count_upsets(bracket: Bracket, picks: dict[str, int], gap: int = 1) -> int:
    played = resolve_picks(bracket, {g: bracket.teams[t] for g, t in picks.items()})
    return sum(is_upset(bracket, p["winner"], p["loser"], gap) for p in played)


def favourites(bracket: Bracket, Wm: np.ndarray) -> dict[str, int]:
    """Greedy bracket: advance whichever team is favoured in each game as it comes."""
    winners: dict[str, int] = {}
    for g in bracket.games:
        a, b = (v if kind == "team" else winners[v] for kind, v in g.sources)
        winners[g.id] = a if Wm[a, b] >= 0.5 else b
    return winners


def _satisfies(bracket: Bracket, picks: dict[str, int], c: Constraints) -> bool:
    if any(picks.get(g) != t for g, t in c.forced.items()):
        return False
    return c.n_upsets is None or count_upsets(bracket, picks, c.upset_gap) == c.n_upsets


def _refine(bracket: Bracket, W: np.ndarray, logW: np.ndarray, picks: dict, c: Constraints,
            max_iter: int = 50) -> tuple[dict, float]:
    """Minorize-maximize: reweight posterior draws by the current bracket, re-solve the DP."""
    ll = bracket_log_likelihood(bracket, picks, W)
    for _ in range(max_iter):
        played = resolve_picks(bracket, {g: bracket.teams[t] for g, t in picks.items()})
        wi = np.array([p["winner"] for p in played])
        li = np.array([p["loser"] for p in played])
        per_draw = logW[:, wi, li].sum(1)
        wt = np.exp(per_draw - per_draw.max())
        L = np.tensordot(wt / wt.sum(), logW, axes=1)
        new = _dp(bracket, lambda g, L=L: L, c)[0]
        new_ll = bracket_log_likelihood(bracket, new, W)
        if new == picks or new_ll <= ll + 1e-9:
            break
        picks, ll = new, new_ll
    return picks, ll


def _with_champion(c: Constraints, t: int) -> Constraints:
    return Constraints(forced={**c.forced, "R6": int(t)}, n_upsets=c.n_upsets, upset_gap=c.upset_gap)


def most_likely_bracket(bracket: Bracket, W: np.ndarray, c: Constraints = Constraints(),
                        n_champions: int = 10) -> dict:
    """The most probable complete bracket under the model (subject to ``c``).

    The exact objective log E_k[prod_g W_k(g)] is not a plain sum over games,
    because games share uncertainty about team skill.  With per-game
    probabilities averaged first ("plug-in") it is, and the DP solves that
    exactly.  For the exact objective, a minorize-maximize loop climbs from a
    start: weight each posterior draw by how well it explains the current
    bracket, then re-solve the DP on the weighted log-probabilities; each step
    can only raise the true probability.  That climb is local, so it is
    restarted from:

      * both plug-in optima and the game-by-game favourites, and
      * the best bracket for each of the ``n_champions`` most promising
        champions (ranked by the DP's per-champion scores): optimise with the
        champion held fixed, then release it and climb again.

    The best result wins; ``by_champion`` lists the per-champion candidates.
    """
    logW = np.log(W)
    Lp = np.log(W.mean(0))
    starts = {"mean probability": _dp(bracket, lambda g: Lp, c)[0],
              "mean log-probability": _dp(bracket, lambda g: logW.mean(0), c)[0]}
    fav = favourites(bracket, W.mean(0))
    if _satisfies(bracket, fav, c):
        starts["game-by-game favourites"] = fav
    results = [(*_refine(bracket, W, logW, p, c), name) for name, p in starts.items()]

    root = _dp(bracket, lambda g: Lp, c)[2]
    order = [int(t) for t in np.argsort(-root) if np.isfinite(root[t])][:n_champions]
    by_champion = []
    for t in order:
        cc = _with_champion(c, t)
        cands = [_dp(bracket, lambda g: Lp, cc)[0]] + [r[0] for r in results if r[0]["R6"] == t]
        fixed = max((_refine(bracket, W, logW, p, cc) for p in cands), key=lambda r: r[1])
        by_champion.append({"champion": t, "picks": fixed[0], "log_likelihood": fixed[1]})
        results.append((*_refine(bracket, W, logW, fixed[0], c), f"champion restart: {bracket.teams[t]}"))
    picks, ll, name = max(results, key=lambda r: r[1])
    by_champion.sort(key=lambda r: -r["log_likelihood"])
    return {"picks": picks, "log_likelihood": ll, "start": name, "by_champion": by_champion}


def best_bracket_by_champion(bracket: Bracket, W: np.ndarray, champions, c: Constraints = Constraints()) -> list:
    """The most likely bracket for each of the given champions, best first."""
    out = []
    for t in champions:
        try:
            res = most_likely_bracket(bracket, W, _with_champion(c, t), n_champions=1)
        except ValueError:
            continue
        out.append({"champion": int(t), "picks": res["picks"], "log_likelihood": res["log_likelihood"]})
    return sorted(out, key=lambda r: -r["log_likelihood"])


def _win_probabilities(bracket: Bracket, summary: dict) -> np.ndarray:
    pwin = np.zeros((len(bracket.games), len(bracket.teams)))
    for i, g in enumerate(summary["games"]):
        for t in g["teams"]:
            pwin[i, t["team"]] = t["p_win"]
    return pwin


def max_expected_points_bracket(bracket: Bracket, summary: dict, c: Constraints = Constraints(),
                                points=ESPN_POINTS) -> dict:
    """The bracket with the highest expected pool score (ESPN scoring by default).

    A pick scores if the picked team really wins that game, so the expected
    score is the sum over games of points x P(picked team wins that game),
    using the simulated marginal probabilities.  The DP enforces that a team
    picked to win a game is also picked to win every earlier game on its path.
    """
    pwin = _win_probabilities(bracket, summary)
    T = len(bracket.teams)

    def value(g):
        return np.repeat((points[g.round] * pwin[bracket.gidx[g.id]])[:, None], T, axis=1)

    picks, total, _ = _dp(bracket, value, c)
    return {"picks": picks, "expected_points": total}


def expected_points(bracket: Bracket, picks: dict[str, str], summary: dict, points=ESPN_POINTS) -> float:
    """Expected pool score of a set of pre-tournament picks."""
    pwin = _win_probabilities(bracket, summary)
    total = 0.0
    for gid, name in picks.items():
        if gid in bracket.gidx and name in bracket.tidx:
            i = bracket.gidx[gid]
            total += points[bracket.games[i].round] * pwin[i, bracket.tidx[name]]
    return total


def upset_distribution(bracket: Bracket, sims: dict, gap: int = 1) -> dict:
    """How many upsets a simulated tournament contains (a guide to a 'realistic' count)."""
    seeds = np.array([bracket.seed[t] for t in bracket.teams])
    w = sims["winners"].astype(int)
    sides = sims["sides"].astype(int)
    lose = np.where(sides[..., 0] == w, sides[..., 1], sides[..., 0])
    n = (seeds[w] - seeds[lose] >= gap).sum(1)
    return {"mean": float(n.mean()), "median": int(np.median(n)), "p10": int(np.quantile(n, 0.1)),
            "p90": int(np.quantile(n, 0.9)), "hist": np.bincount(n, minlength=MAX_UPSETS + 1)[:MAX_UPSETS + 1].tolist()}


def espn_points(bracket: Bracket, picks: dict[str, str], sims: dict, actual: dict[str, str],
                points=ESPN_POINTS) -> dict:
    """Predicted vs realised pool score of a set of picks.

    A pick for a game scores ``points[round]`` when the picked team wins that
    game.  The prediction is the distribution of that score over the simulated
    tournaments (its mean is the expected score); the realisation scores the
    same picks against the actual results.
    """
    gi = [bracket.gidx[g] for g in picks if g in bracket.gidx and picks[g] in bracket.tidx]
    pick_idx = np.array([bracket.tidx[picks[bracket.games[i].id]] for i in gi])
    pts = np.array([points[bracket.games[i].round] for i in gi], dtype=float)
    rounds = np.array([bracket.games[i].round for i in gi])
    hit = sims["winners"][:, gi] == pick_idx                    # sims x picked games
    sim_scores = hit @ pts
    real_hit = np.array([actual.get(bracket.games[i].id) == picks[bracket.games[i].id] for i in gi], dtype=bool)
    realized = float(real_hit @ pts) if len(gi) else 0.0
    by_round = []
    for r in range(1, len(ROUNDS)):
        m = rounds == r
        if m.any():
            by_round.append({"round": ROUNDS[r], "picks": int(m.sum()), "points_each": points[r],
                             "expected_correct": float(hit[:, m].sum(1).mean()),
                             "realized_correct": int(real_hit[m].sum()),
                             "expected_points": float((hit[:, m] @ pts[m]).mean()),
                             "realized_points": float(real_hit[m] @ pts[m])})
    return {"expected": float(sim_scores.mean()), "sd": float(sim_scores.std()),
            "p10": float(np.quantile(sim_scores, 0.1)), "p90": float(np.quantile(sim_scores, 0.9)),
            "realized": realized, "delta": realized - float(sim_scores.mean()),
            "percentile": float((sim_scores < realized).mean() + 0.5 * (sim_scores == realized).mean()),
            "by_round": by_round,
            "hist": np.histogram(sim_scores, bins=np.arange(0, 1930, 40))[0].tolist(), "bin": 40,
            "max": int(pts.sum())}
