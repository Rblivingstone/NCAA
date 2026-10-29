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


def simulate_tournament(bracket: Bracket, post: Posterior, n_sims: int, seed: int = 0) -> dict:
    """Play the whole tournament ``n_sims`` times.

    Each simulated tournament uses a single posterior draw for every game, so
    parameter uncertainty is carried through the bracket coherently.
    """
    rng = np.random.default_rng(seed)
    draws = rng.permutation(np.resize(np.arange(post.n_draws), n_sims))
    pidx = post.team_index(bracket.teams)
    G = len(bracket.games)
    winners = np.full((n_sims, G), -1, dtype=np.int16)
    sides = np.full((n_sims, G, 2), -1, dtype=np.int16)
    scores = np.zeros((n_sims, G, 2), dtype=np.int16)
    for rnd in range(len(ROUNDS)):
        gs = [i for i, g in enumerate(bracket.games) if g.round == rnd]
        if not gs:
            continue
        for i in gs:
            for side, (kind, v) in enumerate(bracket.games[i].sources):
                sides[:, i, side] = v if kind == "team" else winners[:, bracket.gidx[v]]
        top, bot = sides[:, gs, 0].ravel(), sides[:, gs, 1].ravel()
        k = np.repeat(draws, len(gs))
        res = simulate_games(post, k, pidx[top], pidx[bot], rng, full_box=False)
        win = np.where(res["a_wins"], top, bot).reshape(n_sims, len(gs))
        winners[:, gs] = win
        scores[:, gs] = res["pts"].reshape(n_sims, len(gs), 2)
    return {"winners": winners, "sides": sides, "scores": scores, "draws": draws}


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
