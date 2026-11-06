"""Road to March: the rest of the season, simulated.

Each simulation draws one sample of every team's skills from the current posterior (so the uncertainty in the
ratings carries through) and then plays, in order:

1. **the rest of the regular season**: every scheduled game not yet played, with home court;
2. **conference tournaments**: every conference's members seeded by simulated conference record, single
   elimination on a neutral floor, top seeds getting byes up to the next power of two (the Ivy League takes its
   top four). Real formats vary a little; this is the common shape;
3. **selection**: each conference tournament champion gets the automatic bid; the remaining at-large places go
   to the best other teams by a résumé score that weighs the team's strength in that simulation and its
   simulated winning percentage equally (a stand-in for the committee, which can't be modelled exactly);
4. **the NCAA tournament**: 68 teams, seeded 1-16 down an S-curve by the same score, four First Four games (the
   four lowest automatic bids and the last four at-large teams), then six rounds on neutral floors.

Every game is a full simulated game from the model, the same simulator the rest of the app uses; only the
winner is kept. The result is, for every Division I team: projected record, conference titles, chance of making
the tournament, seed distribution, and the chance of reaching each round up to the championship.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import data, model
from .simulate import expected_efficiency, simulate_games

ROUNDS = ["First Four", "Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Final", "Champion"]
FIELD, AUTO_BIDS_PLAYIN, ATLARGE_PLAYIN = 68, 4, 4
BATCH = 200_000                              # simulated games held in memory at once (~0.4 GB)
SHORT_TOURNEY = {12: 4}                      # conference id -> teams in its tournament (Ivy League: top four)

# ESPN's conference (group) ids; any not listed is shown by number.
CONFERENCES = {1: "America East", 2: "ACC", 3: "Atlantic 10", 4: "Big East", 5: "Big Sky", 6: "Big South",
               7: "Big Ten", 8: "Big 12", 9: "Big West", 10: "CAA", 11: "Conference USA", 12: "Ivy League",
               13: "MAAC", 14: "MAC", 16: "MEAC", 18: "Missouri Valley", 19: "NEC", 20: "Ohio Valley",
               21: "Pac-12", 22: "Patriot", 23: "SEC", 24: "Southern", 25: "Southland", 26: "SWAC",
               27: "Sun Belt", 29: "West Coast", 30: "WAC", 43: "Mountain West", 44: "Mountain West",
               45: "Horizon", 46: "ASUN", 49: "Summit", 62: "American"}

# Positions of seeds 1..16 in a 16-team region, top to bottom: 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15.
REGION_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]


def conference_name(cid) -> str:
    return CONFERENCES.get(int(cid), f"Conference {int(cid)}") if cid == cid else "Independent"


def _bracket_order(size: int) -> list[int]:
    """Seed numbers (1-based) in bracket order for a single-elimination field of ``size`` (a power of two)."""
    order = [1]
    while len(order) < size:
        n = 2 * len(order)
        order = [x for s in order for x in (s, n + 1 - s)]
    return order


class _Sim:
    """Plays batches of games: team indices into the posterior, one posterior draw per simulation."""

    def __init__(self, post, draws, rng):
        self.post, self.draws, self.rng, self.games = post, draws, rng, 0

    def play(self, sim, a, b, home=None) -> np.ndarray:
        """Winner flags (True = ``a``) for games between ``a`` and ``b`` in simulations ``sim``."""
        sim, a, b = np.asarray(sim), np.asarray(a), np.asarray(b)
        h = np.zeros(len(a)) if home is None else np.asarray(home, dtype=float)
        out = np.zeros(len(a), dtype=bool)
        for s in range(0, len(a), BATCH):
            e = slice(s, s + BATCH)
            res = simulate_games(self.post, self.draws[sim[e]], a[e], b[e], self.rng, home=h[e], full_box=False)
            out[e] = res["a_wins"]
        self.games += len(a)
        return out


def _knockout(sim: _Sim, slots: np.ndarray) -> tuple[np.ndarray, list]:
    """Single elimination over ``slots`` (n_sims x size, team index or -1 for a bye), neutral floors.
    Returns the champion per simulation and, per round, the teams still alive after it."""
    n = slots.shape[0]
    alive = []
    while slots.shape[1] > 1:
        a, b = slots[:, 0::2], slots[:, 1::2]
        win = np.where(b < 0, True, False) | np.zeros_like(a, dtype=bool)
        both = (a >= 0) & (b >= 0)
        if both.any():
            r, c = np.nonzero(both)
            win[r, c] = sim.play(r, a[r, c], b[r, c])
        slots = np.where(win, a, b)
        slots = np.where((a < 0) & (b < 0), -1, slots)
        alive.append(slots.copy())
    return slots[:, 0], alive


def _net(post, draws) -> np.ndarray:
    """Net rating (pts/100 vs an average team) of every team in each sampled draw: n_sims x teams."""
    from .model import S, def_index

    di = def_index()
    z = post.z[draws].astype(float)                                   # n x T x D
    mu = post.mu[draws][:, None, :]                                   # n x 1 x S
    off = expected_efficiency((mu + z[..., :S]).reshape(-1, S)).reshape(z.shape[:2])
    zdef = z[..., np.maximum(di, 0)] * (di >= 0)
    dfn = expected_efficiency((mu + zdef).reshape(-1, S)).reshape(z.shape[:2])
    return 100 * (off - dfn)


def project(season: int, post, sched: pd.DataFrame, n_sims: int = 1000, seed: int = 0, log=print) -> dict:
    """Simulate the rest of the season ``n_sims`` times. ``sched`` is the schedule with known results filled in."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    d1 = sorted(data.d1_set(season))
    post = model.with_teams(post, set(sched.home) | set(sched.away), set(d1))
    idx = {t: i for i, t in enumerate(post.teams)}
    d1 = [t for t in d1 if t in idx]
    T = len(d1)
    col = {t: j for j, t in enumerate(d1)}                            # D1 team -> column in the tallies
    conf = {}
    for side in ("home", "away"):
        for t, c in zip(sched[side], sched[f"{side}_conf"]):
            if t in col and c == c:
                conf.setdefault(t, []).append(int(c))
    conf = {t: max(set(v), key=v.count) for t, v in conf.items()}
    reg = sched[~sched.tournament.astype(bool)]
    is_conf = (reg.home_conf == reg.away_conf) & reg.home_conf.notna()

    # --- the season so far
    W = np.zeros(T); L = np.zeros(T); CW = np.zeros(T); CL = np.zeros(T)
    done = reg[reg.completed.astype(bool)]
    for r, c in zip(done.itertuples(), is_conf[reg.completed.astype(bool)]):
        hw = r.home_score > r.away_score
        for t, won in ((r.home, hw), (r.away, not hw)):
            if t in col:
                (W if won else L)[col[t]] += 1
                if c:
                    (CW if won else CL)[col[t]] += 1

    # --- the rest of the regular season
    draws = rng.integers(post.n_draws, size=n_sims)
    sim = _Sim(post, draws, rng)
    left = reg[~reg.completed.astype(bool) & ~reg.no_result.astype(bool)]
    left = left[left.home.isin(idx) & left.away.isin(idx)].sort_values("date")
    G = len(left)
    ga, gb = left.home.map(idx).to_numpy(), left.away.map(idx).to_numpy()
    gh = np.where(left.neutral.astype(bool), 0.0, 1.0)
    gconf = is_conf.reindex(left.index).fillna(False).to_numpy()
    s_idx = np.repeat(np.arange(n_sims), G)
    home_won = sim.play(s_idx, np.tile(ga, n_sims), np.tile(gb, n_sims), np.tile(gh, n_sims)).reshape(n_sims, G)
    wins = np.tile(W, (n_sims, 1)); cwins = np.tile(CW, (n_sims, 1)); cgames = np.tile(CW + CL, (n_sims, 1))
    games = np.tile(W + L, (n_sims, 1))
    for j in range(G):
        for t, won in ((left.home.iloc[j], home_won[:, j]), (left.away.iloc[j], ~home_won[:, j])):
            if t in col:
                c = col[t]
                wins[:, c] += won; games[:, c] += 1
                if gconf[j]:
                    cwins[:, c] += won; cgames[:, c] += 1
    log(f"regular season: {G} games left, simulated {n_sims} times ({time.time() - t0:.0f}s)")

    # --- conference tournaments (and regular-season titles)
    cpct = np.where(cgames > 0, cwins / np.maximum(cgames, 1), 0.0) + rng.random((n_sims, T)) * 1e-6   # ties split at random
    pidx = np.array([idx[t] for t in d1])                              # D1 column -> posterior index
    to_col = {int(p): j for j, p in enumerate(pidx)}                   # posterior index -> D1 column
    by_conf = {}
    for t, c in conf.items():
        by_conf.setdefault(c, []).append(col[t])
    reg_title = np.zeros(T); tourney_title = np.zeros(T)
    auto = {}
    for c, members in by_conf.items():
        members = np.array(members)
        order = members[np.argsort(-cpct[:, members], axis=1)]          # n_sims x m, best first
        np.add.at(reg_title, order[:, 0], 1)
        m = min(len(members), SHORT_TOURNEY.get(c, len(members)))
        size = 1 << max(1, int(np.ceil(np.log2(max(m, 2)))))
        pos = _bracket_order(size)
        slots = np.full((n_sims, size), -1)
        for p, s in enumerate(pos):
            if s <= m:
                slots[:, p] = pidx[order[:, s - 1]]
        champ, _ = _knockout(sim, slots)
        champ_col = np.array([to_col[int(x)] for x in champ])
        np.add.at(tourney_title, champ_col, 1)
        auto[c] = champ_col
    log(f"conference tournaments: {len(by_conf)} conferences ({time.time() - t0:.0f}s)")

    # --- selection and seeding
    net = _net(post, draws)[:, pidx]                                    # n_sims x T
    pct = wins / np.maximum(games, 1)
    zs = lambda x: (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-9)
    score = zs(net) + zs(pct)
    autos = np.stack(list(auto.values()), axis=1) if auto else np.zeros((n_sims, 0), dtype=int)
    n_auto = autos.shape[1]
    n_at = FIELD - n_auto
    make = np.zeros(T); seeds = np.zeros((T, 16)); reach = np.zeros((T, len(ROUNDS)))
    made = np.zeros((n_sims, T), dtype=bool)
    field_rows = []
    for s in range(n_sims):
        is_auto = np.zeros(T, dtype=bool); is_auto[autos[s]] = True
        others = np.flatnonzero(~is_auto)
        atl = others[np.argsort(-score[s, others])][:n_at]
        field = np.r_[autos[s], atl]
        make[field] += 1
        made[s, field] = True
        # First Four: the four lowest automatic bids and the last four at-large teams, paired off.
        low_auto = autos[s][np.argsort(score[s, autos[s]])][:AUTO_BIDS_PLAYIN]
        last_atl = atl[-ATLARGE_PLAYIN:]
        playin = [(low_auto[0], low_auto[1]), (low_auto[2], low_auto[3]), (last_atl[0], last_atl[1]), (last_atl[2], last_atl[3])]
        in_pi = set(np.r_[low_auto, last_atl].tolist())
        entries = [(score[s, t], [t]) for t in field if t not in in_pi] + [(max(score[s, a], score[s, b]), [a, b]) for a, b in playin]
        entries.sort(key=lambda e: -e[0])
        field_rows.append(entries)
    # play the First Four across all simulations at once
    pi_pos = [[e for e, ent in enumerate(ents) if len(ent[1]) == 2] for ents in field_rows]   # where the 4 pairs sit
    pa = np.array([[field_rows[s][e][1][0] for e in pi_pos[s]] for s in range(n_sims)])
    pb = np.array([[field_rows[s][e][1][1] for e in pi_pos[s]] for s in range(n_sims)])
    r, c = np.nonzero(pa >= 0)
    pw = sim.play(r, pidx[pa[r, c]], pidx[pb[r, c]]).reshape(pa.shape)
    slots = np.full((n_sims, 64), -1)
    for s, ents in enumerate(field_rows):
        result = {e: bool(pw[s, i]) for i, e in enumerate(pi_pos[s])}   # entry position -> first team won
        for line in range(16):                                         # S-curve: snake the four regions
            for reg_i in range(4):
                e = 4 * line + (reg_i if line % 2 == 0 else 3 - reg_i)
                _, teams = ents[e]
                if len(teams) == 2:
                    t = teams[0] if result[e] else teams[1]
                    reach[teams, 0] += 1
                else:
                    t = teams[0]
                for tt in teams:
                    seeds[tt, line] += 1
                slots[s, reg_i * 16 + REGION_ORDER.index(line + 1)] = pidx[t]
                reach[t, 1] += 1
    champ, alive = _knockout(sim, slots)
    for rnd, a in enumerate(alive):                                    # alive after round 1 -> reached round 32, ...
        vals, cnt = np.unique(a[a >= 0], return_counts=True)
        for v, n_ in zip(vals, cnt):
            reach[to_col[v], rnd + 2] += n_
    elapsed = time.time() - t0
    log(f"NCAA tournament simulated; {sim.games:,} games in {elapsed:.0f}s")

    # --- leverage of each team's next game on its tournament chances
    nxt = {}
    for j in range(G):
        for t, won in ((left.home.iloc[j], home_won[:, j]), (left.away.iloc[j], ~home_won[:, j])):
            if t in col and t not in nxt:
                nxt[t] = (j, won)
    teams = []
    for t in d1:
        c = col[t]
        w = wins[:, c]
        row = {"team": t, "conference": conference_name(conf.get(t, np.nan)), "conf_id": conf.get(t),
               "wins_now": int(W[c]), "losses_now": int(L[c]), "conf_wins_now": int(CW[c]), "conf_losses_now": int(CL[c]),
               "wins": float(w.mean()), "losses": float((games[:, c] - w).mean()),
               "wins_lo": float(np.percentile(w, 10)), "wins_hi": float(np.percentile(w, 90)),
               "conf_wins": float(cwins[:, c].mean()), "conf_losses": float((cgames[:, c] - cwins[:, c]).mean()),
               "p_conf_regular": float(reg_title[c] / n_sims), "p_conf_tourney": float(tourney_title[c] / n_sims),
               "p_field": float(make[c] / n_sims), "seed_dist": (seeds[c] / n_sims).round(4).tolist(),
               "reach": {ROUNDS[i]: float(reach[c, i] / n_sims) for i in range(len(ROUNDS))}}
        sd = seeds[c]
        row["avg_seed"] = float((sd * np.arange(1, 17)).sum() / sd.sum()) if sd.sum() else None
        if t in nxt:
            j, won = nxt[t]
            g = left.iloc[j]
            inf = np.asarray(won, dtype=bool)
            p_w = made[inf, c].mean() if inf.any() else None
            p_l = made[~inf, c].mean() if (~inf).any() else None
            row["next_game"] = {"game_id": int(g.game_id), "date": str(g.date)[:10], "home": g.home, "away": g.away,
                                "p_win": float(won.mean()), "p_field_if_win": None if p_w is None else float(p_w),
                                "p_field_if_loss": None if p_l is None else float(p_l)}
        teams.append(row)
    return {"season": season, "n_sims": n_sims, "games_left": G, "games_simulated": sim.games, "seconds": round(elapsed, 1),
            "updated": pd.Timestamp.now(tz="UTC").isoformat(), "teams": teams}


# --- storage ------------------------------------------------------------------

def path(season: int):
    return model.artifact_dir(season) / "projection.json"


def history_path(season: int):
    return model.artifact_dir(season) / "projection_history.csv"


def save(season: int, out: dict) -> None:
    import os

    p = path(season)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(out))
    os.replace(tmp, p)
    day = out["updated"][:10]
    rows = pd.DataFrame([{"date": day, "team": r["team"], "p_field": r["p_field"], "p_champion": r["reach"]["Champion"],
                          "p_final_four": r["reach"]["Final Four"], "wins": r["wins"], "avg_seed": r["avg_seed"]}
                         for r in out["teams"]])
    h = history_path(season)
    if h.exists():
        old = pd.read_csv(h)
        rows = pd.concat([old[old.date != day], rows], ignore_index=True)
    rows.to_csv(h, index=False)


def load(season: int) -> dict | None:
    p = path(season)
    return json.loads(p.read_text()) if p.exists() else None


def load_history(season: int) -> pd.DataFrame:
    h = history_path(season)
    return pd.read_csv(h) if h.exists() else pd.DataFrame()


def run(season: int, n_sims: int = 1000, log=print) -> dict:
    """Project the season from the current model and save it (nightly, after the update)."""
    from .season import current_posterior, schedule_with_results

    post, label = current_posterior(season)
    if post is None:
        raise RuntimeError(f"no model for {season} yet")
    out = project(season, post, schedule_with_results(season), n_sims=n_sims, log=log)
    out["posterior"] = label
    save(season, out)
    return out
