"""Are a team's wins earned? Results against the box-score evidence.

Two things move when a team plays:

* **results beyond expectation**: each game's result minus the pre-game chance the model gave the team (a win
  the model gave 30% counts +0.7; a loss it gave 80% counts -0.8). Their sum is wins above expectation;
* **box-score evidence**: the change in the team's net rating from that game's Bayesian update, which is
  positive when the box score was better than the model expected against that opponent, win or lose.

Winning because a team is better shows up in both. Across all D1 teams, wins above expectation rise with the
rating gained; a least-squares line gives how many extra wins a point of rating normally comes with. A team's
**unexplained wins** are its wins above expectation minus what its rating gain accounts for: large and positive
means it is winning more than its box scores justify (close games going its way, opponents missing shots), the
kind of run that tends not to last; negative means its results lag its play.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CLOSE = 5          # a game decided by this many points or fewer is "close"
STREAK_MIN = 3     # a streak this long gets a verdict
VERDICT_MIN = 5    # games with a pre-game forecast before a season-long lucky/unlucky verdict


def team_games(updates: pd.DataFrame, ledger: pd.DataFrame, team: str) -> pd.DataFrame:
    """One row per game the team has had folded into its rating, oldest first: result, pre-game chance,
    rating change."""
    g = updates[updates.team == team].sort_values(["date", "game_id"]).copy()
    if not len(g):
        return g
    g["won"] = g.pts > g.opp_pts
    g["margin"] = g.pts - g.opp_pts
    g["dnet"] = g.net_after - g.net_before
    p = {}
    if len(ledger):
        for r in ledger.itertuples():
            ph = float(r.p_home)
            p[(int(r.game_id), r.home)] = ph
            p[(int(r.game_id), r.away)] = 1 - ph
    g["p_pre"] = [p.get((int(gid), team), np.nan) for gid in g.game_id]
    g["beyond"] = np.where(g.p_pre.notna(), g.won.astype(float) - g.p_pre, np.nan)
    return g.reset_index(drop=True)


def _streak(g: pd.DataFrame) -> dict:
    """The current run of wins or losses at the end of the team's games."""
    if not len(g):
        return {"kind": None, "length": 0}
    last = bool(g.won.iloc[-1])
    n = 0
    for w in g.won.iloc[::-1]:
        if bool(w) != last:
            break
        n += 1
    s = g.iloc[len(g) - n:]
    return {"kind": "W" if last else "L", "length": n, "rating_change": float(s.dnet.sum()),
            "per_game": float(s.dnet.mean()), "beyond": float(np.nansum(s.beyond)),
            "close": int((s.margin.abs() <= CLOSE).sum()), "start": str(s.date.iloc[0])[:10]}


def summary(g: pd.DataFrame) -> dict:
    """A team's record against its box scores (before the league-wide comparison)."""
    if not len(g):
        return {"games": 0}
    known = g.p_pre.notna()
    wins = int(g.won.sum())
    close = g.margin.abs() <= CLOSE
    return {"games": int(len(g)), "wins": wins, "losses": int(len(g) - wins),
            "expected_wins": float(g.p_pre[known].sum()), "games_with_forecast": int(known.sum()),
            "beyond": float(g.beyond[known].sum()), "rating_change": float(g.dnet.sum()),
            "unearned_wins": int((g.won & (g.dnet < 0)).sum()),       # won, but the box score pulled the rating down
            "close_wins": int((g.won & close).sum()), "close_losses": int((~g.won & close).sum()),
            "streak": _streak(g)}


def league(updates: pd.DataFrame, ledger: pd.DataFrame, teams) -> dict:
    """Every team's summary plus unexplained wins: wins above expectation not accounted for by rating gained,
    from a least-squares line fitted across the teams."""
    rows = {}
    for t in teams:
        s = summary(team_games(updates, ledger, t))
        if s.get("games"):
            rows[t] = s
    pts = [(s["rating_change"], s["beyond"]) for s in rows.values() if s["games_with_forecast"]]
    slope = intercept = 0.0
    if len(pts) >= 10:
        x, y = np.array(pts).T
        slope, intercept = (float(v) for v in np.polyfit(x, y, 1))
    resid = []
    for s in rows.values():
        s["explained"] = intercept + slope * s["rating_change"]
        s["unexplained"] = s["beyond"] - s["explained"]
        resid.append(s["unexplained"])
    sd = float(np.std(resid)) if len(resid) > 1 else 0.0
    order = np.sort(resid)
    for t, s in rows.items():
        # share of the *other* teams with fewer unexplained wins (1.0 = the most of any team)
        s["unexplained_pct"] = float(np.searchsorted(order, s["unexplained"], side="left") / (len(order) - 1)) if len(order) > 1 else None
        s["verdict"] = verdict(s, sd)
    return {"teams": rows, "slope": slope, "intercept": intercept, "resid_sd": sd, "n": len(rows)}


def verdict(s: dict, resid_sd: float) -> dict:
    """A plain reading of whether the winning is backed by the box scores."""
    st = s["streak"]
    hot = st["kind"] == "W" and st["length"] >= STREAK_MIN
    unexplained = s.get("unexplained", 0.0)
    loud = resid_sd > 0 and abs(unexplained) > resid_sd
    if hot and st["per_game"] <= 0:
        level, text = "unearned", (f"Won {st['length']} straight, but the box scores pulled the rating "
                                   f"{st['rating_change']:+.1f} over the run: the wins are running ahead of the play.")
    elif hot and st["per_game"] >= 0.75:
        level, text = "earned", (f"Won {st['length']} straight, and the box scores back it: the rating rose "
                                 f"{st['rating_change']:+.1f} over the run.")
    elif hot:
        level, text = "mixed", (f"Won {st['length']} straight; the box scores only partly back it "
                                f"(rating {st['rating_change']:+.1f} over the run).")
    elif s.get("games_with_forecast", 0) < VERDICT_MIN:
        level, text = "too early", f"Only {s['games']} game{'s' if s['games'] != 1 else ''} so far: too few to tell luck from play."
    elif loud and unexplained > 0:
        level, text = "lucky", (f"{unexplained:+.1f} more wins than the box scores account for this season: "
                                "results have run ahead of the play.")
    elif loud:
        level, text = "unlucky", (f"{-unexplained:.1f} fewer wins than the box scores account for: "
                                  "the play has been better than the record.")
    else:
        level, text = "in line", "Results and box scores agree: the record is about what the play deserves."
    return {"level": level, "text": text}
