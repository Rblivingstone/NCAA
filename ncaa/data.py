"""Download and clean NCAA men's basketball box scores and the tournament bracket.

Source: the sportsdataverse ``hoopR-mbb-data`` mirror of ESPN data, which is
served from raw.githubusercontent.com as parquet files.

Two products are written to ``data/<season>/``:

* ``games.csv``   one row per *team-game* (offense perspective) for every game
                  played before the NCAA tournament began; used to fit the model.
* ``bracket.json`` the 68-team field (seeds, regions, First Four) plus the
                  actual results of every tournament game that has been played.
"""
from __future__ import annotations

import io
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/mbb"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ESPN tournament id for the NCAA men's tournament.
NCAA_TOURNAMENT_ID = 22
# Teams with fewer games than this (non-D1 opponents) are pooled into one team.
MIN_GAMES = 8
OTHER_TEAM = "Non-D1"

# Box-score columns we keep, renamed to short names.
BOX_COLUMNS = {
    "field_goals_made": "fgm",
    "field_goals_attempted": "fga",
    "three_point_field_goals_made": "fg3m",
    "three_point_field_goals_attempted": "fg3a",
    "free_throws_made": "ftm",
    "free_throws_attempted": "fta",
    "offensive_rebounds": "oreb",
    "defensive_rebounds": "dreb",
    "turnovers": "tov",
    "assists": "ast",
    "steals": "stl",
    "blocks": "blk",
    "fouls": "pf",
    "team_score": "pts",
}

ROUND_NAMES = ["First Four", "1st Round", "2nd Round", "Sweet 16", "Elite 8",
               "Final Four", "National Championship"]


def season_dir(season: int) -> Path:
    return DATA_DIR / str(season)


def _fetch_parquet(url: str) -> pd.DataFrame:
    with urllib.request.urlopen(url, timeout=120) as resp:
        return pd.read_parquet(io.BytesIO(resp.read()))


def fetch_raw(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    box = _fetch_parquet(f"{BASE}/team_box/parquet/team_box_{season}.parquet")
    sched = _fetch_parquet(f"{BASE}/schedules/parquet/mbb_schedule_{season}.parquet")
    return box, sched


def _tournament_games(sched: pd.DataFrame) -> pd.DataFrame:
    t = sched[(sched.tournament_id == NCAA_TOURNAMENT_ID)
              & sched.notes_headline.fillna("").str.contains("Men's Basketball Championship")].copy()
    t["round_name"] = t.notes_headline.str.split(" - ").str[-1]
    t["region"] = t.notes_headline.str.extract(r"- (\w+) Region -", expand=False)
    return t


def build_games(box: pd.DataFrame, sched: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Team-game rows for every completed game strictly before ``cutoff``."""
    sched = sched.drop_duplicates("game_id").set_index("game_id")
    box = box[box.game_id.isin(sched.index)].copy()
    box["date"] = pd.to_datetime(box.game_date_time, utc=True)
    box = box[box.date < cutoff]
    # Keep only games where both teams' box scores exist.
    box = box[box.groupby("game_id").game_id.transform("size") == 2]

    keep = ["game_id", "date", "team_id", "team_location", "team_home_away",
            "opponent_team_id"] + list(BOX_COLUMNS)
    df = box[keep].rename(columns=BOX_COLUMNS | {"team_location": "team"})
    opp = df[["game_id", "team_id"] + list(BOX_COLUMNS.values())].rename(
        columns={c: "opp_" + c for c in list(BOX_COLUMNS.values()) + ["team_id"]})
    df = df.merge(opp, left_on=["game_id", "opponent_team_id"],
                  right_on=["game_id", "opp_team_id"]).drop(columns="opp_team_id")
    names = box.drop_duplicates("team_id").set_index("team_id").team_location
    df["opponent"] = df.opponent_team_id.map(names)

    neutral = sched.neutral_site.reindex(df.game_id).to_numpy()
    df["home"] = np.where(neutral, 0, np.where(df.team_home_away == "home", 1, -1))
    periods = sched.status_period.reindex(df.game_id).fillna(2).to_numpy()
    df["ot"] = np.clip(periods - 2, 0, None).astype(int)

    # Drop obviously broken box scores (missing stats recorded as zeros).
    ok = (df.fga > 20) & (df.opp_fga > 20) & (df.fgm <= df.fga) & (df.fg3m <= df.fg3a) \
        & (df.fg3a <= df.fga) & (df.ftm <= df.fta) & (df.pts == 2 * df.fgm + df.fg3m + df.ftm)
    good_games = df[ok].groupby("game_id").size()
    df = df[df.game_id.isin(good_games[good_games == 2].index)]

    # Pool rarely-seen (non-D1) teams into a single pseudo-team.
    counts = df.team.value_counts()
    rare = set(counts[counts < MIN_GAMES].index)
    df.loc[df.team.isin(rare), "team"] = OTHER_TEAM
    df.loc[df.opponent.isin(rare), "opponent"] = OTHER_TEAM
    df = df[~((df.team == OTHER_TEAM) & (df.opponent == OTHER_TEAM))]

    cols = ["game_id", "date", "team", "opponent", "home", "ot"] + list(BOX_COLUMNS.values()) \
        + ["opp_" + c for c in BOX_COLUMNS.values()]
    return df[cols].sort_values(["date", "game_id", "team"]).reset_index(drop=True)


def build_bracket(sched: pd.DataFrame, box: pd.DataFrame) -> dict:
    """Bracket definition (regions, seeds, First Four) and actual results."""
    t = _tournament_games(sched)
    names = box.drop_duplicates("team_id").set_index("team_id").team_location.to_dict()

    def name(tid):
        return names.get(tid) or sched.loc[sched.home_id == tid, "home_location"].iloc[0]

    regions: dict[str, dict[int, list[str]]] = {}
    for _, g in t[t.round_name.isin(["1st Round", "First Four"])].iterrows():
        for side in ("home", "away"):
            seed = int(g[f"{side}_current_rank"])
            team = name(g[f"{side}_id"])
            slot = regions.setdefault(g.region, {}).setdefault(seed, [])
            if team not in slot:
                slot.append(team)
    # A First Four winner also appears in the 1st round; keep the pair only
    # for slots that actually had a First Four game.
    ff_slots = {(g.region, int(g.home_current_rank)) for _, g in t[t.round_name == "First Four"].iterrows()}
    for region, seeds in regions.items():
        for seed, teams in seeds.items():
            if (region, seed) not in ff_slots:
                assert len(teams) == 1, (region, seed, teams)

    # Final Four pairings: the two semifinals each join two regions.
    region_of = {team: r for r, seeds in regions.items() for ts in seeds.values() for team in ts}
    semis = []
    for _, g in t[t.round_name == "Final Four"].sort_values("date").iterrows():
        semis.append(sorted([region_of[name(g.home_id)], region_of[name(g.away_id)]]))
    if len(semis) != 2:  # Tournament not reached yet: fall back to the usual layout.
        order = sorted(regions)
        semis = [order[:2], order[2:]]
    used = [r for pair in semis for r in pair]
    region_order = used  # left column top, left bottom, right top, right bottom

    results = []
    for _, g in t.sort_values("date").iterrows():
        if not bool(g.status_type_completed):
            continue
        home, away = name(g.home_id), name(g.away_id)
        winner = home if g.home_score > g.away_score else away
        results.append({"round": g.round_name, "region": g.region if isinstance(g.region, str) else None,
                        "teams": [home, away], "score": [int(g.home_score), int(g.away_score)],
                        "winner": winner})

    first_game = pd.to_datetime(t.date, utc=True).min()
    return {
        "season": int(sched.season.iloc[0]),
        "start": first_game.isoformat(),
        "regions": region_order,
        "semifinals": semis,
        "seeds": {r: {str(s): ts for s, ts in sorted(regions[r].items())} for r in region_order},
        "results": results,
    }


def fetch(season: int) -> tuple[pd.DataFrame, dict]:
    box, sched = fetch_raw(season)
    bracket = build_bracket(sched, box)
    games = build_games(box, sched, cutoff=pd.Timestamp(bracket["start"]))
    out = season_dir(season)
    out.mkdir(parents=True, exist_ok=True)
    games.to_csv(out / "games.csv", index=False)
    (out / "bracket.json").write_text(json.dumps(bracket, indent=1))
    return games, bracket


def load_games(season: int) -> pd.DataFrame:
    return pd.read_csv(season_dir(season) / "games.csv", parse_dates=["date"])


def load_bracket(season: int) -> dict:
    return json.loads((season_dir(season) / "bracket.json").read_text())


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
