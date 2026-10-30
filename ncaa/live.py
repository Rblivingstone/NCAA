"""Live game data: ESPN's public JSON feeds, and a replay feed for testing.

ESPN's site API (unofficial, but what most public trackers use) serves

    scoreboard:  .../mens-college-basketball/scoreboard?dates=YYYYMMDD&groups=50&limit=500
    game:        .../mens-college-basketball/summary?event=<game id>

The summary carries the header (status, period, clock, score), team box
scores, and play-by-play.  The sportsdataverse mirror archives the same JSON
for finished games (with plays flattened to dotted keys), which the
``ReplayFeed`` uses to step through a real game as if it were live.

Everything is reduced to a ``LiveState``: who is playing, how much of the game
is left, and each team's box score so far.
"""
from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
RAW_GAME_JSON = "https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-raw/main/mbb/json/final/{game_id}.json"
STATS = ["pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "tov", "ast", "stl", "blk", "pf"]
HALF, OT = 20.0, 5.0


@dataclass
class LiveState:
    game_id: int
    teams: list[str]                 # [home, away] by ESPN location name (matches the model's team names)
    team_ids: list[str]
    neutral: bool
    status: str                      # "pre", "in" or "post"
    period: int                      # 1-2 halves, 3+ overtimes
    clock: float                     # minutes left in the period
    box: dict = field(default_factory=dict)   # team name -> {stat: count}
    espn_home_wp: float | None = None
    detail: str = ""
    possession: str | None = None    # team name with the ball, inferred from the last plays

    @property
    def elapsed(self) -> float:
        """Minutes played so far (regulation 40, plus 5 per overtime)."""
        if self.status == "pre":
            return 0.0
        if self.period <= 2:
            return (self.period - 1) * HALF + (HALF - self.clock)
        return 2 * HALF + (self.period - 3) * OT + (OT - self.clock)

    @property
    def remaining(self) -> float:
        """Minutes left, counting only regulation or the current overtime."""
        if self.status == "post":
            return 0.0
        if self.period <= 2:
            return (2 - self.period) * HALF + self.clock
        return self.clock

    def to_json(self) -> dict:
        return {"game_id": self.game_id, "teams": self.teams, "neutral": self.neutral, "status": self.status,
                "period": self.period, "clock": self.clock, "elapsed": self.elapsed, "remaining": self.remaining,
                "box": self.box, "espn_home_wp": self.espn_home_wp, "detail": self.detail,
                "possession": self.possession}


def flatten(d: dict, prefix: str = "") -> dict:
    """{"type": {"text": x}} -> {"type.text": x}; already-flat dicts pass through."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _clock_minutes(text) -> float:
    if text in (None, ""):
        return 0.0
    m = re.match(r"(\d+):(\d+(?:\.\d+)?)", str(text))
    return int(m.group(1)) + float(m.group(2)) / 60 if m else float(text) / 60


def play_elapsed(p: dict) -> float:
    period = int(p.get("period.number") or 1)
    left = _clock_minutes(p.get("clock.displayValue"))
    return (period - 1) * HALF + (HALF - left) if period <= 2 else 2 * HALF + (period - 3) * OT + (OT - left)


def box_from_plays(plays: list[dict], team_ids: list[str], until: float | None = None) -> dict:
    """Each team's box score accumulated from play-by-play (optionally only up to ``until`` minutes)."""
    box = {tid: dict.fromkeys(STATS, 0) for tid in team_ids}
    for raw in plays:
        p = flatten(raw)
        if until is not None and play_elapsed(p) > until + 1e-9:
            break
        tid = str(p.get("team.id")) if p.get("team.id") is not None else None
        if tid not in box:
            continue
        b, kind = box[tid], str(p.get("type.text", ""))
        made = bool(p.get("scoringPlay"))
        if p.get("shootingPlay"):
            if "FreeThrow" in kind:
                b["fta"] += 1
                b["ftm"] += made
                b["pts"] += made
            else:
                three = int(p.get("pointsAttempted") or p.get("scoreValue") or 2) == 3
                b["fga"] += 1
                b["fg3a"] += three
                if made:
                    b["fgm"] += 1
                    b["fg3m"] += three
                    b["pts"] += 3 if three else 2
                    text = str(p.get("text", "")).lower()
                    b["ast"] += "assist" in text  # "(Name assists)" or "assisted by Name"
        elif kind == "Offensive Rebound":
            b["oreb"] += 1
        elif kind == "Defensive Rebound":
            b["dreb"] += 1
        elif "Turnover" in kind:
            b["tov"] += 1
        elif kind == "Steal":
            b["stl"] += 1
        elif kind == "Block Shot":
            b["blk"] += 1
        elif "Foul" in kind and "Technical" not in kind:
            b["pf"] += 1
    return box


def ball_after(plays: list[dict], team_ids: list[str], until: float | None = None) -> str | None:
    """Which team has the ball after the plays so far (None if unclear).

    A made field goal, a defensive rebound, or a turnover hands the ball to the
    other team; an offensive rebound keeps it; free throws hand it over after
    the last one of a trip if it is made.
    """
    ball = None
    for raw in plays:
        p = flatten(raw)
        if until is not None and play_elapsed(p) > until + 1e-9:
            break
        tid = str(p.get("team.id")) if p.get("team.id") is not None else None
        if tid not in team_ids:
            continue
        other = team_ids[1] if tid == team_ids[0] else team_ids[0]
        kind = str(p.get("type.text", ""))
        text = str(p.get("text", "")).lower()
        if p.get("shootingPlay") and "FreeThrow" in kind:
            last = re.search(r"(\d) of (\d)", text)
            if last and last.group(1) == last.group(2):
                ball = other if p.get("scoringPlay") else None
        elif p.get("shootingPlay"):
            ball = other if p.get("scoringPlay") else None
        elif kind == "Defensive Rebound" or kind == "Offensive Rebound":
            ball = tid
        elif "Turnover" in kind:
            ball = other
    return ball


_BOX_FIELDS = {"fieldGoalsMade-fieldGoalsAttempted": ("fgm", "fga"),
               "threePointFieldGoalsMade-threePointFieldGoalsAttempted": ("fg3m", "fg3a"),
               "freeThrowsMade-freeThrowsAttempted": ("ftm", "fta"),
               "offensiveRebounds": "oreb", "defensiveRebounds": "dreb", "assists": "ast", "steals": "stl",
               "blocks": "blk", "fouls": "pf", "turnovers": "tov"}


def _box_from_teams(teams: list[dict]) -> dict:
    out = {}
    for t in teams:
        b = dict.fromkeys(STATS, 0)
        for s in t.get("statistics", []):
            f = _BOX_FIELDS.get(s.get("name"))
            v = s.get("displayValue", "")
            if isinstance(f, tuple) and "-" in str(v):
                b[f[0]], b[f[1]] = (int(x) for x in str(v).split("-"))
            elif isinstance(f, str) and str(v).lstrip("-").isdigit():
                b[f] = int(v)
        b["pts"] = 2 * (b["fgm"] - b["fg3m"]) + 3 * b["fg3m"] + b["ftm"]
        out[str(t["team"]["id"])] = b
    return out


def parse_summary(d: dict, until: float | None = None) -> LiveState:
    """An ESPN game summary (live API or archived) as a LiveState.

    With ``until`` the state is rebuilt from the plays as of that many minutes
    into the game, which is how a finished game is replayed.
    """
    comp = d["header"]["competitions"][0]
    order = sorted(comp["competitors"], key=lambda c: c["homeAway"] != "home")   # home first
    ids = [str(c["id"]) for c in order]
    names = [c["team"]["location"] for c in order]
    status = comp.get("status", {})
    st = status.get("type", {})
    state = st.get("state", "pre")
    period = int(status.get("period") or 1)
    clock = _clock_minutes(status.get("displayClock")) if "displayClock" in status else 0.0
    plays = d.get("plays") or []
    if until is not None:
        flat = [flatten(p) for p in plays]
        last = max((play_elapsed(p) for p in flat), default=40.0)
        until = min(until, last)
        box = box_from_plays(plays, ids, until)
        state = "post" if until >= last - 1e-9 else ("pre" if until <= 0 else "in")
        period = 1 if until <= HALF else 2 if until <= 2 * HALF else 3 + int((until - 2 * HALF - 1e-9) // OT)
        clock = (HALF * min(period, 2) - until if period <= 2 else OT - (until - 2 * HALF - (period - 3) * OT))
        wp = _espn_wp_at(d, flat, until)
        ball = ball_after(plays, ids, until)
    else:
        box = _box_from_teams(d.get("boxscore", {}).get("teams", [])) if state != "pre" else {}
        if state != "pre" and (not box or not any(v["fga"] for v in box.values())) and plays:
            box = box_from_plays(plays, ids)
        wps = d.get("winprobability") or []
        wp = wps[-1].get("homeWinPercentage") if wps else None
        ball = ball_after(plays, ids) if plays and state == "in" else None
    box = {names[i]: box.get(ids[i], dict.fromkeys(STATS, 0)) for i in range(2)}
    neutral = bool((d.get("teamInfo") or {}).get("neutralSite") or comp.get("neutralSite"))
    return LiveState(game_id=int(d["header"]["id"]), teams=names, team_ids=ids, neutral=neutral, status=state,
                     period=period, clock=max(clock, 0.0), box=box, espn_home_wp=wp,
                     detail=st.get("shortDetail", "") if until is None else _detail(state, period, clock),
                     possession=names[ids.index(ball)] if ball in ids else None)


def _detail(state, period, clock) -> str:
    if state == "post":
        return "Final"
    if state == "pre":
        return "Scheduled"
    name = {1: "1st", 2: "2nd"}.get(period, f"OT{period - 2 if period > 3 else ''}")
    return f"{int(clock)}:{int(round(clock % 1 * 60)):02d} - {name}"


def _espn_wp_at(d: dict, flat: list[dict], until: float):
    wps = {str(w.get("playId")): w.get("homeWinPercentage") for w in d.get("winprobability") or []}
    best = None
    for p in flat:
        if play_elapsed(p) > until + 1e-9:
            break
        best = wps.get(str(p.get("id")), best)
    return best


def _get_json(url: str, timeout: float = 20) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (ncaa-bayes)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class ESPNFeed:
    """Today's games and live game states from ESPN's public site API."""

    def __init__(self, base: str = ESPN_BASE):
        self.base = base

    def scoreboard(self, date: str) -> list[dict]:
        """Games on ``date`` (YYYYMMDD): id, teams (home first), status."""
        d = _get_json(f"{self.base}/scoreboard?dates={date}&groups=50&limit=500")
        out = []
        for ev in d.get("events", []):
            comp = ev["competitions"][0]
            order = sorted(comp["competitors"], key=lambda c: c["homeAway"] != "home")
            out.append({"game_id": int(ev["id"]), "teams": [c["team"]["location"] for c in order],
                        "score": [int(c.get("score") or 0) for c in order], "neutral": bool(comp.get("neutralSite")),
                        "status": comp["status"]["type"]["state"], "detail": comp["status"]["type"].get("shortDetail", ""),
                        "start": ev.get("date")})
        return out

    def game(self, game_id: int) -> LiveState:
        return parse_summary(_get_json(f"{self.base}/summary?event={game_id}"))


class ReplayFeed:
    """Replays a finished game from the archived ESPN JSON, at any point in the game."""

    def __init__(self, cache: dict | None = None):
        self.cache = {} if cache is None else cache

    def raw(self, game_id: int) -> dict:
        if game_id not in self.cache:
            self.cache[game_id] = _get_json(RAW_GAME_JSON.format(game_id=game_id), timeout=60)
        return self.cache[game_id]

    def game(self, game_id: int, minute: float) -> LiveState:
        return parse_summary(self.raw(game_id), until=minute)

    def length(self, game_id: int) -> float:
        flat = [flatten(p) for p in self.raw(game_id).get("plays") or []]
        return max((play_elapsed(p) for p in flat), default=40.0)
