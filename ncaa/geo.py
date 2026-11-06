"""Geography for the bracket: where every team plays, and where the tournament is played.

``python -m ncaa geo --season 2027`` builds three files in ``data/<season>/``:

* ``geo_teams.csv``: every Division I team's home arena, city, state and coordinates. The arena is the one the
  team plays most of its true home games in (from the season's schedule); the coordinates come from a public
  table of US cities, with a few hand-entered places it lacks;
* ``geo_venues.csv``: how many regular-season games each team plays (played and scheduled) at each arena, for the
  rule that no team plays a tournament game at an arena where it played more than three games;
* ``sites.json``: the tournament's first/second-round sites (with which days they play: Thursday-Saturday or
  Friday-Sunday) and the four regional sites with their region names. For a finished season these are read
  from the tournament's own games; otherwise from the template below, or left as already written so hand edits
  survive (``--force`` rebuilds it). Edit it when sites are announced or corrected.
"""
from __future__ import annotations

import io
import json
import re
import urllib.request

import pandas as pd

from . import data

CITIES_URL = "https://raw.githubusercontent.com/kelvins/US-Cities-Database/main/csv/us_cities.csv"
GEO_DIR = data.DATA_DIR / "geo"
MAX_GAMES_AT_SITE = 3

# Places the city table lacks (city, state) -> (lat, lon).
PLACES = {
    ("ashwaubenon", "WI"): (44.482, -88.070), ("riverdale", "NY"): (40.890, -73.906),
    ("coral gables", "FL"): (25.721, -80.268), ("highland heights", "KY"): (39.033, -84.452),
    ("lawrenceville", "NJ"): (40.297, -74.729), ("moon township", "PA"): (40.520, -80.219),
    ("homewood", "AL"): (33.472, -86.801), ("storrs", "CT"): (41.808, -72.250),
    ("storrs mansfield", "CT"): (41.808, -72.250), ("queens", "NY"): (40.728, -73.794),
    ("bronx", "NY"): (40.845, -73.865), ("brooklyn", "NY"): (40.650, -73.950),
}

# Announced sites for seasons not yet played. ``days`` is "thu-sat", "fri-sun" or null when not yet known.
TEMPLATES = {
    2027: {
        "source": "NCAA site announcements (October 2024; field expanded to 76 in May 2026). Days marked null "
                  "were not announced when this was written; set them when the schedule is out.",
        "pods": [
            {"id": "charlotte", "city": "Charlotte", "state": "NC", "venue": "Spectrum Center", "days": None},
            {"id": "minneapolis", "city": "Minneapolis", "state": "MN", "venue": "Target Center", "days": None},
            {"id": "omaha", "city": "Omaha", "state": "NE", "venue": "CHI Health Center", "days": "thu-sat"},
            {"id": "fort-worth", "city": "Fort Worth", "state": "TX", "venue": "Dickies Arena", "days": None},
            {"id": "louisville", "city": "Louisville", "state": "KY", "venue": "KFC Yum! Center", "days": None},
            {"id": "pittsburgh", "city": "Pittsburgh", "state": "PA", "venue": "PPG Paints Arena", "days": "thu-sat"},
            {"id": "sacramento", "city": "Sacramento", "state": "CA", "venue": "Golden 1 Center", "days": None},
            {"id": "spokane", "city": "Spokane", "state": "WA", "venue": "Spokane Veterans Memorial Arena", "days": None},
        ],
        "regionals": [
            {"id": "east", "region": "East", "city": "New York", "state": "NY", "venue": "Madison Square Garden", "days": None},
            {"id": "midwest", "region": "Midwest", "city": "Kansas City", "state": "MO", "venue": "T-Mobile Center", "days": "fri-sun"},
            {"id": "south", "region": "South", "city": "San Antonio", "state": "TX", "venue": None, "days": None},
            {"id": "west", "region": "West", "city": "Los Angeles", "state": "CA", "venue": None, "days": None},
        ],
        "final_four": {"city": "Detroit", "state": "MI", "venue": "Ford Field"},
    },
}


def _norm_city(s) -> str:
    s = str(s or "").lower().strip()
    s = re.sub(r"\bst\.?\s", "saint ", s)
    s = re.sub(r"\bft\.?\s", "fort ", s)
    return " ".join(re.sub(r"[^a-z ]", " ", s).split())


def norm_venue(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def cities() -> dict:
    """(normalised city, state) -> (lat, lon), from the public US cities table (downloaded once)."""
    path = GEO_DIR / "us_cities.csv"
    if not path.exists():
        GEO_DIR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(CITIES_URL, timeout=120) as r:
            path.write_bytes(r.read())
    c = pd.read_csv(path)
    out = {}
    for city, st, lat, lon in zip(c.CITY, c.STATE_CODE, c.LATITUDE, c.LONGITUDE):
        out.setdefault((_norm_city(city), st), (float(lat), float(lon)))
    out.update({(_norm_city(k[0]), k[1]): v for k, v in PLACES.items()})
    return out


def locate(table: dict, city, state) -> tuple[float, float] | None:
    return table.get((_norm_city(city), state)) or table.get((_norm_city(city).replace(" ", ""), state))


def team_locations(raw: pd.DataFrame, table: dict) -> pd.DataFrame:
    """Home arena and coordinates of every team with a conference."""
    h = raw[(raw.neutral_site != True) & raw.home_conference_id.notna() & raw.venue_full_name.notna()]  # noqa: E712
    rows = []
    for team, g in h.groupby("home_location"):
        top = g.groupby(["venue_full_name", "venue_address_city", "venue_address_state"]).size().idxmax()
        ll = locate(table, top[1], top[2])
        rows.append({"team": team, "venue": top[0], "city": top[1], "state": top[2],
                     "lat": ll[0] if ll else None, "lon": ll[1] if ll else None})
    return pd.DataFrame(rows)


def venue_counts(raw: pd.DataFrame) -> pd.DataFrame:
    """Regular-season games (no tournaments of any kind) each team plays at each arena, played and scheduled."""
    r = raw[raw.tournament_id.isna()] if "tournament_id" in raw else raw     # conference tournaments don't count
    r = r[r.venue_full_name.notna()]
    both = pd.concat([r[["home_location", "venue_full_name", "venue_address_city", "venue_address_state"]]
                      .set_axis(["team", "venue", "city", "state"], axis=1),
                      r[["away_location", "venue_full_name", "venue_address_city", "venue_address_state"]]
                      .set_axis(["team", "venue", "city", "state"], axis=1)])
    return both.groupby(["team", "venue", "city", "state"]).size().rename("games").reset_index()


def _days(dates: pd.Series) -> str | None:
    wd = set((pd.to_datetime(dates, utc=True) - pd.Timedelta(hours=5)).dt.dayofweek)   # US Eastern, roughly
    return "fri-sun" if wd & {4, 6} else "thu-sat" if wd else None


def sites_from_schedule(raw: pd.DataFrame) -> dict | None:
    """A finished (or seeded) tournament's sites, read from its games."""
    t = raw[(raw.tournament_id == data.NCAA_TOURNAMENT_ID)
            & raw.notes_headline.fillna("").str.contains("Men's Basketball Championship")].copy()
    if not len(t):
        return None
    t["rnd"] = t.notes_headline.str.split(" - ").str[-1]
    t["region"] = t.notes_headline.str.extract(r"- (\w+) Region -", expand=False)
    pods, regionals = [], []
    for (venue, city, st), g in t[t.rnd.isin(["1st Round", "2nd Round"])].groupby(
            ["venue_full_name", "venue_address_city", "venue_address_state"]):
        pods.append({"id": re.sub(r"[^a-z]+", "-", city.lower()).strip("-"), "city": city, "state": st,
                     "venue": venue, "days": _days(g.game_date_time)})
    for (venue, city, st), g in t[t.rnd.isin(["Sweet 16", "Elite 8"])].groupby(
            ["venue_full_name", "venue_address_city", "venue_address_state"]):
        reg = g.region.dropna().mode()
        regionals.append({"id": str(reg.iat[0]).lower() if len(reg) else city.lower(), "region": reg.iat[0] if len(reg) else city,
                          "city": city, "state": st, "venue": venue, "days": _days(g.game_date_time)})
    if len(pods) != 8 or len(regionals) != 4:
        return None
    return {"source": "read from the tournament's games", "pods": pods, "regionals": regionals}


def build(season: int, force: bool = False, raw: pd.DataFrame | None = None, log=print) -> dict:
    """Write geo_teams.csv, geo_venues.csv and (unless it exists and ``force`` is off) sites.json."""
    if raw is None:
        raw = data._fetch_parquet(f"{data.BASE}/schedules/parquet/mbb_schedule_{season}.parquet")
    table = cities()
    out = data.season_dir(season)
    out.mkdir(parents=True, exist_ok=True)
    teams = team_locations(raw, table)
    teams.to_csv(out / "geo_teams.csv", index=False)
    venue_counts(raw).to_csv(out / "geo_venues.csv", index=False)
    missing = teams[teams.lat.isna()]
    log(f"geo: {len(teams)} teams located, {len(missing)} without coordinates"
        + (": " + ", ".join(f"{r.team} ({r.city}, {r.state})" for r in missing.itertuples()) if len(missing) else ""))
    sp = out / "sites.json"
    sites = sites_from_schedule(raw)
    if sites is None and (force or not sp.exists()) and season in TEMPLATES:
        sites = json.loads(json.dumps(TEMPLATES[season]))
    if sites is not None and (force or not sp.exists() or sites.get("source", "").startswith("read")):
        for s in sites["pods"] + sites["regionals"] + ([sites["final_four"]] if "final_four" in sites else []):
            ll = locate(table, s["city"], s["state"])
            s["lat"], s["lon"] = (ll if ll else (None, None))
        sp.write_text(json.dumps(sites, indent=1))
        log(f"geo: wrote {len(sites['pods'])} first/second-round sites and {len(sites['regionals'])} regionals")
    elif not sp.exists():
        log(f"geo: no tournament sites known for {season}; the bracket will be placed without geography")
    return {"teams": len(teams), "missing": len(missing), "sites": sp.exists()}


def load(season: int) -> dict | None:
    """Teams' coordinates, the sites, and which sites each team may not play at; None without the files."""
    d = data.season_dir(season)
    tp, sp, vp = d / "geo_teams.csv", d / "sites.json", d / "geo_venues.csv"
    if not (tp.exists() and sp.exists()):
        return None
    sites = json.loads(sp.read_text())
    if len(sites.get("pods", [])) != 8 or len(sites.get("regionals", [])) != 4:
        return None
    teams = pd.read_csv(tp)
    loc = {r.team: (float(r.lat), float(r.lon)) for r in teams.itertuples() if r.lat == r.lat}
    blocked: dict = {}
    if vp.exists():
        v = pd.read_csv(vp)
        v = v[v.games > MAX_GAMES_AT_SITE]
        for s in sites["pods"] + sites["regionals"]:
            if not s.get("venue"):
                continue
            key = norm_venue(s["venue"])
            hit = v[(v.state == s["state"]) & v.venue.map(lambda x: _same_venue(norm_venue(x), key))]
            for t in hit.team:
                blocked.setdefault(t, set()).add(s["id"])
        for s in sites["pods"] + sites["regionals"]:
            if s.get("host"):
                blocked.setdefault(s["host"], set()).add(s["id"])
    return {"sites": sites, "loc": loc, "blocked": blocked}


def _same_venue(a: str, b: str) -> bool:
    return bool(a) and bool(b) and (a == b or a in b or b in a)
