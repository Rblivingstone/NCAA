"""Men's and women's college basketball: everything that differs between the two leagues.

The whole pipeline (data, model fits, live tracking, brackets, projections) runs for one league at a time. The
current league lives in a context variable: a worker process or command line takes it from ``NCAA_LEAGUE``
("mens", the default, or "womens"); the web app sets it per request from the league toggle.

Men's files stay where they always were (``data/<season>``, ``artifacts/<season>``); women's live under a ``wbb``
folder (``data/wbb/<season>``, ``artifacts/wbb/<season>``).

What differs, besides the data source and ESPN's endpoint:

* the clock: men play two 20-minute halves, women four 10-minute quarters (overtimes are 5 minutes in both);
* the tournament's headlines (the women's regions are named "Regional 1 in Albany" rather than "East");
* the bracket rules: from 2027 the women's top 16 are placed by true seed regardless of conference;
* the simulator's calibration, which is fitted per league (the women's runs raw until it has its own).
"""
from __future__ import annotations

import contextvars
import os
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class League:
    key: str
    label: str                      # "Men's" / "Women's"
    data_base: str                  # sportsdataverse mirror of ESPN's data
    file_prefix: str                # "mbb" / "wbb" in the mirror's file names
    raw_game_json: str              # ESPN's archived game files
    espn_base: str                  # ESPN's live site API
    subdir: str                     # folder under data/ and artifacts/ ("" = the top level)
    periods: int                    # regulation periods
    period_minutes: float
    headline: str                   # regex for the NCAA tournament's headlines
    protect_conferences: bool = True  # a conference's top four on seed lines 1-4 in different regions
    blowout_margin: float | None = None  # final margins beyond this count for less (see model.blowout_weight)

    @property
    def half_periods(self) -> int:
        return self.periods // 2


OT_MINUTES = 5.0
# Blowout damper threshold by league, in points of final margin (None = off). See ncaa.model.blowout_weight.
# Women's: on at 25. Replaying the 2026-27 season from preseason, it gave the best forecasts of the later games
# (log loss 0.3197 -> 0.3171; 40 and 15 points did worse) and stopped two 90-100-point wins over the weakest teams
# from making LSU a runaway favourite. Men's: off. Refitting ten tournaments' seasons with it at 25 made the
# ratings more compressed and tournament forecasts worse, even recalibrated (log loss 0.5511 -> 0.5548): men's
# blowouts, mostly against overmatched opponents, still carry real information.
BLOWOUT_MARGIN = {"mens": None, "womens": 25.0}

LEAGUES = {
    "mens": League(
        key="mens", label="Men's",
        data_base="https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/mbb", file_prefix="mbb",
        raw_game_json="https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-raw/main/mbb/json/final/{game_id}.json",
        espn_base="https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball",
        subdir="", periods=2, period_minutes=20.0, headline=r"(?<!wo)men's basketball championship",
        blowout_margin=BLOWOUT_MARGIN["mens"]),
    "womens": League(
        key="womens", label="Women's",
        data_base="https://raw.githubusercontent.com/sportsdataverse/wehoop-wbb-data/main/wbb", file_prefix="wbb",
        raw_game_json="https://raw.githubusercontent.com/sportsdataverse/wehoop-wbb-raw/main/wbb/json/final/{game_id}.json",
        espn_base="https://site.api.espn.com/apis/site/v2/sports/basketball/womens-college-basketball",
        subdir="wbb", periods=4, period_minutes=10.0, headline=r"women's basketball championship",
        protect_conferences=False, blowout_margin=BLOWOUT_MARGIN["womens"]),
}

_current = contextvars.ContextVar("ncaa_league", default=os.environ.get("NCAA_LEAGUE", "mens"))


def current() -> League:
    key = _current.get()
    return LEAGUES.get(key, LEAGUES["mens"])


def key() -> str:
    return current().key


def set_league(k: str):
    """Make ``k`` the current league (until reset with the returned token)."""
    if k not in LEAGUES:
        raise ValueError(f"unknown league {k!r}; choose from {', '.join(LEAGUES)}")
    return _current.set(k)


def reset(token) -> None:
    _current.reset(token)


@contextmanager
def using(k: str):
    token = set_league(k)
    try:
        yield LEAGUES[k]
    finally:
        reset(token)


# --- the clock ---------------------------------------------------------------

def elapsed(period: int, left: float) -> float:
    """Minutes played at ``left`` minutes to go in ``period`` (regulation 40, then 5 per overtime)."""
    lg = current()
    if period <= lg.periods:
        return (period - 1) * lg.period_minutes + (lg.period_minutes - left)
    return lg.periods * lg.period_minutes + (period - lg.periods - 1) * OT_MINUTES + (OT_MINUTES - left)


def remaining(period: int, left: float) -> float:
    """Minutes left in regulation (or in the current overtime)."""
    lg = current()
    if period <= lg.periods:
        return (lg.periods - period) * lg.period_minutes + left
    return left


def period_at(minute: float) -> tuple[int, float]:
    """(period, minutes left in it) ``minute`` minutes into a game."""
    lg = current()
    reg = lg.periods * lg.period_minutes
    if minute <= reg:
        period = max(1, int(-(-minute // lg.period_minutes)))       # ceil, with the tip-off in period 1
        return period, period * lg.period_minutes - minute
    k = int((minute - reg - 1e-9) // OT_MINUTES)
    return lg.periods + 1 + k, OT_MINUTES - (minute - reg - k * OT_MINUTES)


def period_name(period: int) -> str:
    lg = current()
    if period <= lg.periods:
        return {1: "1st", 2: "2nd"}[period] if lg.periods == 2 else f"Q{period}"
    k = period - lg.periods
    return "OT" if k == 1 else f"OT{k}"


def overtimes(periods_played: int) -> int:
    return max(int(periods_played) - current().periods, 0)


def in_regulation(period: int) -> bool:
    return period <= current().periods
