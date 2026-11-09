"""A committee-style seed list and bracket, for the Road to March projection.

The selection committee works from team sheets, not from a single rating. This module builds the same kind of
sheet for every team in every simulated season, then selects, seeds and places the field by the committee's
published bracketing principles.

**The sheet** (per simulation, Division I games only, as on the real sheets):

* *strength*: the team's net rating in that simulation (the model's predictive number, the role KenPom, BPI and
  Torvik play on the real sheet);
* *wins above bubble* (WAB): for every game, win (1) or loss (0) minus the chance a bubble-quality team would have
  won that game, at that site. The committee now lists WAB on the sheet and it tracks at-large selection closely;
* a *NET-like ranking*: half strength, half results (WAB), the same blend of efficiency and results the NET uses;
* *quadrant records* by opponent NET rank and game site (Quadrant 1 is home vs 1-30, neutral vs 1-50, away vs 1-75,
  and so on), with Quadrant 1 wins and Quadrant 3-4 losses the two lines the committee talks about most.

The committee score weighs those together (``WEIGHTS``); at-large bids go down it and the seed list follows it.

**The bracket** follows the principles the committee publishes:

* the top four seed lines are "protected": each goes to the regional closest to campus and to the closest open
  first/second-round site, in seed-list order. Two of the four can't be pushed out of balance, so the strongest 2
  seed never joins the overall No. 1 seed's region;
* the top four teams from a conference on those lines go to different regions;
* teams from the same conference that met three or more times can't meet before the regional final; twice, not
  before the regional semifinal; otherwise not before the second round;
* no team plays at an arena where it played more than three games this season;
* BYU (no Sunday play) goes only to Thursday-Saturday sites;
* when no region on a team's line satisfies the rules, the team moves down one line (and the best team on the next
  line moves up), as the committee does;
* the four regions are paired in the Final Four by the overall rank of their No. 1 seeds (1 v 4, 2 v 3).

Without geography (no sites file) the rules that need distance fall back to the S-curve; the conference rules
always apply.
"""
from __future__ import annotations

import itertools

import numpy as np

# Win chance of a team that is ``d`` points per 100 possessions better, at home edge ``h`` (+1/0/-1).
POSS, HFA_POINTS, MARGIN_SD = 68.0, 3.0, 11.0
# Quadrant cut-offs (opponent NET rank) by site, from the team's side: home, neutral, away.
QUAD_CUTS = {1: (30, 75, 160), 0: (50, 100, 200), -1: (75, 135, 240)}
# Committee score weights: on standardised (z-score) values within each simulation.
WEIGHTS = {"wab": 0.50, "net": 0.30, "q1w": 0.10, "q34l": 0.10}
SUNDAY_RESTRICTED = {"BYU"}

# Positions of seeds 1..16 in a 16-team region, top to bottom: 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15.
REGION_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
POS_OF_SEED = {s: REGION_ORDER.index(s) for s in REGION_ORDER}


def _win_prob(diff: np.ndarray, home: np.ndarray) -> np.ndarray:
    x = (diff * POSS / 100 + HFA_POINTS * home) / MARGIN_SD
    return 1 / (1 + np.exp(-1.702 * x))                            # logistic approximation to the normal


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-9)


def _rank(x: np.ndarray) -> np.ndarray:
    """1 = highest value, per row."""
    r = np.empty_like(x, dtype=np.int32)
    order = np.argsort(-x, axis=1)
    np.put_along_axis(r, order, np.arange(1, x.shape[1] + 1, dtype=np.int32)[None].repeat(x.shape[0], 0), axis=1)
    return r


def sheets(eff: np.ndarray, shared: dict, extra: dict, bubble_rank: int, chunk: int = 100) -> dict:
    """Team-sheet numbers for every simulation.

    ``eff``: n_sims x T strength. ``shared``: the games every simulation has (the regular season), with ``a``,
    ``b`` (team columns), ``home`` (+1 when ``a`` is at home, 0 neutral, -1 away) and ``a_won`` (n_sims x G, or
    G for games already played). ``extra``: games particular to each simulation (conference tournaments, on
    neutral floors) with ``sim``, ``a``, ``b``, ``a_won``. Returns n_sims x T arrays."""
    n, T = eff.shape
    a, b, h = shared["a"], shared["b"], shared["home"].astype(float)
    G = len(a)
    A = np.zeros((G, T)); A[np.arange(G), a] = 1                      # game -> team incidence
    B = np.zeros((G, T)); B[np.arange(G), b] = 1
    out = {k: np.zeros((n, T)) for k in ("wab", "q1w", "q1l", "q2w", "q2l", "q34l", "games")}
    bubble = np.sort(eff, axis=1)[:, ::-1][:, bubble_rank - 1:bubble_rank]  # n x 1
    es, ea, eb_, ew = extra["sim"], extra["a"], extra["b"], extra["a_won"].astype(float)

    def results(rows, won_rows):
        """WAB and game counts for simulations ``rows``."""
        e = eff[rows]
        pa = _win_prob(bubble[rows] - e[:, b], h[None])             # bubble team in a's place, vs b
        pb = _win_prob(bubble[rows] - e[:, a], -h[None])
        wab = (won_rows - pa) @ A + ((1 - won_rows) - pb) @ B
        return wab

    won_all = shared["a_won"]
    for s0 in range(0, n, chunk):
        rows = np.arange(s0, min(n, s0 + chunk))
        won = np.broadcast_to(won_all, (n, G))[rows].astype(float)
        out["wab"][rows] = results(rows, won)
        out["games"][rows] = (A + B).sum(0)[None]
    # conference tournaments: neutral floor
    if len(es):
        pa = _win_prob(bubble[es, 0] - eff[es, eb_], 0.0)
        pb = _win_prob(bubble[es, 0] - eff[es, ea], 0.0)
        np.add.at(out["wab"], (es, ea), ew - pa)
        np.add.at(out["wab"], (es, eb_), (1 - ew) - pb)
        np.add.at(out["games"], (es, ea), 1); np.add.at(out["games"], (es, eb_), 1)

    net_score = 0.5 * _z(eff) + 0.5 * _z(out["wab"])
    net_rank = _rank(net_score)

    def quad(opp_rank, home):
        """Quadrant (1-4) of a game against an opponent of ``opp_rank`` at site ``home`` (team's side)."""
        cuts = np.stack([np.choose(home.astype(int) + 1, [QUAD_CUTS[-1][i], QUAD_CUTS[0][i], QUAD_CUTS[1][i]])
                         for i in range(3)])                         # 3 x ...
        return 1 + (opp_rank > cuts[0]) + (opp_rank > cuts[1]) + (opp_rank > cuts[2])

    for s0 in range(0, n, chunk):
        rows = np.arange(s0, min(n, s0 + chunk))
        won = np.broadcast_to(won_all, (n, G))[rows].astype(bool)
        qa = quad(net_rank[rows][:, b], np.broadcast_to(h, (len(rows), G)))   # quadrant of the game for a
        qb = quad(net_rank[rows][:, a], np.broadcast_to(-h, (len(rows), G)))
        for k, q, w in (("q1w", 1, True), ("q1l", 1, False), ("q2w", 2, True), ("q2l", 2, False)):
            out[k][rows] = ((qa == q) & (won == w)).astype(float) @ A + ((qb == q) & (won != w)).astype(float) @ B
        out["q34l"][rows] = ((qa >= 3) & ~won).astype(float) @ A + ((qb >= 3) & won).astype(float) @ B
    if len(es):
        qa = quad(net_rank[es, eb_], np.zeros(len(es))); qb = quad(net_rank[es, ea], np.zeros(len(es)))
        w = ew.astype(bool)
        for k, q, ww in (("q1w", 1, True), ("q1l", 1, False), ("q2w", 2, True), ("q2l", 2, False)):
            np.add.at(out[k], (es, ea), ((qa == q) & (w == ww)).astype(float))
            np.add.at(out[k], (es, eb_), ((qb == q) & (w != ww)).astype(float))
        np.add.at(out["q34l"], (es, ea), ((qa >= 3) & ~w).astype(float))
        np.add.at(out["q34l"], (es, eb_), ((qb >= 3) & w).astype(float))

    out["eff"] = eff
    out["net_rank"] = net_rank
    out["score"] = (WEIGHTS["wab"] * _z(out["wab"]) + WEIGHTS["net"] * _z(net_score)
                    + WEIGHTS["q1w"] * _z(out["q1w"]) - WEIGHTS["q34l"] * _z(out["q34l"]))
    return out


# --- bracket placement --------------------------------------------------------

def earliest_round(p: int, q: int) -> int:
    """Round (1 = round of 64 ... 4 = regional final) in which bracket positions ``p`` and ``q`` of one region
    can first meet."""
    for k in range(1, 5):
        if p >> k == q >> k:
            return k
    return 5


def required_round(meetings: int) -> int:
    """Earliest round two teams from the same conference may meet, given how often they've played."""
    return 4 if meetings >= 3 else 3 if meetings == 2 else 2


def _km(lat1, lon1, lat2, lon2) -> float:
    if lat1 is None or lat2 is None:
        return 0.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(12742 * np.arcsin(np.sqrt(a)))


class Bracketer:
    """Places a seed list into four regions by the bracketing principles. Built once per projection; ``place``
    runs per simulation.

    ``conf``: conference id per team column (or -1). ``geo``: optional dict with ``loc`` (team column ->
    (lat, lon)), ``regionals`` and ``pods`` (lists of sites with ``lat``, ``lon``, ``days``), ``blocked``
    (team column -> set of site ids it may not play at). ``sunday``: team columns that can't play on Sunday."""

    def __init__(self, conf: np.ndarray, geo: dict | None = None, sunday: set | None = None,
                 protect_conferences: bool = True):
        self.conf = conf
        self.protect = protect_conferences     # the women's bracket places its top 16 by true seed alone
        self.geo = geo
        self.sunday = sunday or set()
        if geo:
            loc = geo["loc"]
            self.d_reg = {t: [_km(*loc[t], s["lat"], s["lon"]) for s in geo["regionals"]] for t in loc}
            self.d_pod = {t: [_km(*loc[t], s["lat"], s["lon"]) for s in geo["pods"]] for t in loc}

    # -- rule checks
    def _site_ok(self, t: int, site: dict) -> bool:
        if site is None:
            return True
        if t in self.sunday and site.get("days") == "fri-sun":
            return False
        return site["id"] not in self.geo["blocked"].get(t, ()) if self.geo else True

    def _fits(self, teams, r, pos, st, meet) -> bool:
        for t in teams:
            for u, q in st["members"][r]:
                if self.conf[t] >= 0 and self.conf[t] == self.conf[u]:
                    if earliest_round(pos, q) < required_round(meet(t, u)):
                        return False
            if self.geo:
                if not self._site_ok(t, self.geo["regionals"][st["reg_site"][r]]):
                    return False
                pod = st["pod_site"][r][pos // 4]
                if pod is not None and not self._site_ok(t, self.geo["pods"][pod]):
                    return False
        return True

    def _protected_ok(self, teams, r, st) -> bool:
        """Top-four-line teams from one conference go to different regions."""
        if not self.protect:
            return True
        for t in teams:
            if t not in st["top4"]:                 # only a conference's top four on the seed list
                continue
            if any(self.conf[u] == self.conf[t] for u in st["protected"][r] if u in st["top4"]):
                return False
        return True

    def _pick_pod(self, t: int, st) -> int | None:
        if not self.geo:
            return None
        left = [i for i, n_ in enumerate(st["pod_left"]) if n_ > 0]
        ok = [i for i in left if self._site_ok(t, self.geo["pods"][i])] or left
        d = self.d_pod.get(t)
        return min(ok, key=lambda i: d[i] if d else i) if ok else None

    def _ok(self, ent, r, line, st, meet) -> bool:
        return self._fits(ent, r, POS_OF_SEED[line + 1], st, meet) and (line >= 4 or self._protected_ok(ent, r, st))

    def _perms(self, ents, line, st, meet) -> list[tuple]:
        """Every way of giving the line's entries distinct open regions that keeps the rules."""
        cand = [r for r in range(4) if line in st["open"][r]]
        ok = [[r for r in cand if self._ok(e, r, line, st, meet)] for e in ents]
        out = []

        def rec(i, used, acc):
            if i == len(ents):
                out.append(tuple(acc)); return
            for r in ok[i]:
                if r not in used:
                    rec(i + 1, used | {r}, acc + [r])
        rec(0, frozenset(), [])
        return out

    @staticmethod
    def _assign(st, ent, r, line, rank):
        pos = POS_OF_SEED[line + 1]
        st["open"][r].discard(line)
        for t in ent:
            st["members"][r].append((t, pos))
            if line < 4:
                st["protected"][r].append(t)
        st["sum"][r] += rank + 1

    @staticmethod
    def _copy(st):
        return {"members": [list(m) for m in st["members"]], "protected": [list(p) for p in st["protected"]],
                "sum": list(st["sum"]), "open": [set(o) for o in st["open"]], "reg_site": st["reg_site"],
                "pod_site": st["pod_site"], "pod_left": st["pod_left"], "top4": st["top4"]}

    def _completes(self, queue, line, st, meet, rank_of) -> bool:
        """Whether the protected lines after ``line`` can still be filled within the rules."""
        if line >= 3:
            return True
        nxt = line + 1
        for perm in self._perms(queue[nxt], nxt, st, meet):
            s2 = self._copy(st)
            for e, r in zip(queue[nxt], perm):
                self._assign(s2, e, r, nxt, rank_of[id(e)])
            if self._completes(queue, nxt, s2, meet, rank_of):
                return True
        return False

    def _cost(self, ents, perm, line, st, rank_of):
        if line < 4 and self.geo:
            d = [self.d_reg.get(e[0], [0] * 4)[st["reg_site"][r]] for e, r in zip(ents, perm)]
            if line == 0:
                return tuple(d)                     # in seed-list order: the No. 1 overall seed chooses first
            new = list(st["sum"])
            for e, r in zip(ents, perm):
                new[r] += rank_of[id(e)] + 1
            return (sum(d) / 500 + 0.25 * float(np.std(new)) * 2,)
        # S-curve: keep the regions' running totals of seed-list places as even as possible
        new = list(st["sum"])
        for e, r in zip(ents, perm):
            new[r] += rank_of[id(e)] + 1
        return (float(np.var(new)), perm)

    def place(self, entries: list[list[int]], meet) -> tuple[list, dict]:
        """``entries``: the 64 seed-list entries in order (a list of one team, or two for an opening-round
        pair). ``meet(t, u)``: times two teams have played. Returns, per entry, (region, position 0-15,
        seed line 1-16) and a summary (moves, unresolved conflicts, region order for the Final Four, the pod
        site per region and pod)."""
        st = {"members": [[] for _ in range(4)], "protected": [[] for _ in range(4)], "sum": [0] * 4,
              "open": [set(range(16)) for _ in range(4)],
              "reg_site": list(range(4)), "pod_site": [[None] * 4 for _ in range(4)], "pod_left": [2] * 8, "top4": set()}
        rank_of = {id(e): i for i, e in enumerate(entries)}
        seen: dict = {}
        for e in entries[:16]:                      # each conference's top four teams on the protected lines
            for t in e:
                c = self.conf[t]
                if c >= 0 and seen.get(c, 0) < 4:
                    seen[c] = seen.get(c, 0) + 1
                    st["top4"].add(t)
        result = {}
        moves = conflicts = 0
        conflict_lines = []
        queue = [list(entries[4 * l:4 * l + 4]) for l in range(16)]

        def good(line, ents, st):
            perms = self._perms(ents, line, st, meet)
            if line < 3:
                q = list(queue); q[line] = ents
                perms = [p for p in perms if self._completes(q, line, self._apply(st, ents, p, line, rank_of), meet, rank_of)]
            if line == 1 and perms:                 # the strongest 2 seed is not put with the No. 1 overall seed
                one = result[id(entries[0])][0]
                keep = [p for p in perms if not any(rank_of[id(e)] == 4 and r == one for e, r in zip(ents, p))]
                perms = keep or perms
            return perms

        for line in range(16):
            ents = queue[line]
            perms = good(line, ents, st)
            if not perms and 0 < line < 15:
                # move a team down a line and bring one up from the next: the lowest here and the best there first
                for i, j in itertools.product(reversed(range(len(ents))), range(len(queue[line + 1]))):
                    nxt = queue[line + 1]
                    trial = ents[:i] + ents[i + 1:] + [nxt[j]]
                    p2 = good(line, trial, st)
                    if p2:
                        queue[line + 1] = [ents[i]] + nxt[:j] + nxt[j + 1:]
                        ents, perms = trial, p2
                        queue[line] = ents
                        moves += 1
                        break
            if not perms:                           # nothing satisfies every rule: the fewest broken
                conflicts += 1
                conflict_lines.append(line + 1)
                cand = [r for r in range(4) if line in st["open"][r]]
                perms = sorted(itertools.permutations(cand, len(ents)),
                               key=lambda p: sum(not self._ok(e, r, line, st, meet) for e, r in zip(ents, p)))[:1]
            perm = min(perms, key=lambda p: self._cost(ents, p, line, st, rank_of))
            for e, r in zip(ents, perm):
                self._assign(st, e, r, line, rank_of[id(e)])
                result[id(e)] = (r, POS_OF_SEED[line + 1], line + 1)
            if line < 4 and self.geo:               # first/second-round sites, in seed-list order
                for e, r in sorted(zip(ents, perm), key=lambda x: rank_of[id(x[0])]):
                    pod = self._pick_pod(e[0], st)
                    st["pod_site"][r][POS_OF_SEED[line + 1] // 4] = pod
                    if pod is not None:
                        st["pod_left"][pod] -= 1
        # Final Four: region of the best No. 1 seed plays the region of the fourth, second plays third
        ones = sorted(range(4), key=lambda r: min(rank_of[id(e)] for e in queue[0] if result[id(e)][0] == r))
        order = [ones[0], ones[3], ones[1], ones[2]]
        return [result[id(e)] for e in entries], {"moves": moves, "conflicts": conflicts, "conflict_lines": conflict_lines, "region_order": order,
                                                  "pod_site": st["pod_site"], "reg_site": st["reg_site"]}

    def _apply(self, st, ents, perm, line, rank_of):
        s2 = self._copy(st)
        for e, r in zip(ents, perm):
            self._assign(s2, e, r, line, rank_of[id(e)])
        return s2
