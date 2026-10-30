// Shared bracket rendering for the distribution view and the results picker.
"use strict";

const fmtPct = (p) => (p >= 0.995 ? ">99%" : p < 0.005 && p > 0 ? "<1%" : Math.round(p * 100) + "%");

function teamLabel(t) {
  return `<span class="seed">${t.seed}</span><span class="name">${t.name}</span>`;
}

class BracketView {
  /**
   * opts.mode: "dist" (probability bracket) or "pick" (enter results)
   * opts.onPick(gameId, teamIndex) in pick mode
   */
  constructor(el, summary, opts = {}) {
    this.el = el;
    this.s = summary;
    this.b = summary.bracket;
    this.mode = opts.mode || "dist";
    this.onPick = opts.onPick || (() => {});
    this.picks = {};          // gameId -> team index (pick mode)
    this.gameScores = {};     // gameId -> scoring info (pick mode)
    this.highlight = null;    // team index highlighted in dist mode
    this.compare = null;      // /api/compare data: overlay the real results
    this.markActual = false;  // pick mode: mark picks right/wrong against the real results
    this.simGames = null;     // pick mode: a simulated tournament's games (scores, box scores)
    this.games = {};
    this.b.games.forEach((g, i) => (this.games[g.id] = { ...g, dist: summary.games[i], idx: i }));
    this.tip = document.getElementById("tooltip");
    this.build();
  }

  team(i) { return this.b.teams[i]; }

  // ----- layout -------------------------------------------------------------
  build() {
    const R = this.b.regions;
    const ff = this.b.games.filter((g) => g.round === 0);
    let html = "";
    if (ff.length) {
      html += `<div class="first-four"><div class="ff-title">First Four</div>`;
      html += ff.map((g) => this.gameHtml(g.id)).join("") + "</div>";
    }
    html += `<div class="bracket">`;
    html += `<div class="half left">${this.regionHtml(R[0], false)}${this.regionHtml(R[1], false)}</div>`;
    html += `<div class="center">
        <div class="center-col">${this.labelHtml("Final Four")}${this.gameHtml("R5-0")}</div>
        <div class="center-col champ">${this.labelHtml("Championship")}${this.gameHtml("R6")}
          <div class="champion" id="champion"></div></div>
        <div class="center-col">${this.labelHtml("Final Four")}${this.gameHtml("R5-1")}</div>
      </div>`;
    html += `<div class="half right">${this.regionHtml(R[2], true)}${this.regionHtml(R[3], true)}</div>`;
    html += `</div>`;
    this.el.innerHTML = html;
    this.el.querySelectorAll(".game").forEach((node) => {
      node.addEventListener("mouseenter", (e) => this.showTip(node.dataset.id, e));
      node.addEventListener("mousemove", (e) => this.moveTip(e));
      node.addEventListener("mouseleave", () => this.hideTip());
    });
    this.el.querySelectorAll(".slot").forEach((node) => {
      node.addEventListener("click", () => this.slotClick(node));
    });
    this.refresh();
  }

  labelHtml(t) { return `<div class="round-label">${t}</div>`; }

  regionHtml(region, mirrored) {
    const cols = [];
    const names = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8"];
    for (let r = 1; r <= 4; r++) {
      const gs = this.b.games.filter((g) => g.round === r && g.region === region);
      cols.push(`<div class="round r${r}">${this.labelHtml(names[r - 1])}<div class="round-games">` +
        gs.map((g) => this.gameHtml(g.id)).join("") + "</div></div>");
    }
    if (mirrored) cols.reverse();
    return `<div class="region ${mirrored ? "mirrored" : ""}"><div class="region-name">${region}</div>
      <div class="region-rounds">${cols.join("")}</div></div>`;
  }

  gameHtml(id) {
    return `<div class="game" data-id="${id}">
      <div class="slot" data-id="${id}" data-side="0"></div>
      <div class="slot" data-id="${id}" data-side="1"></div></div>`;
  }

  // ----- content ------------------------------------------------------------
  // Team occupying a side of a game in pick mode (null if undecided).
  pickedSide(g, side) {
    const [kind, v] = g.sources[side];
    if (kind === "team") return v;
    return this.picks[v] ?? null;
  }

  refresh() {
    this.el.querySelectorAll(".game").forEach((node) => {
      const c = this.compare && this.compare.games[node.dataset.id];
      const show = c && (this.mode === "dist");
      node.classList.toggle("model-wrong", !!(show && c.wrong));
      node.classList.toggle("model-right", !!(show && !c.wrong));
    });
    this.el.querySelectorAll(".slot").forEach((node) => {
      const g = this.games[node.dataset.id];
      const side = +node.dataset.side;
      node.className = "slot";
      if (this.mode === "dist") this.fillDist(node, g, side);
      else this.fillPick(node, g, side);
    });
    const champ = this.el.querySelector("#champion");
    if (this.mode === "dist" && this.compare && this.compare.games["R6"]) {
      const c = this.compare.games["R6"];
      const w = this.games["R6"].dist.teams.find((t) => this.team(t.team).name === c.winner);
      champ.innerHTML = `Actual champion<br><b>${c.winner}</b> model gave ${fmtPct(w ? w.p_win : 0)}`;
    } else if (this.mode === "dist") {
      const w = this.games["R6"].dist.teams.slice().sort((a, b) => b.p_win - a.p_win)[0];
      champ.innerHTML = `Most likely champion<br><b>${this.team(w.team).name}</b> ${fmtPct(w.p_win)}`;
    } else {
      const w = this.picks["R6"];
      champ.innerHTML = w != null ? `Champion<br><b>${this.team(w).name}</b>` : "";
    }
  }

  fillDist(node, g, side) {
    const dist = g.dist.sides[side];
    const c = this.compare && this.compare.games[g.id];
    if (c) {
      // Who was really here, shaded by how likely the model thought that was.
      const name = c.teams[side], p = c.p_slot[side];
      const t = this.b.teams.findIndex((x) => x.name === name);
      node.style.setProperty("--p", Math.round(p * 100) + "%");
      node.classList.add(name === c.winner ? "winner" : "loser");
      if (name !== c.model_top[side]) node.classList.add("not-top");
      node.innerHTML = `${teamLabel(this.team(t))}<span class="prob">${fmtPct(p)}</span>`;
      node.dataset.team = t;
      return;
    }
    let t, p;
    if (this.highlight != null) {
      t = this.highlight;
      p = dist[t] || 0;
      if (!p) {
        // Show this side's favourite, dimmed, when the highlighted team can't be here.
        [t, p] = this.topOf(dist);
        node.classList.add("dim");
      } else node.classList.add("hl");
    } else [t, p] = this.topOf(dist);
    const shade = Math.round(p * 100);
    node.style.setProperty("--p", shade + "%");
    node.innerHTML = `${teamLabel(this.team(t))}<span class="prob">${fmtPct(p)}</span>`;
    node.dataset.team = t;
  }

  topOf(dist) {
    let best = null, bp = -1;
    for (const [t, p] of Object.entries(dist)) if (p > bp) { best = +t; bp = p; }
    return [best, bp];
  }

  fillPick(node, g, side) {
    const t = this.pickedSide(g, side);
    node.style.removeProperty("--p");
    node.removeAttribute("title");
    if (t == null) { node.innerHTML = `<span class="empty">—</span>`; delete node.dataset.team; return; }
    node.dataset.team = t;
    const w = this.picks[g.id];
    let extra = "";
    const sc = this.gameScores[g.id];
    const sim = this.simGames && this.simGames[g.id];
    if (sim && w != null) {
      // A simulated game: show its final score instead of the model probability.
      node.classList.add(w === t ? "winner" : "loser");
      const s = sim.score[sim.teams.indexOf(this.team(t).name)];
      node.innerHTML = `${teamLabel(this.team(t))}<span class="score">${s}${sim.ot && w === t ? ` <span class="muted">${sim.ot > 1 ? sim.ot : ""}OT</span>` : ""}</span>`;
      if (w === t && sc && sc.p_win < 0.5) node.classList.add("upset");
      return;
    }
    if (w != null) {
      node.classList.add(w === t ? "winner" : "loser");
      const c = this.markActual && this.compare && this.compare.games[g.id];
      if (c && w === t) {
        const right = c.winner === this.team(t).name;
        node.classList.add(right ? "pick-right" : "pick-wrong");
        node.title = right ? "Correct pick" : `Wrong: ${c.winner} won this game`;
      }
      if (w === t && sc) {
        node.classList.add(sc.p_win >= 0.5 ? "expected" : "upset");
        extra = `<span class="prob">${fmtPct(sc.p_win)}</span>`;
      }
    } else node.classList.add("open");
    node.innerHTML = `${teamLabel(this.team(t))}${extra}`;
  }

  slotClick(node) {
    if (node.dataset.team === undefined) return;
    if (this.mode === "dist" && this.compare) return;
    const t = +node.dataset.team;
    if (this.mode === "dist") {
      this.highlight = this.highlight === t ? null : t;
      this.refresh();
      document.dispatchEvent(new CustomEvent("highlight", { detail: this.highlight }));
      return;
    }
    const id = node.dataset.id;
    this.simGames = null;  // editing picks leaves the simulated tournament
    if (this.picks[id] === t) delete this.picks[id];
    else this.picks[id] = t;
    this.prune();
    this.refresh();
    this.onPick(id, t);
  }

  // Drop picks that are no longer consistent with earlier rounds.
  prune() {
    for (const g of this.b.games) {
      const w = this.picks[g.id];
      if (w == null) continue;
      if (this.pickedSide(g, 0) !== w && this.pickedSide(g, 1) !== w) delete this.picks[g.id];
    }
  }

  setPicksByName(byName) {
    this.simGames = null;
    const idx = {};
    this.b.teams.forEach((t, i) => (idx[t.name] = i));
    this.picks = {};
    for (const g of this.b.games) {        // games are in round order
      const name = byName[g.id];
      if (name != null && idx[name] != null) this.picks[g.id] = idx[name];
    }
    this.prune();
    this.refresh();
  }

  picksByName() {
    const out = {};
    for (const [id, t] of Object.entries(this.picks)) out[id] = this.team(t).name;
    return out;
  }

  // ----- tooltip ------------------------------------------------------------
  roundName(g) {
    const r = this.b.rounds[g.round];
    return g.region ? `${g.region} · ${r}` : (g.label ? `${r} · ${g.label}` : r);
  }

  showTip(id, e) {
    const g = this.games[id];
    const d = g.dist;
    const teams = d.teams.slice(0, 16);
    const top2 = d.teams.slice(0, 2);
    let h = `<div class="tip-head">${this.roundName(g)}</div>`;
    const sim = this.mode === "pick" && this.simGames && this.simGames[id];
    if (sim) {
      const L = { pts: "PTS", fgm: "FGM", fga: "FGA", fg3m: "3PM", fg3a: "3PA", ftm: "FTM", fta: "FTA", oreb: "OREB",
        dreb: "DREB", tov: "TOV", ast: "AST", stl: "STL", blk: "BLK", pf: "PF", poss: "Poss" };
      h += `<div class="tip-sub">Simulated game${sim.ot ? ` (${sim.ot > 1 ? sim.ot : ""}OT)` : ""} ·
        model gave ${sim.winner} ${fmtPct(sim.p_winner)} head to head</div><table class="tip-table box"><tr><th></th>` +
        Object.values(L).map((l) => `<th>${l}</th>`).join("") + "</tr>" +
        sim.teams.map((t, j) => `<tr><td>${t === sim.winner ? "<b>" + t + "</b>" : t}</td>` +
          Object.keys(L).map((k) => `<td>${sim.box[k][j]}</td>`).join("") + "</tr>").join("") + "</table>";
      this.tip.innerHTML = h;
      this.tip.style.display = "block";
      this.moveTip(e);
      return;
    }
    const c = this.compare && this.compare.games[id];
    if (c && (this.mode === "dist" || this.markActual)) {
      h += `<div class="tip-pick ${c.wrong ? "bad" : "good"}">${c.wrong ? "✗ Model got this wrong" : "✓ Model's favorite won"}:
        <b>${c.winner}</b> beat ${c.loser}; model gave ${c.winner} ${fmtPct(c.p_winner)} head to head.
        Pre-tournament chance of being in this game: ${c.teams.map((t, i) => `${t} ${fmtPct(c.p_slot[i])}`).join(", ")}.</div>`;
    }
    if (this.mode === "pick" && this.gameScores[id]) {
      const sc = this.gameScores[id];
      h += `<div class="tip-pick"><b>${sc.winner}</b> over ${sc.loser}: model gave ${fmtPct(sc.p_win)}
        (${sc.surprise_bits.toFixed(2)} bits of surprise)</div>`;
    }
    h += `<div class="tip-sub">Most likely to be here</div><div class="tip-top2">` +
      top2.map((t) => `<div>${teamLabel(this.team(t.team))}<b>${fmtPct(t.p_here)}</b></div>`).join("") + "</div>";
    h += `<table class="tip-table"><tr><th>Team</th><th>Here</th><th>Wins</th><th></th></tr>`;
    for (const t of teams) {
      h += `<tr><td>${teamLabel(this.team(t.team))}</td><td>${fmtPct(t.p_here)}</td><td>${fmtPct(t.p_win)}</td>
        <td class="bar-cell"><span class="bar here" style="width:${t.p_here * 60}px"></span><span class="bar win" style="width:${t.p_win * 60}px"></span></td></tr>`;
    }
    if (d.teams.length > teams.length) h += `<tr><td colspan="4" class="muted">+ ${d.teams.length - teams.length} more teams</td></tr>`;
    h += `</table>`;
    if (d.matchups.length) {
      h += `<div class="tip-sub">Most likely matchups</div><table class="tip-table">`;
      for (const m of d.matchups.slice(0, 4)) {
        const [a, b] = m.teams.map((t) => this.team(t));
        h += `<tr><td>${a.name} vs ${b.name}</td><td>${fmtPct(m.p)}</td>
          <td class="muted">${a.name} wins ${fmtPct(m.p_top_wins)}, avg ${m.avg_score[0].toFixed(0)}–${m.avg_score[1].toFixed(0)}</td></tr>`;
      }
      h += `</table>`;
    }
    this.tip.innerHTML = h;
    this.tip.style.display = "block";
    this.moveTip(e);
  }

  moveTip(e) {
    const tip = this.tip, pad = 16;
    let x = e.clientX + pad, y = e.clientY + pad;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    if (x + w > window.innerWidth - 8) x = e.clientX - w - pad;
    if (y + h > window.innerHeight - 8) y = Math.max(8, window.innerHeight - h - 8);
    tip.style.left = Math.max(8, x) + "px";
    tip.style.top = y + "px";
  }

  hideTip() { this.tip.style.display = "none"; }
}
