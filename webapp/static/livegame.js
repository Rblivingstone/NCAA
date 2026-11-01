// Rendering of a game view (live, replayed, upcoming or finished), shared by the Live and Game pages.
"use strict";

const LiveGame = (() => {
  const LABELS = { pts: "Points", fgm: "FGM", fga: "FGA", fg3m: "3PM", fg3a: "3PA", ftm: "FTM", fta: "FTA",
    oreb: "Off reb", dreb: "Def reb", tov: "Turnovers", ast: "Assists", stl: "Steals", blk: "Blocks", pf: "Fouls" };
  // opts.mode: "live" (default), "pre" (before tip-off: prior only) or "post" (finished: update history vs prior)
  let mode = "live";
  function render(r) {
    const st = r.state, m = r.model, [H, A] = st.teams;
    const sc = st.teams.map((t) => st.box[t] ? st.box[t].pts : 0);
    let h = `<div class="live-top">
      <div class="panel scorebug">
        <div class="row"><span class="tm">${A}${st.status === "in" && st.possession === A ? ' <i class="ball">●</i>' : ""}</span><span class="pts">${sc[1]}</span></div>
        <div class="row"><span class="tm">${H}${st.status === "in" && st.possession === H ? ' <i class="ball">●</i>' : ""} <span class="muted">${st.neutral ? "(neutral)" : "(home)"}</span></span><span class="pts">${sc[0]}</span></div>
        <div class="muted">${st.detail}${st.status === "in" ? ` · ${st.remaining.toFixed(1)} min left` : ""}</div></div>`;
    if (m) {
      const pw = (p, who) => `<div class="stat"><div class="v">${fmtPct(p)}</div><div class="k">${who}</div></div>`;
      h += mode === "pre" ? `<div class="stats">${pw(m.pregame.p_win[0], `${H} wins — pre-game forecast`)}${pw(m.pregame.p_win[1], `${A} wins`)}` :
        mode === "post" ? `<div class="stats"><div class="stat"><div class="v">${sc[0] > sc[1] ? H : A}</div><div class="k">won by ${Math.abs(sc[0] - sc[1])}</div></div>` +
          pw(sc[0] > sc[1] ? m.pregame.p_win[0] : m.pregame.p_win[1], `pre-game chance ${sc[0] > sc[1] ? H : A} would win`) :
        `<div class="stats">${pw(m.p_win[0], `${H} wins — model now`)}${pw(m.pregame.p_win[0], "pre-game")}` +
        (st.espn_home_wp != null && mode !== "post" ? pw(st.espn_home_wp, "ESPN's win probability") : "") +
        (mode === "post" ? "" : `<div class="stat"><div class="v">${m.margin.mean >= 0 ? "+" : ""}${m.margin.mean.toFixed(1)}</div>
          <div class="k">projected final margin (${H}); 80%: ${m.margin.p10}…${m.margin.p90}</div></div>`) +
        (m.unusualness ? `<div class="stat"><div class="v" style="color:${m.unusualness.p_value < 0.05 ? "var(--upset)" : "inherit"}">${(100 * m.unusualness.p_value).toFixed(0)}%</div>
          <div class="k">${mode === "post" ? "unusualness of the whole box score" : "unusual so far"}: share of simulated games this unusual after ${m.unusualness.possessions} possessions${m.unusualness.drivers.length ?
            " · " + m.unusualness.drivers.map((d) => `${d.team} ${LABELS[d.stat] || d.stat}`).join(", ") : ""}</div></div>` : "") + `</div>`;
    } else h += `<div class="muted">One of these teams is not in the model.</div>`;
    h += `</div>`;
    h += `<div class="cols2" style="margin-top:12px">`;
    if (mode !== "pre") h += `<div class="panel"><b>Win probability for ${H}${mode === "post" ? ", play by play" : ""}</b>
      <div class="kde-legend"><span class="key"><i style="background:var(--ink)"></i>this model</span>
      <span class="key"><i style="background:var(--muted)"></i>ESPN (dashed)</span></div>${timeline(r.timeline || [])}</div>`;
    if (m && mode === "pre") {
      h += `<div class="panel"><b>Projected box score</b> <span class="muted">(prior: mean, 80% interval)</span>
        <table class="data"><tr><th>Stat</th><th class="num">${H}</th><th class="num">${A}</th></tr>` +
        Object.keys(LABELS).map((k) => `<tr><td>${LABELS[k]}</td>` + [0, 1].map((j) => {
          const f = m.final[k];
          return `<td class="num"><b>${f.mean[j]}</b> <span class="muted">(${f.p10[j]}–${f.p90[j]})</span></td>`;
        }).join("") + `</tr>`).join("") + `</table></div>`;
    } else if (m) {
      h += `<div class="panel"><b>Box score: ${mode === "post" ? "final vs pre-game forecast" : "so far → projected final"}</b> <span class="muted">${mode === "post" ? "(pre-game: posterior mean)" : "(posterior mean, 80% interval)"}</span>
        <table class="data"><tr><th>Stat</th><th class="num">${H} ${mode === "post" ? "final" : "now"}</th><th class="num">${mode === "post" ? "pre-game" : "final"}</th>
          <th class="num">${A} ${mode === "post" ? "final" : "now"}</th><th class="num">${mode === "post" ? "pre-game" : "final"}</th></tr>` +
        Object.keys(LABELS).map((k) => `<tr><td>${LABELS[k]}</td>` + [0, 1].map((j) => {
          const now = st.box[st.teams[j]] ? st.box[st.teams[j]][k] : 0, f = m.final[k], d = m.densities[k];
          return mode === "post" ? `<td class="num"><b>${now}</b></td><td class="num muted">${d.mean_pre[j]}</td>`
            : `<td class="num">${now}</td><td class="num"><b>${f.mean[j]}</b> <span class="muted">(${f.p10[j]}–${f.p90[j]})</span></td>`;
        }).join("") + `</tr>`).join("") + `</table>
        <div class="muted" style="margin-top:4px">${mode === "post" ? "" : `~${m.possessions_left.toFixed(0)} possessions each left · overtime chance ${fmtPct(m.p_overtime)} · `}
          ${m.effective_draws.toFixed(0)} effective posterior draws after the in-game update · ${r.posterior} posterior</div></div>`;
    }
    h += `</div>`;
    if (m) h += densityPanel(m, st, r.timeline);
    return h;
  }

  // ----- end-of-game distribution of every stat: now vs pre-game -----
  const shiftOf = (q) => Math.abs(2 * q - 1);
  function densityPanel(m, st, tl) {
    const teams = st.teams, lvl = m.shift_level;
    const flagged = [];
    let cards = "";
    for (const k of Object.keys(LABELS)) {
      const d = m.densities[k];
      [0, 1].forEach((j) => {
        const q = d.q[j], sh = shiftOf(q), sig = sh >= lvl;
        if (sig) flagged.push(`${teams[j]} ${LABELS[k]} ${q > 0.5 ? "↑" : "↓"}`);
        const now = st.box[teams[j]] ? st.box[teams[j]][k] : 0;
        if (mode === "pre") {
          cards += `<div class="kde-card live-kde"><div class="t"><span class="ttl"><i class="dot" style="background:var(--series-${j + 1})"></i>${teams[j]} · ${LABELS[k]}</span>
            <span>${d.mean_pre[j]} <em>expected</em></span></div>${densitySvg(d, j, null, true)}</div>`;
          return;
        }
        cards += `<div class="kde-card live-kde${sig ? " shifted" : ""}">
          <div class="t"><span class="ttl"><i class="dot" style="background:var(--series-${j + 1})"></i>${teams[j]} · ${LABELS[k]}</span>
            <span>${mode === "post" ? now : d.mean_live[j]} <em>vs ${d.mean_pre[j]} pre</em></span></div>
          ${densitySvg(d, j, now)}
          <div class="shift-row"><span>shift</span><b class="${sig ? "off" : ""}">${q > 0.5 ? "↑" : q < 0.5 ? "↓" : ""} ${(100 * sh).toFixed(0)}%</b></div>
          ${shiftSpark(tl || [], k, j, lvl)}</div>`;
      });
    }
    const intro = mode === "pre" ? `<b>Prior distribution of every stat</b> <span class="muted">— what the model expects each team's final box score to look like before tip-off.</span>`
      : mode === "post" ? `<b>Every stat: result vs the prior, and how the forecast moved</b>
      <span class="muted">— solid: the final value (the forecast after every update collapses onto it); dotted: the pre-game forecast.
      The small chart under each traces the shift play by play; a red edge means the stat ended outside the prior's central 95%.</span>` : "";
    if (intro) return `<div class="panel" style="margin-top:12px">${intro}
      <div class="kde-legend">${teams.map((t, j) => `<span class="key"><i style="background:var(--series-${j + 1})"></i>${t}</span>`).join("")}
        ${mode === "post" ? `<span class="muted">${flagged.length ? `Flagged: ${flagged.join(", ")}` : "Nothing outside the prior's central 95%."}</span>` : ""}</div>
      <div class="kde-grid live-grid">${cards}</div></div>`;
    return `<div class="panel" style="margin-top:12px"><b>End-of-game distribution of every stat</b>
      <span class="muted">— solid: the forecast now, after the Bayesian in-game updates; dotted: the pre-game forecast;
      thin line: the count so far. <b>Shift</b> = |2q − 1| with q = P(final now &gt; final pre-game): 0 when the forecasts agree,
      1 when they no longer overlap. Edges turn red at ${(100 * lvl).toFixed(0)}% (a two-sided 95% level). The small chart under
      each shows the shift through the game.</span>
      <div class="kde-legend">${teams.map((t, j) => `<span class="key"><i style="background:var(--series-${j + 1})"></i>${t}</span>`).join("")}
        <span class="muted">${flagged.length ? `Significant shifts: ${flagged.join(", ")}` : "No significant shifts yet."}</span></div>
      <div class="kde-grid live-grid">${cards}</div></div>`;
  }
  function densitySvg(d, j, now, priorOnly = false) {
    const W = 240, H = 90, P = { l: 4, r: 4, t: 6, b: 16 }, g = d.grid;
    const ymax = (priorOnly ? Math.max(...d.pre[j]) : Math.max(...d.live[j], ...d.pre[j])) || 1;
    const x = (v) => P.l + (v - g[0]) / (g[g.length - 1] - g[0]) * (W - P.l - P.r);
    const y = (v) => H - P.b - v / ymax * (H - P.t - P.b);
    const pts = (ys) => g.map((v, i) => `${x(v).toFixed(1)},${y(ys[i]).toFixed(1)}`).join("L");
    const col = `var(--series-${j + 1})`;
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="end-of-game distribution now vs pre-game">`;
    s += `<line class="axis" x1="${P.l}" x2="${W - P.r}" y1="${H - P.b}" y2="${H - P.b}"/>`;
    const main = priorOnly ? d.pre[j] : d.live[j];
    s += `<path class="area" fill="${col}" d="M${x(g[0])},${H - P.b}L${pts(main)}L${x(g[g.length - 1])},${H - P.b}Z"/>`;
    if (!priorOnly) s += `<path d="M${pts(d.pre[j])}" fill="none" stroke="${col}" stroke-width="1.5" stroke-dasharray="2 3" opacity=".9"/>`;
    s += `<path class="curve" stroke="${col}" d="M${pts(main)}"/>`;
    if (now != null && now >= g[0] && now <= g[g.length - 1]) s += `<line x1="${x(now)}" x2="${x(now)}" y1="${P.t}" y2="${H - P.b}" stroke="var(--muted)" stroke-width="1"/>`;
    const fmt = (v) => (Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(1));
    s += `<text x="${P.l}" y="${H - 3}">${fmt(g[0] + 2)}</text><text x="${W - P.r}" y="${H - 3}" text-anchor="end">${fmt(g[g.length - 1] - 2)}</text>`;
    return s + `</svg>`;
  }
  function shiftSpark(tl, k, j, lvl) {
    const W = 240, H = 34, maxX = Math.max(40, ...tl.map((t) => t.elapsed));
    const x = (v) => 4 + v / maxX * (W - 8), y = (v) => H - 3 - v * (H - 6);
    const pts = tl.filter((t) => t.q && t.q[k]).map((t) => `${x(t.elapsed).toFixed(1)},${y(shiftOf(t.q[k][j])).toFixed(1)}`);
    let s = `<svg class="spark" viewBox="0 0 ${W} ${H}" role="img" aria-label="shift over the game">`;
    s += `<rect x="4" y="${y(1)}" width="${W - 8}" height="${y(lvl) - y(1)}" fill="var(--upset-soft)"/>`;
    s += `<line x1="4" x2="${W - 4}" y1="${y(lvl)}" y2="${y(lvl)}" stroke="var(--upset)" stroke-width="1" stroke-dasharray="3 2"/>`;
    s += `<line x1="4" x2="${W - 4}" y1="${y(0)}" y2="${y(0)}" class="axis"/>`;
    s += `<line x1="${x(20)}" x2="${x(20)}" y1="${y(1)}" y2="${y(0)}" stroke="var(--line)"/>`;
    if (pts.length) s += `<polyline points="${pts.join(" ")}" fill="none" stroke="var(--series-${j + 1})" stroke-width="1.5"/>`;
    return s + `</svg>`;
  }

  function timeline(tl) {
    const W = 520, H = 200, pad = 26, left = 40, maxX = Math.max(40, ...tl.map((t) => t.elapsed));
    const x = (v) => left + v / maxX * (W - left - pad), y = (p) => H - pad - p * (H - 2 * pad);
    let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="win probability over the game">`;
    for (const p of [0, 0.5, 1]) s += `<line x1="${left}" x2="${W - pad}" y1="${y(p)}" y2="${y(p)}" stroke="var(--line)" ${p === 0.5 ? 'stroke-dasharray="3 3"' : ""}/>
      <text x="${left - 4}" y="${y(p) + 3}" text-anchor="end">${p * 100}%</text>`;
    s += `<line x1="${x(20)}" x2="${x(20)}" y1="${pad / 2}" y2="${H - pad}" stroke="var(--line)"/><text x="${x(20)}" y="${H - 8}" text-anchor="middle">half</text>`;
    s += `<text x="${left}" y="${H - 8}">tip</text><text x="${x(40)}" y="${H - 8}" text-anchor="middle">40'</text>`;
    const line = (key, col, dash) => {
      const pts = tl.filter((t) => t[key] != null).map((t) => `${x(t.elapsed).toFixed(1)},${y(t[key]).toFixed(1)}`);
      return pts.length ? `<polyline points="${pts.join(" ")}" fill="none" stroke="${col}" stroke-width="2" ${dash ? 'stroke-dasharray="5 3"' : ""}/>` : "";
    };
    s += line("espn", "var(--muted)", true) + line("ours", "var(--ink)", false);
    const last = tl[tl.length - 1];
    if (last && last.ours != null) s += `<circle cx="${x(last.elapsed)}" cy="${y(last.ours)}" r="4" fill="var(--ink)" stroke="var(--panel)" stroke-width="2"/>`;
    return s + `</svg>`;
  }
  return {
    render(el, r, opts = {}) { mode = opts.mode || (r.state.status === "pre" ? "pre" : "live"); el.innerHTML = render(r); },
    LABELS,
  };
})();
