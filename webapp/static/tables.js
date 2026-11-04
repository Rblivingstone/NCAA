// Sortable, filterable tables and shared game helpers.
// columns: [{key, label, num, render(row) -> html, sort(row) -> value, dir: 1 | -1 (first click), title}]
function DataTable(el, columns, opts = {}) {
  const st = { rows: [], key: opts.sortKey || null, dir: opts.dir || -1, q: "" };
  const text = opts.text || (() => "");
  function cmp(a, b) {
    if (a == null && b == null) return 0;
    if (a == null) return 1;
    if (b == null) return -1;
    return (a < b ? -1 : a > b ? 1 : 0) * st.dir;
  }
  function draw() {
    const q = st.q.trim().toLowerCase();
    let rows = q ? st.rows.filter((r) => text(r).toLowerCase().includes(q)) : st.rows.slice();
    const col = columns.find((c) => c.key === st.key);
    if (col && col.sort) rows.sort((a, b) => cmp(col.sort(a), col.sort(b)));
    const limit = opts.limit || 500;
    const head = `<tr>${columns.map((c) => `<th class="${c.num ? "num" : ""}${c.sort ? " sortable" : ""}${c.key === st.key ? " sorted" : ""}"
      data-k="${c.key}"${c.title ? ` title="${c.title}"` : ""}>${c.label}${c.key === st.key ? (st.dir > 0 ? " ▲" : " ▼") : ""}</th>`).join("")}</tr>`;
    el.innerHTML = head + (rows.length
      ? rows.slice(0, limit).map((r) => `<tr${opts.rowClass ? ` class="${opts.rowClass(r) || ""}"` : ""}>${columns.map((c) => `<td class="${c.num ? "num" : ""}">${c.render(r)}</td>`).join("")}</tr>`).join("")
      : `<tr><td class="muted" colspan="${columns.length}">${q ? `No games match “${st.q}”.` : opts.empty || "Nothing here yet."}</td></tr>`)
      + (rows.length > limit ? `<tr><td class="muted" colspan="${columns.length}">Showing ${limit} of ${rows.length}. Type a team name to narrow the list.</td></tr>` : "");
    el.querySelectorAll("th.sortable").forEach((th) => (th.onclick = () => {
      const k = th.dataset.k;
      if (st.key === k) st.dir = -st.dir; else { st.key = k; st.dir = columns.find((c) => c.key === k).dir || -1; }
      draw();
    }));
    if (opts.onDraw) opts.onDraw(rows);
  }
  return { set(rows) { st.rows = rows || []; draw(); }, filter(q) { st.q = q; draw(); } };
}

// "#12" for a ranked D1 team, "non-D1" otherwise.
function rankTag(r) { return r ? `<span class="rank">#${r}</span>` : `<span class="rank">non-D1</span>`; }

// Upset flag for a game: before it, when the ranking's underdog has a real chance; after it, when the underdog won.
function upsetBadge(g, done) {
  if (!g || !g.underdog) return "";
  const who = `${g.underdog} (${g.underdog === g.home ? (g.rank_home ? "#" + g.rank_home : "non-D1") : (g.rank_away ? "#" + g.rank_away : "non-D1")})`;
  if (done) return g.upset ? `<span class="badge upset" title="${who} beat higher-ranked ${g.favorite}; the model gave it ${fmtPct(g.p_underdog)}">upset${g.upset_level ? " · flagged" : ""}</span>` : "";
  if (!g.upset_level) return "";
  return g.upset_level === "pick"
    ? `<span class="badge upset" title="The model favours lower-ranked ${who} over ${g.favorite}">upset pick ${fmtPct(g.p_underdog)}</span>`
    : `<span class="badge watch" title="Lower-ranked ${who} has a ${fmtPct(g.p_underdog)} chance against ${g.favorite}">upset watch ${fmtPct(g.p_underdog)}</span>`;
}

// Sort key for "team ranking": the better-ranked team in the game (unranked teams last).
function bestRank(g) { const r = [g.rank_home, g.rank_away].filter((x) => x); return r.length ? Math.min(...r) : null; }
