# NCAA Bayesian bracket simulator

A generative Bayesian model of NCAA men's basketball games, a Monte Carlo
tournament simulator built on it, and a Flask app to explore the resulting
**distribution bracket** and to score a real bracket against the model.

## The model (`ncaa/model.py`)

Every team has a latent skill vector with 22 components: an offensive and a
defensive effect for each part of a possession, plus tempo. The vectors share a
multivariate normal prior whose correlation matrix is learned (LKJ prior), so
being good at one thing informs the others.

When team *o* has the ball against team *d*, every event rate combines both
teams' skills:

```
logit p_s = mu_s + off[o, s] + def[d, s] + hfa_s * home
```

The box score is generated possession by possession, and the model is fit to
every one of these counts from each regular-season game:

| stat | likelihood |
|---|---|
| possessions (per 40 min) | log-normal, mean from both teams' tempo |
| turnovers | BetaBinomial(possessions, p_to) |
| foul-shot trips vs field goal attempts | BetaBinomial(plays, p_ftr) |
| 3PA share | BetaBinomial(FGA, p_three) |
| 2P / 3P makes | BetaBinomial(2PA, p_2), BetaBinomial(3PA, p_3) |
| offensive rebounds (new play) | BetaBinomial(misses, p_orb) |
| free throws made | BetaBinomial(FTA, p_ft) (offense only) |
| assists | BetaBinomial(FGM, p_ast) |
| steals, blocks (defense) | BetaBinomial(TOV, p_stl), BetaBinomial(misses, p_blk) |
| personal fouls (defense) | Poisson(possessions × rate) |

The beta-binomial concentrations capture night-to-night variation (hot and cold
shooting nights), which matters for realistic upset rates. **Points are never
modelled directly**: a simulated game's score is whatever its simulated makes
add up to, and the score decides the winner (ties go to overtime).

### Form: recent games count more

A team's residual over its last two weeks of games predicts its residual over
the following month (coefficient 0.15 ± 0.05 in the 2026 data); residuals from
earlier in the season add nothing. So skill isn't fixed over the season. Each
team has an offensive and a defensive **form**: random walks in two-week steps,
measured in points per 100 possessions, acting along the minimum-change
direction in skill space that adds one point per 100 possessions. Form is
pinned to zero for the last two weeks, so `z` is the team's skill entering the
tournament, and an older game sees `z + form(that period)`. The step size is
learned, so the data decide how quickly old results go stale. With the same
history, a team that won its last games looks stronger than one that won its
first ones. (A straight-line season drift was tried first; it did not improve
holdout predictions.)

### Within-game dynamics

Simulating possessions independently makes margins too wide (SD ≈ 14 points
for a given matchup, against ≈ 11 in real games): real teams protect leads
less as they grow (bench minutes, fouling, effort). The half-time scores in
the data show that about 17% of an *unexpected* first-half lead is given back
in the second half. The simulator therefore plays each game in eight
5-minute segments, and before each segment both teams' shooting shifts
slightly against whoever is ahead of expectation, at the rate that reproduces
that half-time regression. The regression's uncertainty is carried into the
simulations as posterior draws.

### Inference and validation

Inference is NUTS (NumPyro) over all ~8,000 parameters, run in float64 and
initialised (starting point and mass matrix) from a mean-field variational
fit. `--method vi` stops after the variational fit: a few minutes instead of
an hour, but it understates uncertainty.

On a holdout (fit on games before Feb 15 2026, predict the 1,240 games after),
win probabilities are well calibrated:

| model favourite's win probability | 0.55 | 0.65 | 0.75 | 0.84 | 0.92 |
|---|---|---|---|---|---|
| favourite actually won | 0.57 | 0.64 | 0.72 | 0.87 | 0.90 |

Simulated box-score averages and spreads match the real ones for every stat
except defensive rebounds, which the simulator undercounts by ~2.5 per game
(rebounds of missed free throws and team rebounds are not simulated).

## Tournament simulation (`ncaa/bracket.py`, `ncaa/simulate.py`)

Each simulated tournament draws one posterior sample of every team's skills and
plays all 67 games (First Four through the final), each a full box-score
simulation. Repeating that 10,000 times gives, for every node of the bracket,
the probability that each team is there and that it wins there.

To score a realized bracket, the probability of the exact set of outcomes is

```
P(bracket) = E_theta[ prod_g P(winner_g beats loser_g | theta) ]
```

averaging the product over posterior draws (outcomes are correlated through
the shared uncertainty about team skill). Head-to-head probabilities come from
simulating 100 games per posterior draw for every pair of tournament teams.
The realized log-likelihood is also placed within the distribution of
log-likelihoods of brackets the model generates itself, as a calibration check.

## Optimal brackets

Advancing the head-to-head favourite in every game is only *locally*
optimal: a slight favourite with a brutal next-round matchup can be worth
passing over. `ncaa/bracket.py` searches globally.

* **Dynamic programme.** A bracket's score is a sum over its 67 games, and the
  bracket is a tree. For every game, every team that could win it, and every
  upset count, the DP keeps the best way to fill the sub-bracket below that
  game. At the final this gives the best bracket *for each possible
  champion*, built back down from that champion, and the overall best is the
  best of those. It is exact for any per-game score (tested against brute
  force).
* **Most likely bracket.** The model's exact probability averages the product
  over posterior draws, so it isn't quite a per-game sum. A minorize–maximize
  loop climbs it (reweight the draws by the current bracket, re-solve the
  DP), and each step can only raise the exact probability. The climb is local,
  so it restarts from the plug-in optima, the game-by-game favourites, and
  each of the 10 most promising champions (optimise with the champion held
  fixed, then release it). Champion restarts do find better brackets that a
  single climb misses.
* **Upsets.** A likelihood-optimal bracket has few upsets (7 in 2026, against
  a median of 16 in simulated tournaments). The builder can require exactly
  *k* upsets ("winner seeded at least 1/3/5 lines lower"). The DP then
  spends them where they cost the least probability, and can hold the Final
  Four fixed so the upsets don't change the outcome. In 2026 the best
  16-upset bracket is only about 7× less likely than the optimum.
* **Maximum expected points.** For bracket pools (ESPN 10-20-40-80-160-320),
  the same DP, with the same constraints, maximises
  Σ points × P(picked team wins that game).

* **Win a pool.** Pools are won by beating every other entry, not by
  expected points, so a pick is worth most when it's right and few rivals
  made it. That only pays where the *public's* picks differ from the true
  odds, so rivals are simulated from an a-priori model of how people pick,
  not from the game model. Simulated draws from the game model agree with it
  everywhere, so the chalk bracket would beat them. Each rival advances a
  team with probability sigmoid(κ·(u_a − u_b)), where perceived strength u
  mixes our rating with seed line (weight α), and κ > 1 means favourites are
  over-picked. α and κ are drawn per rival from a range, so the pool is a
  mixture over plausible publics (four presets: mixed, chalky, informed,
  random). Given a simulated tournament, rivals' scores are independent,
  which gives P(finishing first against N rivals) in closed form for any N.
  Candidates come from the DP with per-pick value points × p × (p/q)^λ, where
  q is the public's pick share (λ = 0 is the max-points bracket), with and
  without a fixed champion. They are chosen on half the simulated
  tournaments and reported on the other half.

  2026, mixed public: against 10 rivals the max-points bracket is already
  best (18.7%). Against 100, an Illinois-champion bracket wins 2.9% vs 2.0%;
  Illinois is a 3-seed the model gives 5.5% for the title but the public
  picks 3.2% of the time. Against 1,000, a Michigan State champion wins
  0.63%, versus 0.16% for max-points and 0.10% for a random entry. If the
  public picks upsets *more* freely than the model, the logic reverses and
  chalk wins, so the public model is a choice in the UI.

* **Pool composition and difficulty.** Rivals are a configurable mix of
  archetypes: *casual* (mostly seeds), *informed* (close to the model),
  *chalk* (favourites nearly always) and *upset* (discounts seeds, picks
  freely). Each rival draws its own strategy parameters from its archetype's
  range. **Pick correlation** ρ uses a Gaussian copula: a rival picks team a
  when √ρ·Z_g + √(1−ρ)·Z_ig < Φ⁻¹(p_ig), where Z_g is a pool-wide shock for
  that game. Every pick share stays exactly the same whatever ρ is; only how
  often rivals make the *same combinations* changes. Simulated tournaments
  are paired with independent pool "climates", and given both, rivals are
  independent, so these all have exact formulas: expected/median/90th-
  percentile winning score, and P(first), P(top 10%), P(last) for any
  bracket (checked against brute-force pools). In 2026 at 100 rivals, raising
  ρ from 0 to 0.8 lowers the median winning score from 1,400 to 1,230. It
  also removes the edge from contrarian picks: a herd is beatable by the
  favourites bracket.

On the *Build and score* page: objective, champion, upset toggle (count,
definition, keep the Final Four), and a table of each candidate champion's
best bracket with its relative likelihood. Any bracket also gets **ESPN
points, predicted vs realized**. The prediction is its score in each of the
10,000 simulated tournaments (mean, 10–90% range, histogram). The realization
is its score against what actually happened, broken down by round, with the
delta between them. "Mark picks against actual results" puts ✓/✗ on every
pick.

## During a season: data feeds, nightly updates, live games

**Feeds** (`ncaa/data.py`, `ncaa/live.py`)

* *Nightly:* the sportsdataverse mirror of ESPN (box scores, schedules,
  play-by-play), updated daily. `fetch` handles a season in progress (no
  box scores yet, no bracket yet) and saves the full schedule.
* *Live:* ESPN's public site API: the scoreboard for a date, and the game
  summary (status, clock, box score, play-by-play). It is unofficial, but
  it's what public trackers use. It can't be reached from some sandboxes; the
  app then says so.
* *Replay:* the mirror archives each finished game's ESPN JSON, so any game
  can be stepped through minute by minute as if live. Box scores rebuilt from
  play-by-play match ESPN's official box exactly (tested), and possession is
  inferred from the last plays.

**Nightly update** (`python -m ncaa update --season 2027`, `ncaa/season.py`).
Run it from cron or a scheduler each morning:

1. Fetch the latest data.
2. Score every newly finished game against the posterior from *before* it
   (win probability, margin percentile, whole-box-score unusualness) into
   `artifacts/<season>/ledger.csv`, so the season record is out of sample.
3. Refit (`--refit vi` in minutes, `nuts` in about an hour).

Before the first games, the model is last season's posterior carried over:
skills are shrunk by ρ ≈ 0.65 and this season's uncertainty is added back.
Once games are played, the fit learns ρ (prior Beta(6, 3)). Teams new to
Division I start at average. In a backtest (fit on November 2025 games, predict the
1,170 December games), the carry-over prior improves on November games alone
(log-loss 0.434 → 0.421, Brier 0.145 → 0.140, margin RMSE 14.1 → 13.9). The
learned ρ is 0.67, also the prior's mean, so a month of games barely
identifies it.

**Ratings update the moment a game is final** (`ncaa/updates.py`). Between
nightly refits, each team carries a Gaussian belief over its 22 skills. When a
game goes final, its whole box score updates both teams through the model's
own likelihood (the same beta-binomial counts, fouls and tempo the fit uses,
with league-wide parameters at their posterior means). A Laplace
approximation (Newton to the mode, covariance from the curvature) gives the
new beliefs. This is assumed-density filtering: the cross-covariance between
the two teams is dropped, and each update takes under a second.

* On the 2026 final it matches exact importance-sampling updates of the full
  posterior within Monte Carlo noise.
* In a backtest (fit through Feb 15, 2026, then fold in each of the next
  1,240 games after predicting it), next-game log-loss improves from 0.588
  to 0.584. The gain grows with time since the fit (0.599 → 0.592 more than
  two weeks out).

Each game is scored in the out-of-sample ledger against the beliefs from
*before* it, then folded in and logged (`updates.csv`: both teams' rating and
uncertainty before and after). Updates arrive from three sources, and the
first one wins:

* `python -m ncaa watch` (the worker) follows every game from its scheduled
  tip-off, whether or not anyone has the page open (see below);
* a Game or Live page applies the final itself if the worker is behind;
* the nightly `update` reads the per-game ESPN archive (which updates sooner)
  and the box-score files.

**Head-to-head with ESPN.** Every game in the ledger also records ESPN's pre-game win
probability: its pre-game projection when the summary has one, otherwise its win
probability at the opening tip. A tip-off value of exactly 50% is ESPN's placeholder
when it has no pre-game view, so it counts as no forecast. The Season page scores both
models on the same games (log-loss, Brier score, how often the favorite won). It also
shows the per-game difference with a 95% interval, and each model's running log-loss
through the season. ESPN's number is filled in when the tracker records a final, and
the nightly update backfills the rest from the stored summaries or the game archive.

**Live tracking** (`ncaa/tracker.py`): each game is picked up 10 minutes before
its scheduled start, cross-checked against ESPN's scoreboard for delays, and polled
every `WATCH_EVERY` seconds (60 by default) until it is final. Every poll adds a
point to the game's win-probability history, stored in `artifacts/<season>/live/`.
Any gap (the first poll after tip-off, a worker restart, a feed outage) is rebuilt
minute by minute from the play-by-play. Open a game mid-way and you see its whole
history since tip-off; after the final the history stays on the game page.
Finished games are recorded with the beliefs from before the game, and then folded
into the ratings. If the worker's record is more than 2½ minutes old, the page polls
ESPN itself.

The nightly refit then resets the beliefs. **Every Division I team is
ranked** (365 in 2026–27, taken from conference membership in the schedule).
Renamed programs carry over their ratings (`data.TEAM_ALIASES`).

**Teams with no previous-season data** (`model.with_teams`) start at the
average of the teams we do have data on, in their group:

* a program new to Division I starts at the average D1 team;
* a non-D1 opponent starts at the pooled "Non-D1" rating, which is the
  average of the non-D1 schools D1 teams have played (about 35 pts/100
  below an average D1 team).

Either way the team gets the full spread between teams as its uncertainty,
since nothing says where it ranks in its group. This applies to the preseason
prior, live forecasts, win-probability histories and upcoming-game forecasts.

**Live in-game forecast** (`ncaa/ingame.py`). At each refresh the box score so
far updates the model in three Bayesian steps:

* the posterior draws of both teams' skills are reweighted by the
  beta-binomial likelihood of the partial counts;
* this game's rates get a conjugate beta update from the counts;
* the tempo is a precision-weighted blend of the pre-game pace and the pace
  so far.

The rest of the game is then simulated from the current score, giving a
projected final box score (mean and 80% interval) and each team's win
probability. The last two minutes use a measured fouling regime: possession
counts and the leader's free-throw share by margin, from 6,170 games of
2026 play-by-play. *Unusual so far* scores the partial box score against the
pre-game model over the same number of possessions.

Calibration on 300 replayed 2026 regular-season games (Brier; ESPN's own
win probability alongside): halftime 0.114 vs ESPN 0.130, 5:00 left 0.067 vs
0.063, 1:00 left 0.056 vs 0.048. At halftime the model is competitive (these
games were in its training data, which flatters it). In the last two minutes
ESPN is better; it presumably uses shot-clock and timeout state that this
model doesn't.

**How far the game has moved the forecast.** The Live page shows the end-of-game
density of every stat for each team (28 charts). The solid curve is the forecast
now, after the in-game updates; the dotted curve is the pre-game forecast; a
thin line marks the count so far. Movement is measured by the probability of
superiority q = P(final now > final pre-game), with ties counted half. The
**shift** |2q − 1| is 0 when the two forecasts agree and 1 when they no longer
overlap. A chart's edge turns red above 0.95 (a two-sided 95% level), and a
sparkline under each chart traces the shift through the game. (A classic
two-sample test would be meaningless here: with thousands of simulations
every difference is "significant", and the live forecast contains the
observed counts.) At the final whistle q is the real value's mid-percentile
in the pre-game forecast, so a flag means the stat finished outside the
model's central 95%. In the 2026 final, Michigan's 3-pointers ↓ and UConn's
offensive rebounds ↑ drifted throughout and are flagged at the end.

**Live tournament bracket.** `simulate_tournament(forced=..., live=...)` locks
in finished games and settles games in progress with their current in-game
win probability. The distribution bracket's *As of* control re-simulates the
rest of the tournament from any day.

**Pages:** *Season* (status, upcoming-game forecasts, the out-of-sample
ledger) and *Live* (ESPN live with 20-second refresh, or replay any 2026
tournament game with a slider and play button).

## Running it

```bash
pip install -r requirements.txt
```

```bash
python -m ncaa build --season 2026
```

`build` runs `fetch` (box scores and bracket from the sportsdataverse ESPN
mirror into `data/2026/`), `fit` (NUTS, about an hour on 4 cores; writes
`artifacts/2026/`), and `simulate` (10,000 tournaments plus the pairwise win
matrix, a few minutes). The steps can be run one at a time, e.g.
`python -m ncaa fit --method vi` for a quick fit.

```bash
flask --app webapp.app run
```

Then open http://127.0.0.1:5000:

* **Distribution bracket** – each line shows the most likely occupant and its
  probability; hover a game for the two most likely teams, the full
  distribution of who reaches it and who wins it, and the most likely
  matchups. Click a team (or use *Follow team*) to trace its odds through the
  bracket. An advancement-odds table sits below.
* **Potential-upset flags** (both bracket pages, on by default) – a ⚠ badge
  marks any game whose worse-seeded team has at least a 35/40/45/50%
  head-to-head chance (threshold selectable). It uses the picked matchup on
  the build page, the actual one in compare mode, and the most likely one in
  the distribution view.
* **Compare with actual results** (toggle on the distribution bracket) –
  each line shows the team that really got there, shaded by the model's
  pre-tournament probability of that. Games where the model's head-to-head
  favourite lost are marked ✗ (✓ otherwise), with a per-round summary.
* **Score a bracket** – click winners to enter results (or *Load actual
  results*, which fills in the real tournament from the data). Shows the
  log-likelihood, odds of exactly these outcomes, comparison with coin flips,
  Brier score, percentile among model-generated brackets, and the surprise of
  each game. Picks persist in the browser and can be exported/imported as JSON.
* **Simulate a tournament** (on the build page) – one random run of the whole
  bracket: one posterior draw, every game played possession by possession,
  upsets wherever the dice fall. Each game shows its score, hovering shows
  the full box score, and the seed is shown so a run can be replayed. The
  simulated bracket is scored like any other.
* **Simulate a game** – any two Division I teams, neutral or home court:
  win probability, margin distribution, average box score and sample box
  scores, plus a kernel-density chart of every box-score stat for both teams
  (with mean lines; hover for the chance of reaching a value). If the two
  teams really played, pick the game under *Compare with real game*. A
  tournament game is out of sample and is chosen by default; a regular-season
  game is labelled as part of the training data. The venue is matched, every
  real stat is marked on its density with its percentile (red outside the
  model's 90% range), the margin is placed in the simulated margin
  distribution, and the most surprising stats are listed. Tournament box
  scores are saved by `fetch` to `data/<season>/tournament_games.csv` and are
  never used for fitting.

  **Whole-game unusualness** (`ncaa/evaluate.py`). Box-score stats move
  together, so per-stat percentiles can't simply be multiplied. Instead the
  real box score is placed by its Mahalanobis distance from the cloud of
  simulated box scores, using their own correlations. The summary is the
  share of simulated games at least that far out: 46% is ordinary, 1% is
  extraordinary. The page also names the stats that drive it, and a
  tournament-wide panel ranks every game. Over the 66 tournament games of
  2026 the shares spread evenly, as calibration requires: 6% below 0.05,
  17% below 0.10, median 0.54. Field-goal and free-throw *attempts* are left
  out of this joint score. In the simulator they are fixed by possession
  bookkeeping, while real box scores are looser (team rebounds, end-of-half
  heaves), and including them made 20% of games score below 0.05. That
  rigidity is a known limitation of the simulator.
* **Rankings** – overall (net rating), offense or defense, for any season,
  with 90% intervals, tempo and season trend.
* **Team pages** – current strength and rank, and how the assessment has
  moved (within-season form, plus nightly snapshots once a season is
  tracked). Core strengths and weaknesses: each skill's contribution in
  points per 100 possessions with its percentile among D1 teams.
  Matchup vulnerabilities: how much an opponent one SD better at one thing
  (3-point shooting or volume, interior scoring, offensive rebounding,
  drawing fouls, ball pressure, heavy fouling, rim protection, perimeter
  defense, defensive rebounding) hurts this team *relative to how it hurts
  an average team*, with the probability that the difference is real.
  These are second-order effects, since the model has no explicit matchup
  terms. Plus a plain-language assessment, a tempo note and the team's games.
* **Games** – any date or team: upcoming games with forecasts; finished
  games with the logged pre-game forecast and unusualness. Each game has its
  own page: before tip-off the prior distributions; during, the live
  Bayesian forecast (20-second refresh); afterwards, the play-by-play update
  history (win probability vs ESPN, the shift of every stat) against the
  prior.

Set `NCAA_SEASON` to serve a different season for the bracket pages and
`NCAA_LIVE_SEASON` (default 2027) for the season and live pages.

## Deploying to a VPS (Docker, HTTPS on your domain)

Everything runs from `docker-compose.yml`:

- **web**: gunicorn serving the Flask app on `127.0.0.1:$WEB_PORT`, reachable from the server only.
- **worker**: the first build, the live-game watcher and the nightly update.
- **caddy** (optional): a reverse proxy that gets and renews a Let's Encrypt certificate for your domain.

Models, ledgers and team ratings live in Docker volumes (`ncaa_artifacts`, `ncaa_data`), so they survive rebuilds.

The script picks the HTTPS setup from what it finds on the server:

- **Nothing on ports 80/443** (a fresh server): the bundled Caddy takes them and handles HTTPS.
- **Another web server already on 80/443** (a server hosting other sites): the app stays on localhost behind it.
  - **nginx**: the script adds a site for your domain (it leaves your other sites alone and backs out if
    `nginx -t` fails), then gets the certificate with certbot.
  - **Apache or a host Caddy**: it prints the few lines to add.

Force a mode with `PROXY=caddy` or `PROXY=host`.

1. Point your domain's DNS **A** record (and AAAA, if the server has IPv6) at the VPS.
2. On the VPS (Ubuntu or Debian, 2+ vCPUs and 4 GB RAM recommended):

   ```bash
   git clone https://github.com/Rblivingstone/NCAA.git /opt/ncaa && cd /opt/ncaa
   sudo ./deploy/deploy.sh ncaa.example.com you@example.com
   ```

   The script installs Docker if it is missing, checks DNS, writes `.env`, starts the stack and sets up HTTPS.
   Re-running it is safe.
3. The first start builds the models (fetch, fit, simulate: about 20–90 minutes). Until then the site shows a
   "building" page. Follow it with `docker compose logs -f worker`.

To ship new code later, run `sudo ./deploy/deploy.sh update` (git pull, rebuild, restart).

Settings live in `/opt/ncaa/.env`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `NCAA_SEASON` | 2026 | tournament season shown on the bracket pages |
| `NCAA_LIVE_SEASON` | 2027 | season tracked live: final games, ratings, ledger |
| `BOOTSTRAP_FIT` | vi | first-build fit: `vi` takes about 10 min; `nuts` is the exact posterior and takes 1–3 h |
| `NIGHTLY_FIT` | vi | nightly refit method (`none` only folds in new games) |
| `NIGHTLY_HOUR` | 10 | UTC hour of the nightly update |
| `WATCH_EVERY` | 60 | seconds between polls of games in progress |
| `WEB_PORT` | 8000, or the first free port after it | local port the app listens on |
| `COMPOSE_PROFILES` | `caddy` or empty | set by the script: whether the bundled Caddy runs |

After changing `.env`, run `docker compose up -d`. With the bundled Caddy, you can add a password by uncommenting
`basic_auth` in `deploy/Caddyfile`. `/healthz` reports whether the models are built.

## Tests

```bash
python -m pytest -q
```
