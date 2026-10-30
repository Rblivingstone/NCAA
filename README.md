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

On the *Build and score* page: objective, champion, upset toggle (count,
definition, keep the Final Four), and a table of each candidate champion's
best bracket with its relative likelihood. Any bracket also gets **ESPN
points, predicted vs realized**. The prediction is its score in each of the
10,000 simulated tournaments (mean, 10–90% range, histogram). The realization
is its score against what actually happened, broken down by round, with the
delta between them. "Mark picks against actual results" puts ✓/✗ on every
pick.

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
* **Compare with actual results** (toggle on the distribution bracket) –
  each line shows the team that really got there, shaded by the model's
  pre-tournament probability of that. Games where the model's head-to-head
  favourite lost are marked ✗ (✓ otherwise), with a per-round summary.
* **Score a bracket** – click winners to enter results (or *Load actual
  results*, which fills in the real tournament from the data). Shows the
  log-likelihood, odds of exactly these outcomes, comparison with coin flips,
  Brier score, percentile among model-generated brackets, and the surprise of
  each game. Picks persist in the browser and can be exported/imported as JSON.
* **Simulate a game** – any two Division I teams, neutral or home court:
  win probability, margin distribution, average box score and sample box scores.
* **Team ratings** – offensive/defensive efficiency, net rating with 90%
  interval, tempo and component rates.

Set `NCAA_SEASON` to serve a different season.

## Tests

```bash
python -m pytest -q
```
