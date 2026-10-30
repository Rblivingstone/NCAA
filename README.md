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
passing over. `ncaa/bracket.py` finds the global optimum exactly.

* **Most likely bracket.** A bracket's probability is a product over its 67
  games, and the bracket is a tree, so a dynamic programme finds the best one
  exactly. For each game and each team that could win it, it keeps the
  best-scoring way to fill the whole sub-bracket below that game with that
  team as winner, combining the two sides by trying every possible opponent.
  The exact objective averages the product over posterior draws, so it isn't a
  pure sum over games. A minorize–maximize loop handles this: reweight the
  draws by how well they explain the current bracket, then re-solve the DP.
  Each step can only raise the exact probability. It runs from three starts
  and keeps the best; the whole search takes a few hundredths of a second.
* **Maximum expected points.** For bracket pools (ESPN 10-20-40-80-160-320
  scoring by default), the same DP maximises
  Σ points × P(picked team wins that game), using the simulated probabilities.

Both are on the *Score a bracket* page, with how they compare with the
game-by-game favourites.

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
