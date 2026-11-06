#!/usr/bin/env bash
# Background jobs: build the models on first start, then keep the season current.
#   * once:     fetch + fit + simulate the bracket season ($NCAA_SEASON) if not built yet
#   * always:   `ncaa watch` - follow every game from its scheduled tip-off (win-probability history on
#               disk, whether or not anyone is watching) and update team ratings the moment it goes final
#   * nightly:  `ncaa update` at $NIGHTLY_HOUR (UTC) - new box scores, ledger, refit; then the Road to March
#               projection. Geography for the bracket (`ncaa geo`) is built when missing and refreshed weekly.
set -uo pipefail
cd /app
SEASON="${NCAA_SEASON:-2026}"
LIVE="${NCAA_LIVE_SEASON:-2027}"
FIT="${BOOTSTRAP_FIT:-vi}"          # vi: ~10 min; nuts: ~1-3 h (exact MCMC)
NIGHTLY_FIT="${NIGHTLY_FIT:-vi}"
HOUR="${NIGHTLY_HOUR:-10}"
WATCH_EVERY="${WATCH_EVERY:-60}"
PROJECT_SIMS="${PROJECT_SIMS:-1000}"   # Road to March: ~5 min here per 1000 (about 0.7 GB)

log() { echo "[$(date -u +%FT%TZ)] $*"; }

if [ ! -f "artifacts/$SEASON/summary.json" ]; then
  log "first start: building the $SEASON models (fit method: $FIT)"
  python -m ncaa fetch --season "$SEASON" &&
  python -m ncaa fit --season "$SEASON" --method "$FIT" &&
  python -m ncaa simulate --season "$SEASON" || { log "build failed; retrying in 10 min"; sleep 600; exec "$0"; }
  log "build finished"
fi

geo() {   # team locations and tournament sites; keeps a hand-edited sites.json
  python -m ncaa geo --season "$LIVE" || log "geography build failed (the bracket is placed without it)"
}
[ -f "data/$LIVE/geo_teams.csv" ] && [ -f "data/$LIVE/sites.json" ] || geo

log "catching up the $LIVE season"
python -m ncaa update --season "$LIVE" --refit none || log "catch-up failed (will retry tonight)"
python -m ncaa project --season "$LIVE" --project-sims "$PROJECT_SIMS" || log "Road to March projection failed (will retry tonight)"

# Live updates all day; restart the watcher if it ever exits.
( while true; do python -m ncaa watch --season "$LIVE" --every "$WATCH_EVERY"; log "watcher exited; restarting"; sleep 30; done ) &

while true; do
  now=$(date -u +%s)
  next=$(date -u -d "today ${HOUR}:00" +%s)
  [ "$next" -le "$now" ] && next=$((next + 86400))
  log "next nightly update at $(date -u -d "@$next" +%FT%TZ)"
  sleep $((next - now))
  log "nightly update ($NIGHTLY_FIT)"
  python -m ncaa update --season "$LIVE" --refit "$NIGHTLY_FIT" || log "nightly update failed"
  [ "$(date -u +%u)" = "1" ] && geo                              # Mondays: refresh arenas and sites
  log "Road to March projection ($PROJECT_SIMS simulations)"
  python -m ncaa project --season "$LIVE" --project-sims "$PROJECT_SIMS" || log "Road to March projection failed"
done
