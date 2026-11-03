#!/usr/bin/env bash
# Background jobs: build the models on first start, then keep the season current.
#   * once:     fetch + fit + simulate the bracket season ($NCAA_SEASON) if not built yet
#   * always:   `ncaa watch` - update team ratings the moment each game goes final
#   * nightly:  `ncaa update` at $NIGHTLY_HOUR (UTC) - new box scores, ledger, refit
set -uo pipefail
cd /app
SEASON="${NCAA_SEASON:-2026}"
LIVE="${NCAA_LIVE_SEASON:-2027}"
FIT="${BOOTSTRAP_FIT:-vi}"          # vi: ~10 min; nuts: ~1-3 h (exact MCMC)
NIGHTLY_FIT="${NIGHTLY_FIT:-vi}"
HOUR="${NIGHTLY_HOUR:-10}"
WATCH_EVERY="${WATCH_EVERY:-120}"

log() { echo "[$(date -u +%FT%TZ)] $*"; }

if [ ! -f "artifacts/$SEASON/summary.json" ]; then
  log "first start: building the $SEASON models (fit method: $FIT)"
  python -m ncaa fetch --season "$SEASON" &&
  python -m ncaa fit --season "$SEASON" --method "$FIT" &&
  python -m ncaa simulate --season "$SEASON" || { log "build failed; retrying in 10 min"; sleep 600; exec "$0"; }
  log "build finished"
fi

log "catching up the $LIVE season"
python -m ncaa update --season "$LIVE" --refit none || log "catch-up failed (will retry tonight)"

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
done
