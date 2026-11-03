#!/usr/bin/env bash
# One-shot deploy of the NCAA Bayesian bracket to a Linux VPS (Ubuntu/Debian), with HTTPS.
#
#   First time (from a clone of the repo):
#     sudo ./deploy/deploy.sh ncaa.example.com you@example.com
#   or without a clone yet:
#     curl -fsSL <raw url of this file> | sudo bash -s -- ncaa.example.com you@example.com https://github.com/<you>/NCAA.git
#
#   Later, to ship new code:
#     sudo ./deploy/deploy.sh update
#
# Before running: point the domain's DNS A (and AAAA) record at this server.
# Requirements: 2+ vCPUs and 4 GB RAM recommended (the model fit is the heavy part).
set -euo pipefail

INSTALL_DIR="${INSTALL_DIR:-/opt/ncaa}"
BRANCH="${BRANCH:-master}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { echo "error: $*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "run as root (sudo)"

# --- where the code lives -----------------------------------------------------
here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." 2>/dev/null && pwd || true)"
if [ -n "$here" ] && [ -f "$here/docker-compose.yml" ]; then
  APP_DIR="$here"
else
  APP_DIR="$INSTALL_DIR"
fi

compose() { docker compose --project-directory "$APP_DIR" "$@"; }

if [ "${1:-}" = "update" ]; then
  say "pulling the latest code"
  git -C "$APP_DIR" pull --ff-only
  say "rebuilding and restarting"
  compose up -d --build
  compose ps
  exit 0
fi

DOMAIN="${1:-}"; EMAIL="${2:-}"; REPO="${3:-}"
[ -n "$DOMAIN" ] && [ -n "$EMAIL" ] || die "usage: deploy.sh <domain> <email for Let's Encrypt> [git repo url]"

# --- Docker -------------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  say "installing Docker"
  curl -fsSL https://get.docker.com | sh
fi
docker compose version >/dev/null 2>&1 || die "Docker Compose plugin missing (apt install docker-compose-plugin)"
systemctl enable --now docker >/dev/null 2>&1 || true

# --- code ---------------------------------------------------------------------
if [ ! -f "$APP_DIR/docker-compose.yml" ]; then
  [ -n "$REPO" ] || die "no checkout at $APP_DIR; pass the git repo url as the third argument"
  command -v git >/dev/null 2>&1 || { apt-get update -y && apt-get install -y git; }
  say "cloning $REPO ($BRANCH) into $APP_DIR"
  git clone --branch "$BRANCH" "$REPO" "$APP_DIR"
fi

# --- firewall -----------------------------------------------------------------
if command -v ufw >/dev/null 2>&1 && ufw status | grep -q "Status: active"; then
  say "opening ports 80 and 443"
  ufw allow 80/tcp >/dev/null; ufw allow 443/tcp >/dev/null; ufw allow 443/udp >/dev/null
fi

# --- DNS sanity check (HTTPS certificates need it) ----------------------------
ip_here="$(curl -fsS4 https://api.ipify.org 2>/dev/null || true)"
ip_dns="$(getent ahostsv4 "$DOMAIN" 2>/dev/null | awk 'NR==1{print $1}')"
if [ -n "$ip_here" ] && [ "$ip_dns" != "$ip_here" ]; then
  echo "warning: $DOMAIN resolves to '${ip_dns:-nothing}', but this server is $ip_here."
  echo "         Point the A record at $ip_here; Caddy keeps retrying the certificate until it does."
fi

# --- settings -----------------------------------------------------------------
env_file="$APP_DIR/.env"
touch "$env_file"
set_env() { grep -q "^$1=" "$env_file" && sed -i "s|^$1=.*|$1=$2|" "$env_file" || echo "$1=$2" >> "$env_file"; }
set_env DOMAIN "$DOMAIN"
set_env ACME_EMAIL "$EMAIL"
grep -q '^NCAA_SEASON=' "$env_file"      || set_env NCAA_SEASON 2026        # bracket season shown by the bracket pages
grep -q '^NCAA_LIVE_SEASON=' "$env_file" || set_env NCAA_LIVE_SEASON 2027   # season tracked live
grep -q '^BOOTSTRAP_FIT=' "$env_file"    || set_env BOOTSTRAP_FIT vi        # vi (~10 min) or nuts (exact, 1-3 h)
grep -q '^NIGHTLY_FIT=' "$env_file"      || set_env NIGHTLY_FIT vi
grep -q '^NIGHTLY_HOUR=' "$env_file"     || set_env NIGHTLY_HOUR 10         # UTC hour of the nightly update
chmod 600 "$env_file"

# --- go -----------------------------------------------------------------------
say "building and starting (web, worker, caddy)"
compose up -d --build
compose ps

cat <<EOF

Deployed. https://$DOMAIN will serve the app once the certificate is issued (a minute after DNS points here).
The first start builds the models in the worker (fetch, fit, simulate: ~20-90 minutes); until then
the site shows a "building" page. Follow it with:

  docker compose --project-directory $APP_DIR logs -f worker

Ship new code later with:   sudo $APP_DIR/deploy/deploy.sh update
Settings (seasons, fit method, nightly hour) live in $env_file.
EOF
