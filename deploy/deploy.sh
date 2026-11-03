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
# Works on a fresh server (a bundled Caddy takes ports 80/443 and gets the certificate) and on a server that
# already hosts other sites: if something else owns 80/443, the app listens on 127.0.0.1:$WEB_PORT and is put
# behind that server (nginx is configured automatically, with certbot for HTTPS; for Apache or a host Caddy
# the script prints the snippet to add). Force the choice with PROXY=caddy or PROXY=host.
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
# Remove our bundled Caddy container, if any (works whatever profiles are active).
drop_caddy() {
  docker ps -aq --filter "label=com.docker.compose.project=$(basename "$APP_DIR")" \
    --filter label=com.docker.compose.service=caddy | xargs -r docker rm -f >/dev/null
}

if [ "${1:-}" = "update" ]; then
  say "pulling the latest code"
  git -C "$APP_DIR" pull --ff-only
  say "rebuilding and restarting"
  compose up -d --build --remove-orphans
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

env_file="$APP_DIR/.env"
touch "$env_file"
set_env() { if grep -q "^$1=" "$env_file"; then sed -i "s|^$1=.*|$1=$2|" "$env_file"; else echo "$1=$2" >> "$env_file"; fi; }

# --- who answers on 80/443? ---------------------------------------------------
command -v ss >/dev/null 2>&1 || { apt-get update -y && apt-get install -y iproute2; } >/dev/null 2>&1 || true
command -v ss >/dev/null 2>&1 || die "need the ss command (apt install iproute2)"
port_busy() { ss -ltnH "sport = :$1" 2>/dev/null | grep -q .; }
port_owner() { ss -ltnpH "sport = :$1" 2>/dev/null | grep -o 'users:(("[^"]*' | head -1 | cut -d'"' -f2 || true; }

drop_caddy || true                                # if our own Caddy holds 80/443, free them before looking
owner="$(port_owner 80)"; [ -n "$owner" ] || owner="$(port_owner 443)"
if [ -n "${PROXY:-}" ]; then
  mode="$PROXY"
elif port_busy 80 || port_busy 443; then
  mode=host
else
  mode=caddy
fi
[ "$mode" = caddy ] || [ "$mode" = host ] || die "PROXY must be caddy or host"

# The app's local port: keep the one in .env, else the first free one from 8000.
WEB_PORT="${WEB_PORT:-$(grep -s '^WEB_PORT=' "$env_file" | cut -d= -f2 || true)}"
if [ -z "$WEB_PORT" ]; then
  compose down >/dev/null 2>&1 || true            # our own earlier run may hold it
  WEB_PORT=8000
  while port_busy "$WEB_PORT"; do WEB_PORT=$((WEB_PORT + 1)); done
fi
upstream="http://127.0.0.1:$WEB_PORT"

# --- firewall -----------------------------------------------------------------
if [ "$mode" = caddy ] && command -v ufw >/dev/null 2>&1 && ufw status | grep -q "Status: active"; then
  say "opening ports 80 and 443"
  ufw allow 80/tcp >/dev/null; ufw allow 443/tcp >/dev/null; ufw allow 443/udp >/dev/null
fi

# --- DNS sanity check (HTTPS certificates need it) ----------------------------
ip_here="$(curl -fsS4 https://api.ipify.org 2>/dev/null || true)"
ip_dns="$(getent ahostsv4 "$DOMAIN" 2>/dev/null | awk 'NR==1{print $1}' || true)"
if [ -n "$ip_here" ] && [ "$ip_dns" != "$ip_here" ]; then
  echo "warning: $DOMAIN resolves to '${ip_dns:-nothing}', but this server is $ip_here."
  echo "         Point the A record at $ip_here; the certificate can't be issued until it does."
fi

# --- settings -----------------------------------------------------------------
set_env DOMAIN "$DOMAIN"
set_env ACME_EMAIL "$EMAIL"
grep -q '^NCAA_SEASON=' "$env_file"      || set_env NCAA_SEASON 2026        # bracket season shown by the bracket pages
grep -q '^NCAA_LIVE_SEASON=' "$env_file" || set_env NCAA_LIVE_SEASON 2027   # season tracked live
grep -q '^BOOTSTRAP_FIT=' "$env_file"    || set_env BOOTSTRAP_FIT vi        # vi (~10 min) or nuts (exact, 1-3 h)
grep -q '^NIGHTLY_FIT=' "$env_file"      || set_env NIGHTLY_FIT vi
grep -q '^NIGHTLY_HOUR=' "$env_file"     || set_env NIGHTLY_HOUR 10         # UTC hour of the nightly update
set_env WEB_PORT "$WEB_PORT"                                                 # the app listens on 127.0.0.1:WEB_PORT
if [ "$mode" = caddy ]; then set_env COMPOSE_PROFILES caddy; else set_env COMPOSE_PROFILES ""; fi
chmod 600 "$env_file"

# --- go -----------------------------------------------------------------------
say "building and starting (web, worker$([ "$mode" = caddy ] && echo ', caddy' || true))"
compose up -d --build --remove-orphans

next=""
if [ "$mode" = host ]; then
  say "ports 80/443 belong to ${owner:-another web server}; putting the app behind it"
  case "$owner" in
    nginx*)
      if [ -d /etc/nginx/sites-available ]; then conf="/etc/nginx/sites-available/ncaa-$DOMAIN"
      else conf="/etc/nginx/conf.d/ncaa-$DOMAIN.conf"; fi
      v6listen=""; [ -s /proc/net/if_inet6 ] && v6listen="    listen [::]:80;"
      if [ -e "$conf" ]; then
        echo "keeping the existing $conf"
      else
        cat > "$conf" <<NGINX
# NCAA Bayesian bracket (written by deploy/deploy.sh)
server {
    listen 80;
$v6listen
    server_name $DOMAIN;
    client_max_body_size 5m;
    location / {
        proxy_pass $upstream;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 600s;
    }
}
NGINX
        case "$conf" in /etc/nginx/sites-available/*) ln -sf "$conf" /etc/nginx/sites-enabled/ ;; esac
        if nginx -t >/dev/null 2>&1; then
          systemctl reload nginx 2>/dev/null || nginx -s reload
          echo "added $conf and reloaded nginx"
        else
          nginx -t || true
          rm -f "$conf" "/etc/nginx/sites-enabled/$(basename "$conf")"
          die "nginx rejected the new site (removed it again; your other sites are untouched)"
        fi
      fi
      if ! command -v certbot >/dev/null 2>&1; then
        say "installing certbot"
        { apt-get update -y && apt-get install -y certbot python3-certbot-nginx; } >/dev/null 2>&1 || true
      fi
      if command -v certbot >/dev/null 2>&1; then
        say "getting the HTTPS certificate"
        certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "$EMAIL" --redirect \
          || next="certbot failed (usually DNS not pointing here yet). Once it does: sudo certbot --nginx -d $DOMAIN"
      else
        next="Install certbot, then: sudo certbot --nginx -d $DOMAIN"
      fi
      ;;
    caddy*)
      next="Add this to your Caddyfile, then reload Caddy (it gets the certificate itself):

    $DOMAIN {
        reverse_proxy 127.0.0.1:$WEB_PORT
    }"
      ;;
    apache2*|httpd*)
      next="Enable the proxy modules (sudo a2enmod proxy proxy_http), add this site, reload Apache,
then get the certificate with: sudo certbot --apache -d $DOMAIN

    <VirtualHost *:80>
        ServerName $DOMAIN
        ProxyPreserveHost On
        ProxyTimeout 600
        ProxyPass / $upstream/
        ProxyPassReverse / $upstream/
    </VirtualHost>"
      ;;
    *)
      next="Point your web server's site for $DOMAIN at $upstream (reverse proxy, 10-minute read
timeout) and give it a certificate the way you do for your other sites."
      ;;
  esac
fi

compose ps

echo
echo "Deployed. The app is running on $upstream (reachable from this machine only)."
if [ "$mode" = caddy ]; then
  echo "https://$DOMAIN will serve it once the certificate is issued (a minute after DNS points here)."
elif [ -n "$next" ]; then
  printf '\nOne step left for https://%s:\n%s\n' "$DOMAIN" "$next"
else
  echo "https://$DOMAIN is served through $owner."
fi
cat <<EOF

The first start builds the models in the worker (fetch, fit, simulate: ~20-90 minutes); until then
the site shows a "building" page. Follow it with:

  docker compose --project-directory $APP_DIR logs -f worker

Ship new code later with:   sudo $APP_DIR/deploy/deploy.sh update
Settings (seasons, fit method, nightly hour, port) live in $env_file.
EOF
