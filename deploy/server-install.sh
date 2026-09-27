#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# deploy/server-install.sh — runs ON THE SERVER, as root.
# Called by deploy/deploy.sh; not meant to be run from a laptop.
#
#   server-install.sh setup    system packages, user, venv, systemd,
#                              nginx site, basic auth, TLS, then update
#   server-install.sh update   node/python deps (only when their lock
#                              files changed), restart, health check
#
# Both modes are idempotent. Settings come from the environment:
# APP_DIR APP_USER PORT DOMAIN BASIC_AUTH_USER BASIC_AUTH_PASS
# ENABLE_TLS CERTBOT_EMAIL
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

MODE="${1:?usage: server-install.sh setup|update}"
: "${APP_DIR:?}" "${APP_USER:?}" "${PORT:?}" "${DOMAIN:?}"
BASIC_AUTH_USER="${BASIC_AUTH_USER:-}"
BASIC_AUTH_PASS="${BASIC_AUTH_PASS:-}"
ENABLE_TLS="${ENABLE_TLS:-no}"
CERTBOT_EMAIL="${CERTBOT_EMAIL:-}"

APP_HOME="/home/$APP_USER"
VENV="$APP_DIR/.venv"
STATE="$APP_DIR/.deploy-state"
SITE=/etc/nginx/sites-available/trainocr
HTPASSWD=/etc/nginx/trainocr.htpasswd

log()  { echo "  → $*"; }
die()  { echo "  ✗ $*" >&2; exit 1; }
as_app() { sudo -u "$APP_USER" -H env PATH="$VENV/bin:/usr/local/bin:/usr/bin:/bin" "$@"; }

[ "$(id -u)" = 0 ] || die "must run as root"

setup() {
  . /etc/os-release
  [ "$ID" = ubuntu ] || [ "$ID" = debian ] || die "only Ubuntu/Debian is supported (found $ID)"

  log "Installing system packages"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  # tesseract-ocr ships the training tools too (lstmtraining,
  # combine_tessdata, combine_lang_model, unicharset_extractor, …).
  apt-get install -y -qq --no-install-recommends \
    nginx apache2-utils \
    tesseract-ocr tesseract-ocr-kan tesseract-ocr-script-knda \
    libtesseract-dev libleptonica-dev \
    python3 python3-venv python3-dev build-essential \
    libharfbuzz-dev libfreetype-dev libfontconfig1 fontconfig fonts-liberation \
    imagemagick parallel git curl ca-certificates rsync
  [ "$ENABLE_TLS" = yes ] && apt-get install -y -qq --no-install-recommends certbot python3-certbot-nginx

  for t in tesseract lstmtraining combine_tessdata combine_lang_model unicharset_extractor merge_unicharsets; do
    command -v "$t" >/dev/null || die "$t not found after installing tesseract-ocr"
  done

  # Node 20 — the distro's node is too old on 22.04 and fine on 24.04,
  # but pin one version everywhere so behaviour doesn't differ by host.
  if ! node -v 2>/dev/null | grep -q '^v2[0-9]'; then
    log "Installing Node.js 20"
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - >/dev/null
    apt-get install -y -qq nodejs
  fi

  if ! id "$APP_USER" >/dev/null 2>&1; then
    log "Creating user $APP_USER"
    useradd --system --create-home --home-dir "$APP_HOME" --shell /usr/sbin/nologin "$APP_USER"
  fi
  mkdir -p "$APP_DIR"
  chown -R "$APP_USER:$APP_USER" "$APP_DIR"

  if [ ! -x "$VENV/bin/python3" ]; then
    log "Creating Python venv"
    as_app python3 -m venv "$VENV"
  fi

  log "Writing systemd unit"
  cat > /etc/systemd/system/trainocr.service <<EOF
[Unit]
Description=TrainOCR portal
After=network.target

[Service]
Type=simple
User=$APP_USER
Group=$APP_USER
WorkingDirectory=$APP_DIR
Environment=NODE_ENV=production
Environment=PORT=$PORT
Environment=HOST=127.0.0.1
Environment=HOME=$APP_HOME
# Scripts call plain python3 — put the venv first so they get its packages.
Environment=PATH=$VENV/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/usr/bin/node server.js
Restart=on-failure
RestartSec=3
# Training and render jobs are meant to outlive the portal (see JOB_LOCK in
# server.js). Kill only node on stop/restart, so a deploy doesn't abort a
# multi-hour lstmtraining run.
KillMode=process

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  systemctl enable trainocr >/dev/null

  log "Writing nginx site for $DOMAIN"
  local auth="    # basic auth disabled (BASIC_AUTH_USER empty)"
  if [ -n "$BASIC_AUTH_USER" ]; then
    auth="    auth_basic           \"TrainOCR\";
    auth_basic_user_file $HTPASSWD;"
    if [ -n "$BASIC_AUTH_PASS" ]; then
      htpasswd -bcB "$HTPASSWD" "$BASIC_AUTH_USER" "$BASIC_AUTH_PASS" 2>/dev/null
      chown root:www-data "$HTPASSWD"; chmod 640 "$HTPASSWD"
    fi
    [ -f "$HTPASSWD" ] || die "basic auth enabled but no password given and $HTPASSWD missing"
  fi
  local tpl; tpl="$(cat "$APP_DIR/deploy/nginx.conf.template")"
  tpl="${tpl//__DOMAIN__/$DOMAIN}"
  tpl="${tpl//__PORT__/$PORT}"
  tpl="${tpl//__AUTH__/$auth}"
  printf '%s\n' "$tpl" > "$SITE"
  ln -sf "$SITE" /etc/nginx/sites-enabled/trainocr
  nginx -t -q || die "nginx config test failed"
  systemctl reload nginx

  if [ "$ENABLE_TLS" = yes ]; then
    [ -n "$CERTBOT_EMAIL" ] || die "ENABLE_TLS=yes needs CERTBOT_EMAIL"
    log "Obtaining / installing TLS certificate"
    # --keep-until-expiring re-installs an existing cert into the freshly
    # rewritten site file instead of requesting a new one.
    certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "$CERTBOT_EMAIL" \
      --redirect --keep-until-expiring
  fi
}

# Re-run an install step only when its input changed since the last deploy.
changed() {  # changed <file> <key>
  local sum; sum="$(sha256sum "$APP_DIR/$1" | cut -d' ' -f1)"
  [ "$(cat "$STATE/$2" 2>/dev/null)" != "$sum" ]
}
mark() { sha256sum "$APP_DIR/$1" | cut -d' ' -f1 > "$STATE/$2"; }

update() {
  # rsync-as-root keeps the laptop's uid. Hand everything to the app user,
  # but skip the generated trees — they hold millions of files the app
  # created itself, and walking them would make every deploy crawl.
  find "$APP_DIR" \( -path "$APP_DIR/rendered" -o -path "$APP_DIR/lstmf" \
    -o -path "$APP_DIR/output" -o -path "$APP_DIR/node_modules" \) -prune \
    -o ! -user "$APP_USER" -exec chown -h "$APP_USER:$APP_USER" {} +
  as_app mkdir -p "$STATE" "$APP_DIR/logs" "$APP_DIR/output"
  cd "$APP_DIR"

  if changed package-lock.json npm || [ ! -d node_modules ]; then
    log "npm ci"
    as_app npm ci --omit=dev --no-audit --no-fund
    # npm ci's postinstall puts Chrome in ~/.cache/puppeteer, where server.js
    # looks for it. Its shared libraries need apt, hence root here.
    log "Installing Chrome system libraries"
    PUPPETEER_CACHE_DIR="$APP_HOME/.cache/puppeteer" \
      ./node_modules/.bin/puppeteer browsers install chrome --install-deps >/dev/null
    chown -R "$APP_USER:$APP_USER" "$APP_HOME/.cache"
    mark package-lock.json npm
  fi

  if changed requirements.txt pip; then
    log "pip install"
    as_app "$VENV/bin/pip" install -q --upgrade pip
    # requirements.txt covers Pillow/PyYAML/requests; the shaping renderer
    # (corpus/shaping_render.py) additionally needs these.
    as_app "$VENV/bin/pip" install -q -r requirements.txt uharfbuzz freetype-py numpy
    mark requirements.txt pip
  fi

  log "Restarting trainocr"
  systemctl restart trainocr

  for _ in $(seq 1 20); do
    if curl -fsS "http://127.0.0.1:$PORT/api/status" >/dev/null 2>&1; then
      log "Healthy on 127.0.0.1:$PORT"
      return 0
    fi
    sleep 1
  done
  journalctl -u trainocr -n 40 --no-pager >&2
  die "portal did not answer /api/status within 20s"
}

case "$MODE" in
  setup)  setup; update ;;
  update) update ;;
  *)      die "unknown mode: $MODE" ;;
esac
