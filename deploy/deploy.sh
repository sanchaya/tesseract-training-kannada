#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# deploy/deploy.sh — push TrainOCR from this checkout to a server
# and (re)start it behind nginx.
#
#   ./deploy/deploy.sh --setup     first time: install packages, systemd
#                                  unit, nginx site, basic auth, TLS
#   ./deploy/deploy.sh             every time after: sync code, restart
#   ./deploy/deploy.sh --data      also push fonts, tessdata_best, corpus
#                                  text, best/ and test-images (never
#                                  overwrites newer files on the server)
#   ./deploy/deploy.sh --password  set a new basic-auth password
#   ./deploy/deploy.sh --dry-run   show what rsync would transfer
#
# Settings: deploy/deploy.env (copy deploy/deploy.env.example).
# Syncs the working tree as it is — uncommitted changes included.
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$ROOT/deploy/deploy.env"
[ -f "$ENV_FILE" ] || { echo "Missing $ENV_FILE — copy deploy/deploy.env.example and fill it in." >&2; exit 1; }
# shellcheck disable=SC1090
. "$ENV_FILE"
: "${DEPLOY_HOST:?set in deploy.env}" "${DOMAIN:?}" "${APP_DIR:?}" "${APP_USER:?}" "${PORT:?}"
SSH_OPTS="${SSH_OPTS:-}"

MODE=update DATA=no DRY=() NEW_PASS=no
for a in "$@"; do
  case "$a" in
    --setup)    MODE=setup ;;
    --data)     DATA=yes ;;
    --password) MODE=setup NEW_PASS=yes ;;
    --dry-run)  DRY=(--dry-run) ;;
    -h|--help)  sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 1 ;;
  esac
done

# shellcheck disable=SC2086
ssh_() { ssh $SSH_OPTS "$DEPLOY_HOST" "$@"; }
RSYNC=(rsync -az -e "ssh $SSH_OPTS" --rsync-path="sudo mkdir -p $APP_DIR && sudo rsync" ${DRY[@]+"${DRY[@]}"})

echo "━━ Deploying to $DEPLOY_HOST:$APP_DIR ($DOMAIN) — $MODE"

# ── Code ────────────────────────────────────────────────────────
# Everything else in the tree is data the server generates or owns
# (rendered/, lstmf/, output/, …). Excluded paths are also protected
# from --delete, so server-side training state survives a deploy.
"${RSYNC[@]}" --delete \
  --exclude=/.git/ --exclude=/node_modules/ --exclude=/.venv/ --exclude=/.deploy-state/ \
  --exclude=/_private/ --exclude=/.autopilot/ --exclude=/deploy/deploy.env \
  --exclude=/rendered/ --exclude=/lstmf/ --exclude=/output/ --exclude=/best/ \
  --exclude=/scan-input/ --exclude=/tmp/ --exclude=/logs/ --exclude='*.log' \
  --exclude=/fonts/ --exclude=/tessdata_best/ --exclude=/tessdata_expanded/ \
  --exclude=/classical-corpus-kannada/ --exclude=/inventory/ --exclude=/reports/ \
  --exclude=/test-images/ --exclude=/models/ \
  --exclude=/corpus/cache/ --exclude=/corpus/a5-pages/ --exclude=/corpus/coverage/ \
  --exclude=/corpus/kan_corpus.txt --exclude=/corpus/kan_specimen.txt \
  --exclude=/corpus/raw_kannada.txt --exclude=/corpus/remediation.txt \
  --exclude=/a5_render_jobs.json --exclude='*.pptx' --exclude=/ocr-verification-report.html \
  --exclude='/test_*.js' --exclude=/create_presentation.js \
  --exclude=/confirm.txt --exclude=/result.txt --exclude=/tsv.txt --exclude=/fonts.yml.bak \
  --exclude=__pycache__/ --exclude=.DS_Store \
  "$ROOT/" "$DEPLOY_HOST:$APP_DIR/"

# ── Data (opt-in) ───────────────────────────────────────────────
if [ "$DATA" = yes ]; then
  echo "━━ Pushing data (fonts, tessdata_best, corpus text, best, test-images)"
  cd "$ROOT"
  data=()
  for p in fonts tessdata_best best test-images \
           corpus/kan_corpus.txt corpus/kan_specimen.txt corpus/raw_kannada.txt; do
    [ -e "$p" ] && data+=("$p")
  done
  # --relative keeps corpus/x.txt at corpus/x.txt; --update skips any
  # file that is newer on the server.
  "${RSYNC[@]}" --relative --update --exclude=.DS_Store --exclude=.git/ \
    "${data[@]}" "$DEPLOY_HOST:$APP_DIR/"
fi

[ ${#DRY[@]} -gt 0 ] && { echo "━━ Dry run — nothing installed or restarted."; exit 0; }

# ── Basic-auth password ─────────────────────────────────────────
PASS=""
if [ "$MODE" = setup ] && [ -n "${BASIC_AUTH_USER:-}" ]; then
  if [ "$NEW_PASS" = yes ] || ! ssh_ "sudo test -f /etc/nginx/trainocr.htpasswd"; then
    read -rsp "Basic-auth password for '$BASIC_AUTH_USER': " PASS; echo
    read -rsp "Again: " PASS2; echo
    [ "$PASS" = "$PASS2" ] && [ -n "$PASS" ] || { echo "Passwords empty or don't match." >&2; exit 1; }
  fi
fi

# ── Install / restart on the server ─────────────────────────────
# Settings travel on stdin into a root-only temp file rather than on the
# command line, so the password never shows up in `ps`.
REMOTE_ENV="$(printf '%s=%q\n' \
  APP_DIR "$APP_DIR" APP_USER "$APP_USER" PORT "$PORT" DOMAIN "$DOMAIN" \
  BASIC_AUTH_USER "${BASIC_AUTH_USER:-}" BASIC_AUTH_PASS "$PASS" \
  ENABLE_TLS "${ENABLE_TLS:-no}" CERTBOT_EMAIL "${CERTBOT_EMAIL:-}")"

printf '%s\n' "$REMOTE_ENV" | ssh_ "sudo bash -c '
  set -e
  f=\$(mktemp); chmod 600 \$f; cat > \$f
  set -a; . \$f; rm -f \$f; set +a
  bash $APP_DIR/deploy/server-install.sh $MODE
'"

scheme=http; [ "${ENABLE_TLS:-no}" = yes ] && scheme=https
echo "━━ Done: $scheme://$DOMAIN"
