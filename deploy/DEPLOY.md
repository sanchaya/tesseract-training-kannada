# Deploying TrainOCR to trainocr.sanchaya.net

`deploy/deploy.sh` runs on your laptop. It rsyncs this checkout to the
server over SSH and runs `deploy/server-install.sh` there as root.

## Server requirements

- Ubuntu 22.04/24.04 (or Debian), nginx allowed to be already installed
  and serving other sites — only a `trainocr` site is added
- An SSH user with passwordless `sudo`
- DNS A record for `trainocr.sanchaya.net` pointing at the server
  (needed before TLS can be issued)

## First deploy

```bash
cp deploy/deploy.env.example deploy/deploy.env   # set DEPLOY_HOST etc.
./deploy/deploy.sh --setup --data
```

`--setup` installs Tesseract 5 + training tools, Node 20, a Python venv
(Pillow, PyYAML, uharfbuzz, freetype-py, numpy), Chrome for Puppeteer,
a `trainocr` systemd service bound to `127.0.0.1:3000`, the nginx site
with basic auth (you're prompted for the password), and a Let's Encrypt
certificate. `--data` pushes fonts, `tessdata_best`, corpus text, `best/`
and `test-images/`.

## Every deploy after that

```bash
./deploy/deploy.sh             # sync code, reinstall deps if lockfiles changed, restart
./deploy/deploy.sh --dry-run   # see what would be transferred
./deploy/deploy.sh --data      # also push new fonts / base models
./deploy/deploy.sh --password  # change the basic-auth password
```

The working tree is synced as it is, uncommitted changes included.
Server-side data (`rendered/`, `lstmf/`, `output/`, `best/`, `logs/`, …)
is never deleted by a deploy, and `--data` skips files that are newer on
the server.

A running training or render job survives a deploy: the unit uses
`KillMode=process`, so only the portal restarts and it re-finds the job
through `output/.job.lock`.

## On the server

```bash
sudo systemctl status trainocr
sudo journalctl -u trainocr -f
tail -f /opt/trainocr/training.log
tail -f /var/log/nginx/trainocr.access.log
```

The nginx site is rendered from `deploy/nginx.conf.template`. Edit the
template and re-run `--setup`; hand edits to
`/etc/nginx/sites-available/trainocr` are overwritten.

`rendered/` and `scan-input/` grow large — watch disk usage.
