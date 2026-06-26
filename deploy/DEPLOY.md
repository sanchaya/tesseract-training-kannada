# Deploying TrainOCR to trainocr.sanchaya.net

## Requirements

- Ubuntu 22.04+ server with Node.js 18+
- nginx
- PM2 (`npm install -g pm2`)
- Tesseract 5 + training tools
- Python 3.9+ with Pillow and PyYAML
- Domain DNS A-record pointing to the server

## First-time setup

```bash
# 1. Clone the repo
git clone <repo-url> /opt/trainocr
cd /opt/trainocr

# 2. Install Node.js dependencies
npm install

# 3. Install Python dependencies
pip install pillow pyyaml

# 4. Create log directory
mkdir -p logs

# 5. Set up nginx
sudo cp deploy/nginx.conf /etc/nginx/sites-available/trainocr.sanchaya.net
sudo ln -sf /etc/nginx/sites-available/trainocr.sanchaya.net \
            /etc/nginx/sites-enabled/
sudo nginx -t

# 6. Get SSL certificate
sudo certbot --nginx -d trainocr.sanchaya.net

# 7. Reload nginx
sudo systemctl reload nginx

# 8. Start the Node.js app
pm2 start ecosystem.config.js
pm2 save
pm2 startup   # follow the printed command to register with systemd
```

## Updating

```bash
cd /opt/trainocr
git pull
npm install   # if package.json changed
pm2 restart trainocr
```

## Logs

```bash
pm2 logs trainocr          # live application logs
tail -f training.log       # Tesseract training output
tail -f /var/log/nginx/trainocr.access.log
```

## Monitoring

```bash
pm2 status
pm2 monit
```

## Notes

- The training process (`lstmtraining`) runs as a subprocess, not managed by PM2. If the server restarts, re-run the train step from the portal.
- `scan-input/` and `rendered/` can grow large. Monitor disk usage.
- The `/tessdata/` route streams `kan_hist.traineddata` (15–25 MB) to the browser for Tesseract.js. Allow adequate bandwidth.
- Access control: if this portal should be internal-only, uncomment the `allow/deny` block in `nginx.conf` and restrict to your team's IP range.
