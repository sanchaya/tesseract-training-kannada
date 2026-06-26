// PM2 configuration for TrainOCR — trainocr.sanchaya.net
// Usage:
//   npm install -g pm2
//   pm2 start ecosystem.config.js
//   pm2 save          # persist across reboots
//   pm2 startup       # register with systemd/launchd

module.exports = {
  apps: [{
    name:         "trainocr",
    script:       "server.js",
    cwd:          __dirname,
    instances:    1,                // single instance — training state is on disk
    exec_mode:    "fork",
    env: {
      NODE_ENV: "production",
      PORT:     3000,
    },
    error_file:   "logs/pm2-error.log",
    out_file:     "logs/pm2-out.log",
    merge_logs:   true,
    log_date_format: "YYYY-MM-DD HH:mm:ss",
    restart_delay: 3000,
    max_restarts:  10,
    watch:         false,          // don't watch in production
    kill_timeout:  5000,
  }],
};
