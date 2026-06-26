"""
portal.py — TrainOCR by Sanchaya (Flask / Python backend)

Alternative to the Node.js server.js. Serves the same public/index.html
and exposes a compatible REST API for all pipeline operations.

Usage:
    pip install -r requirements-portal.txt
    python portal.py            # development
    gunicorn -w 2 portal:app   # production

Deploy:
    Reverse-proxy with nginx → trainocr.sanchaya.net
"""

import os
import subprocess
import threading
import queue
import glob
import yaml

from flask import Flask, jsonify, request, send_from_directory, Response, stream_with_context
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__, static_folder="public", static_url_path="")

BASE       = os.path.dirname(os.path.abspath(__file__))
FONTS_DIR  = os.path.join(BASE, "fonts")
RENDERED   = os.path.join(BASE, "rendered")
LSTMF_DIR  = os.path.join(BASE, "lstmf")
OUTPUT_DIR = os.path.join(BASE, "output")
BEST_DIR   = os.path.join(BASE, "best")
SCAN_DIR   = os.path.join(BASE, "scan-input")
LOGS_DIR   = os.path.join(BASE, "logs")
TESSDATA   = os.path.join(BASE, "tessdata_best")
TEST_DIR   = os.path.join(BASE, "test-images")

for d in [FONTS_DIR, RENDERED, LSTMF_DIR, OUTPUT_DIR, BEST_DIR, SCAN_DIR, LOGS_DIR, TESSDATA, TEST_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Static frontend ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("public", "index.html")

@app.route("/tessdata/<path:filename>")
def tessdata(filename):
    return send_from_directory(TESSDATA, filename)

# ── Status ────────────────────────────────────────────────────────────────────

@app.route("/api/status")
def status():
    try:
        ver = subprocess.check_output(["tesseract", "--version"], stderr=subprocess.STDOUT).decode().split("\n")[0]
    except Exception:
        ver = "tesseract not found"

    fonts_yml = os.path.join(BASE, "fonts.yml")
    fonts = []
    if os.path.exists(fonts_yml):
        with open(fonts_yml) as f:
            fonts = yaml.safe_load(f).get("fonts", [])

    return jsonify({
        "tesseract": ver,
        "fonts": [fo["name"] for fo in fonts],
        "rendered": len(glob.glob(os.path.join(RENDERED, "*.tif"))),
        "lstmf":    len(glob.glob(os.path.join(LSTMF_DIR, "*.lstmf"))),
        "checkpoints": len(glob.glob(os.path.join(OUTPUT_DIR, "*.checkpoint"))),
        "traineddata": [os.path.basename(p) for p in glob.glob(os.path.join(BEST_DIR, "*.traineddata"))],
    })

# ── Pipeline steps ────────────────────────────────────────────────────────────

def _run_stream(cmd, log_file):
    """Run a shell command, stream stdout/stderr as SSE, write to log file."""
    def generate():
        with open(log_file, "a") as lf:
            proc = subprocess.Popen(
                cmd, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1
            )
            for line in proc.stdout:
                lf.write(line)
                yield f"data: {line.rstrip()}\n\n"
            proc.wait()
            status = "done" if proc.returncode == 0 else f"error:{proc.returncode}"
            yield f"data: [STATUS] {status}\n\n"
    return Response(stream_with_context(generate()), mimetype="text/event-stream")

@app.route("/api/run/render", methods=["POST"])
def run_render():
    script = os.path.join(BASE, "scripts", "01-prep-base.sh")
    log    = os.path.join(LOGS_DIR, "render.log")
    return _run_stream(f"bash {script}", log)

@app.route("/api/run/lstmf", methods=["POST"])
def run_lstmf():
    script = os.path.join(BASE, "scripts", "02-make-lstmf.sh")
    log    = os.path.join(LOGS_DIR, "lstmf.log")
    return _run_stream(f"bash {script}", log)

@app.route("/api/run/train", methods=["POST"])
def run_train():
    script = os.path.join(BASE, "scripts", "03-train.sh")
    log    = os.path.join(LOGS_DIR, "training.log")
    return _run_stream(f"bash {script}", log)

@app.route("/api/run/package", methods=["POST"])
def run_package():
    data       = request.get_json() or {}
    checkpoint = data.get("checkpoint", "")
    script     = os.path.join(BASE, "scripts", "04-package.sh")
    log        = os.path.join(LOGS_DIR, "package.log")
    return _run_stream(f"bash {script} {checkpoint}", log)

@app.route("/api/run/test", methods=["POST"])
def run_test():
    script = os.path.join(BASE, "scripts", "05-test.sh")
    log    = os.path.join(LOGS_DIR, "test.log")
    return _run_stream(f"bash {script}", log)

# ── Checkpoints ───────────────────────────────────────────────────────────────

@app.route("/api/checkpoints")
def checkpoints():
    cps = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.checkpoint")))
    result = []
    for cp in cps:
        stat = os.stat(cp)
        result.append({"name": os.path.basename(cp), "size": stat.st_size, "mtime": stat.st_mtime})
    return jsonify(result)

# ── Scan upload ───────────────────────────────────────────────────────────────

@app.route("/api/upload/scan", methods=["POST"])
def upload_scan():
    saved = []
    for f in request.files.getlist("files"):
        name = secure_filename(f.filename)
        dest = os.path.join(SCAN_DIR, name)
        f.save(dest)
        # Convert to greyscale TIFF if image
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            img = Image.open(dest).convert("L")
            tif = os.path.splitext(dest)[0] + ".tif"
            img.save(tif)
            os.remove(dest)
            saved.append(os.path.basename(tif))
        else:
            saved.append(name)
    return jsonify({"saved": saved})

# ── Log streaming ─────────────────────────────────────────────────────────────

@app.route("/api/log/stream")
def log_stream():
    log_file = os.path.join(LOGS_DIR, "training.log")
    def generate():
        if not os.path.exists(log_file):
            yield "data: (no training log yet)\n\n"
            return
        with open(log_file) as f:
            for line in f:
                yield f"data: {line.rstrip()}\n\n"
    return Response(stream_with_context(generate()), mimetype="text/event-stream")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") == "development"
    print(f"TrainOCR Flask portal → http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
