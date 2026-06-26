# ══════════════════════════════════════════════════════════════════════
#  TrainOCR — Dockerfile
#  Builds a single image with:
#    • Ubuntu 22.04 base
#    • Tesseract 5 + all training tools (lstmtraining, combine_tessdata…)
#    • Python 3 + Pillow + PyYAML  (corpus build & image rendering)
#    • Node.js 20 LTS              (TrainOCR web portal)
#
#  Usage (development):
#    docker compose up
#
#  Usage (standalone):
#    docker build -t trainocr .
#    docker run -p 3000:3000 -v $(pwd)/data:/app/data trainocr
# ══════════════════════════════════════════════════════════════════════

FROM ubuntu:22.04

# ── Build-time env ────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    NODE_ENV=production \
    PORT=3000

# ── System packages ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Tesseract 5 + training tools
    tesseract-ocr \
    tesseract-ocr-kan \
    libtesseract-dev \
    libleptonica-dev \
    tesseract-ocr-script-knda \
    # Build tools needed for some npm native modules
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    # Font rendering dependencies
    fontconfig \
    libfreetype6 \
    # Image processing for Python (Pillow)
    libjpeg-turbo8 \
    libpng-dev \
    # Utilities
    git \
    curl \
    wget \
    ca-certificates \
    unzip \
 && rm -rf /var/lib/apt/lists/*

# ── Tesseract training tools (not in standard apt package) ─────────
# Install from source to get lstmtraining, combine_tessdata, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    autoconf automake libtool pkg-config \
    libicu-dev libpango1.0-dev libcairo2-dev \
 && rm -rf /var/lib/apt/lists/*

# Try apt training tools first (Ubuntu 22 has them in universe)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr-all \
 && rm -rf /var/lib/apt/lists/* || true

# ── Node.js 20 LTS ────────────────────────────────────────────────
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get install -y nodejs \
 && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# ── App directory ─────────────────────────────────────────────────
WORKDIR /app

# ── Node.js dependencies (cached layer) ──────────────────────────
COPY package.json package-lock.json* ./
RUN npm ci --omit=dev

# ── Application source ────────────────────────────────────────────
COPY server.js        ./
COPY fonts.yml        ./
COPY public/          ./public/
COPY corpus/          ./corpus/
COPY scripts/         ./scripts/
COPY docs/            ./docs/

# ── Runtime directories (will be mounted as volumes in production) ─
RUN mkdir -p \
    tessdata_best \
    fonts \
    rendered \
    lstmf \
    output \
    best \
    scan-input \
    logs \
    test-images \
 && chmod +x scripts/*.sh

# ── Expose port ────────────────────────────────────────────────────
EXPOSE 3000

# ── Health check ──────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:3000/api/status || exit 1

# ── Entry point ───────────────────────────────────────────────────
CMD ["node", "server.js"]
