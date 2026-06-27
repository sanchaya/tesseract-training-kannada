#!/bin/bash
# Run this once to cache the Sanchaya logo locally
curl -L "https://pada.sanchaya.net/images/sanchaya-logo.png" -o "$(dirname "$0")/sanchaya-logo.png" && \
echo "✓ Logo saved to public/sanchaya-logo.png"
